import os
import torch
from torch import nn
from tqdm import tqdm
from typing import Optional
from torch.utils.data import DataLoader



class CentralizedBaselineTrainer:
    """
    A trainer class for centralized training of vision transformer models,
    using a supervised classification setting.
    """

    def __init__(self, 
                 model: Optional[nn.Module] = None,
                 loss_fn: Optional[nn.Module] = None,
                 optimizer_class: Optional[torch.optim.Optimizer] = None,
                 lr: float = 0.001,
                 momentum: float = 0.9,
                 weight_decay: float = 0.0,
                 epochs: int = 10):
        """
        Initialize the trainer with model, loss function, optimizer, and device.

        Args:
            model (nn.Module, optional): The model to train. If None, loads DINO ViT-S/16.
            loss_fn (nn.Module, optional): Loss function. Defaults to CrossEntropyLoss.
            optimizer_class (Optimizer, optional): Optimizer class to use. Defaults to SGD.
            lr (float): Learning rate.
            momentum (float): Momentum (used in SGD).
            weight_decay (float): Weight decay.
            epochs (int): Number of training epochs.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        if model is None:
            self.model = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')
            print("Loaded default DINO ViT-S/16 model.")
            print("Model architecture:", self.model)
        else:
            self.model = model

        self.model.to(self.device)
        print(f"Model moved to {self.device}")
        print(f"Model type: {type(self.model)}")

        self.epochs = epochs
        print(f"Training for {self.epochs} epochs")

        self.loss_fn = loss_fn if loss_fn else nn.CrossEntropyLoss()
        print(f"Loss function initialized: {self.loss_fn}")
        
        self.optimizer = (optimizer_class if optimizer_class else torch.optim.SGD)(
            self.model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
        )
        print(f"Optimizer initialized with learning rate {lr}, momentum {momentum} and weight decay {weight_decay}")
       
        # instantiate the cosine annealing scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.epochs, eta_min=0
        )
        print(f"Scheduler initialized: {self.scheduler}")

        self.checkpoint_dir = './checkpoints'
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        print(f"Checkpoint directory: {self.checkpoint_dir}")

    def freeze_base_layers(self, layers: Optional[int] = None) -> None:
        # Freeze all base layers in the "features" section of the model (the feature extractor) by setting requires_grad=False
        for param in self.model.features.parameters():
            param.requires_grad = False

    def change_classifier_layer(self, num_classes: int) -> None:
        """
        Change the classifier layer to match the number of classes.
        
        Args:
            num_classes (int): Number of classes for classification
        """
        # DINO models use 'head' as an nn.Identity initially
        if hasattr(self.model, 'head') and isinstance(self.model.head, nn.Identity):
            # The hidden dimension is typically 384 for ViT-S
            hidden_dim = self.model.blocks[0].mlp.fc1.in_features  # or self.model.norm.normalized_shape[0]
            self.model.head = nn.Linear(hidden_dim, num_classes)
            print(f"✔ Replaced model.head (Identity) with Linear({hidden_dim}, {num_classes})") 
            print(self.model)       
        else:
            raise ValueError("Unknown classifier layer name in model.")
        
    def resume_training(self, checkpoint_path: str):
        """
        Resume training from a checkpoint.

        Args:
            checkpoint_path (str): Path to the checkpoint file.
        """

        if os.path.isfile(checkpoint_path):
            print(f"Loading checkpoint '{checkpoint_path}'")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])

            self.scheduler.load_state_dict(checkpoint['scheduler'])

            start_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_acc']
            self.epochs = checkpoint['epoch']
            print(f"✔ Loaded checkpoint '{checkpoint_path}' (epoch {checkpoint['epoch']})")
        else:
            print(f"No checkpoint found at '{checkpoint_path}'")
        

    def _train_or_eval_loop(self, loader: DataLoader, is_train: bool):
        """
        Internal shared logic for training or evaluation.

        Args:
            loader (DataLoader): Data loader for training or evaluation.
            is_train (bool): Whether this is a training loop.

        Returns:
            tuple: (average_loss, accuracy)
        """
        self.model.train() if is_train else self.model.eval()
        running_loss, correct, total = 0.0, 0, 0

        with torch.set_grad_enabled(is_train):
            pbar = tqdm(loader, desc='Training' if is_train else 'Evaluating', unit='batch')
            for batch_idx, (X, y) in enumerate(pbar):
                X, y = X.to(self.device), y.to(self.device)

                # Forward pass and loss calculation
                logits = self.model(X)
                loss = self.loss_fn(logits, y)

                if is_train: # Backpropagation
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                running_loss += loss.item()
                _, predicted = logits.max(1)
                total += y.size(0)
                correct += predicted.eq(y).sum().item()

                pbar.set_postfix({'loss': running_loss/total})

        avg_loss = running_loss / len(loader)
        accuracy = 100. * correct / total
        return avg_loss, accuracy

    def train(self, train_loader: DataLoader, val_loader: DataLoader, resume: str = None):
        """
        Train the model with the provided training and validation loaders.

        Args:
            train_loader (DataLoader): Training data loader.
            val_loader (DataLoader): Validation data loader.
        """
        print("Starting training...")
        start_epoch = 1
        best_acc = 0.0

        if resume and os.path.exists(resume):
            print(f"Resuming training from checkpoint: {resume}")
            checkpoint = torch.load(resume, map_location=self.device)
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.scheduler.load_state_dict(checkpoint['scheduler'])
            start_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_acc']
            print(f"✔ Resumed from epoch {start_epoch}, best accuracy so far: {best_acc:.2f}%")


        best_acc = 0.0
        for epoch in range(start_epoch, self.epochs + 1):
            train_loss, train_acc = self._train_or_eval_loop(train_loader, is_train=True)
            val_loss, val_acc = self._train_or_eval_loop(val_loader, is_train=False)

            print(f"Epoch {epoch}:")
            print(f"  Train -> Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
            print(f"  Val   -> Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")

            is_best = val_acc > best_acc
            if is_best:
                best_acc = val_acc

            # Update learning rate
            self.scheduler.step()
            
            self.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.state_dict(),
                'best_acc': best_acc,
                'optimizer': self.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict(),
            }, is_best, self.checkpoint_dir)


        print(f'Best Validation Accuracy: {best_acc:.2f}%')

    def test(self, test_loader: DataLoader):
        """
        Evaluate the model on the test dataset.

        Args:
            test_loader (DataLoader): Test data loader.

        Returns:
            float: Test accuracy.
        """
        test_loss, test_acc = self._train_or_eval_loop(test_loader, is_train=False)
        print(f"Test -> Loss: {test_loss:.4f}, Acc: {test_acc:.2f}%")
        return test_acc
    
    def save_checkpoint(self, state, is_best, checkpoint_dir):
        """Save checkpoint to disk"""

        filename = os.path.join(checkpoint_dir, 'checkpoint.pth')
        torch.save(state, filename)
        print(f"Checkpoint saved to {filename}")
        if is_best:
            best_filename = os.path.join(checkpoint_dir, 'model_best.pth')
            torch.save(state, best_filename)
            print(f"(Best) Checkpoint saved to {best_filename}")

    def save_model(self, path: str):
        """
        Save the model's state_dict to the specified file path.

        Args:
            path (str): File path to save the model.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")
