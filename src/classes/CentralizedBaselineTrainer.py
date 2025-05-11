import os
import torch
from torch import nn
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

        self.loss_fn = loss_fn if loss_fn else nn.CrossEntropyLoss()
        print(f"Loss function initialized: {self.loss_fn}")
        self.optimizer = (optimizer_class if optimizer_class else torch.optim.SGD)(
            self.model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
        )
        print(f"Optimizer initialized with learning rate {lr}, momentum {momentum} and weight decay {weight_decay}")
        self.epochs = epochs
        print(f"Training for {self.epochs} epochs")

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
        else:
            raise ValueError("Unknown classifier layer name in model.")
        

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
            for batch, (X, y) in enumerate(loader):
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

        avg_loss = running_loss / len(loader)
        accuracy = 100. * correct / total
        return avg_loss, accuracy

    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        """
        Train the model with the provided training and validation loaders.

        Args:
            train_loader (DataLoader): Training data loader.
            val_loader (DataLoader): Validation data loader.
        """
        print("Starting training...")
        best_acc = 0.0
        for epoch in range(1, self.epochs + 1):
            train_loss, train_acc = self._train_or_eval_loop(train_loader, is_train=True)
            val_loss, val_acc = self._train_or_eval_loop(val_loader, is_train=False)

            print(f"Epoch {epoch}:")
            print(f"  Train -> Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
            print(f"  Val   -> Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")

            if val_acc > best_acc:
                best_acc = val_acc

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

    def save_model(self, path: str):
        """
        Save the model's state_dict to the specified file path.

        Args:
            path (str): File path to save the model.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")
