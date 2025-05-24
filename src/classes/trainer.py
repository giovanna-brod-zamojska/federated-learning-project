import os
import wandb
import torch
from torch import nn
from time import time
from tqdm import tqdm
from typing import Optional
from torch.utils.data import DataLoader
from src.classes.metrics import Metrics


class BaseTrainer:
    """
    Base class for training and evaluating torch classification models.
    """

    def __init__(
        self,
        num_classes: int,
        epochs: int,
        loss_fn: nn.Module,
        optimizer: nn.Module,
        use_wandb: bool = False,
        scheduler: Optional[nn.Module] = None,
        checkpoint_dir: str = "./checkpoints",
        metric_for_best_model: str = "accuracy",
    ):

        self.num_classes = num_classes
        self.use_wandb = use_wandb
        self.metric_for_best_model = metric_for_best_model

        self.epochs = epochs
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler if scheduler else None

        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        print(f"Checkpoint directory: {self.checkpoint_dir}")

    # TODO: ???? Should we freeze all base layers and implement gradual unfreezing of layers?
    def freeze_base_layers(self, layers: Optional[int] = None) -> None:

        if not self.model:
            raise ValueError(
                "Model not initialized. Please initialize the model first."
            )

        # Freeze all base layers in the "features" section of the model (the feature extractor) by setting requires_grad=False
        for param in self.model.features.parameters():
            param.requires_grad = False

    def change_classifier_layer(self, num_classes: int = None) -> None:
        """
        Change the classifier layer to match the dataset number of labels.
        """

        if not self.model:
            raise ValueError(
                "Model not initialized. Please initialize the model first."
            )

        # DINO models use 'head' as an nn.Identity initially
        if hasattr(self.model, "head") and isinstance(self.model.head, nn.Identity):
            # The hidden dimension is typically 384 for ViT-S
            hidden_dim = self.model.blocks[
                0
            ].mlp.fc1.in_features  # or self.model.norm.normalized_shape[0]
            self.model.head = nn.Linear(hidden_dim, self.num_classes)
            print(
                f"Replaced model.head (Identity) with Linear({hidden_dim}, {self.num_classes})"
            )
            print(self.model)
            self.model.to(self.device)
        else:
            raise ValueError("Unknown classifier layer name in model.")

    def _train_or_eval_loop(
        self, loader: DataLoader, is_train: bool, metrics: Optional[Metrics]
    ):
        """
        Internal shared logic for training or evaluation.
        """

        self.model.train() if is_train else self.model.eval()
        running_loss, correct, total = 0.0, 0, 0

        with torch.set_grad_enabled(is_train):
            pbar = tqdm(
                loader, desc="Training" if is_train else "Evaluating", unit="batch"
            )
            for batch_idx, (X, y) in enumerate(pbar):
                X, y = X.to(self.device), y.to(self.device)

                # Forward pass and loss calculation
                features = self.model(X)
                logits = self.model.head(features)  # TODO: check if this is correct

                loss = self.loss_fn(logits, y)

                # Backpropagation
                if is_train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                running_loss += loss.item()
                _, predicted = torch.max(logits, dim=1)

                total += y.size(0)
                correct += predicted.eq(y).sum().item()

                metrics_dict = metrics.update(predicted, y)
                pbar.set_postfix(
                    {
                        "loss": running_loss / total,
                        "f1_macro": metrics_dict["f1_macro"].item(),
                        "f1_micro": metrics_dict["f1_micro"].item(),
                        "accuracy": correct / total,
                        "accuracy": metrics_dict["accuracy"].item(),
                    }
                )

        avg_loss = running_loss / len(loader)

        return avg_loss, metrics.get_metrics()

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        resume: str = None,
    ):
        """
        Train the model with the provided training and validation loaders.

        Args:
            train_loader (DataLoader): Training data loader.
            val_loader (DataLoader): Validation data loader.
            resume (str, optional): Path to checkpoint to resume from.
        """
        if not self.model:
            raise ValueError(
                "Model not initialized. Please initialize the model first."
            )

        print("Training started")

        start_epoch, best_metric = 0, 0.0

        if resume:
            if os.path.isfile(resume):
                print(f"Loading checkpoint from {resume}")
                print(f"Resuming training from checkpoint: {resume}")
                checkpoint = torch.load(resume, map_location=self.device)

                self.model.load_state_dict(checkpoint["state_dict"])
                self.optimizer.load_state_dict(checkpoint["optimizer"])
                self.scheduler.load_state_dict(checkpoint["scheduler"])

                start_epoch = checkpoint["epoch"]
                best_metric = checkpoint["best_metric"]

                print(
                    f"✔ Resumed from epoch {start_epoch}, best  {self.metric_for_best_model} so far: {best_metric:.2f}%"
                )
            else:
                print(f"Checkpoint not found at {resume}. Starting from scratch.")

        print(f"Training. Start epoch: {start_epoch + 1}, End epoch: {self.epochs}.")

        for epoch in range(start_epoch + 1, self.epochs + 1):
            s = time()

            train_metrics = Metrics(self.device, num_classes=self.num_classes)
            val_metrics = Metrics(self.device, num_classes=self.num_classes)

            train_loss, train_metrics = self._train_or_eval_loop(
                train_loader, is_train=True, metrics=train_metrics
            )
            val_loss, val_metrics = self._train_or_eval_loop(
                val_loader, is_train=False, metrics=val_metrics
            )

            print(f"Epoch {epoch}:")
            print(
                f"  Train -> Loss: {train_loss:.4f}, {self.metric_for_best_model}: {train_metrics[self.metric_for_best_model]*100:.2f}%"
            )
            print(
                f"  Val   -> Loss: {val_loss:.4f}, {self.metric_for_best_model}: {val_metrics[self.metric_for_best_model]*100:.2f}%"
            )
            print(f"  Train Metrics: {train_metrics}")
            print(f"  Val Metrics: {val_metrics}")

            val_metric = val_metrics[self.metric_for_best_model].item()
            is_best = val_metric > best_metric
            if is_best:
                best_metric = val_metric

            # Update learning rate
            if self.scheduler is not None:
                self.scheduler.step()

            self.save_checkpoint(
                {
                    "epoch": epoch,
                    "state_dict": self.model.state_dict(),
                    "best_metric": best_metric,
                    "best_metric_name": self.metric_for_best_model,
                    "optimizer": self.optimizer.state_dict(),
                    "scheduler": self.scheduler.state_dict(),
                },
                is_best,
                self.checkpoint_dir,
            )

            if self.use_wandb:
                logs_for_wandb = {f"train_{k}": v for k, v in train_metrics.items()}
                logs_for_wandb.update({f"val_{k}": v for k, v in val_metrics.items()})
                logs_for_wandb.update(
                    {
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "elapsed_time": time() - s,
                        "epoch": epoch,
                    }
                )
                wandb.log(logs_for_wandb)

        print(
            f"Best Validation Metric - {self.metric_for_best_model}: {best_metric*100:.2f}%"
        )
        return best_metric

    def test(self, test_loader: DataLoader):
        """
        Evaluate the model on the test dataset.
        """
        if not self.model:
            raise ValueError(
                "Model not initialized. Please initialize the model first."
            )

        metrics = Metrics(self.device, num_classes=self.num_classes)

        test_loss, test_metrics = self._train_or_eval_loop(
            test_loader, is_train=False, metrics=metrics
        )

        print(
            f"  Test -> Loss: {test_loss:.4f}, {self.metric_for_best_model}: {test_metrics[self.metric_for_best_model]*100:.2f}%"
        )
        print(f"  Test Metrics: {test_metrics}")

        return test_metrics

    def save_checkpoint(self, state, is_best, checkpoint_dir):
        """Save checkpoint to disk"""

        filename = os.path.join(checkpoint_dir, "checkpoint.pth")
        torch.save(state, filename)
        print(f"Checkpoint saved to {filename}")
        if is_best:
            best_filename = os.path.join(checkpoint_dir, "model_best.pth")
            torch.save(state, best_filename)
            print(f"(Best) Checkpoint saved to {best_filename}")

    def save_model(self, path: str):
        """
        Save the model's state_dict to the specified file path.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")


class Trainer(BaseTrainer):
    """
    A trainer class for centralized training of vision transformer models,
    using a supervised classification setting.
    """

    def __init__(
        self,
        num_classes: int,
        loss_fn: Optional[nn.Module] = None,
        scheduler_type: str = "CosineAnnealingLR",
        epochs: int = 1,
        use_wandb: bool = False,
        metric_for_best_model: str = "accuracy",
        checkpoint_dir: str = "/checkpoints",
        **kwargs,
    ):
        self.model = torch.hub.load("facebookresearch/dino:main", "dino_vits16")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        print("Loaded default DINO ViT-S/16 model. Model architecture:", self.model)
        print(f"Using device: {self.device}")

        loss_fn = loss_fn if loss_fn else nn.CrossEntropyLoss()

        # project guidelines: use SGD
        # initialize optimizer with default parameters if not provided:

        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=kwargs.get("lr", 1e-3),
            momentum=kwargs.get("momentum", 0),
            weight_decay=kwargs.get("weight_decay", 0),
            nesterov=kwargs.get("nesterov", False),
        )

        # project guidelines: we suggest you use the *cosine annealing scheduler*. Which scheduler performs best?
        # Lets define a few schedulers to choose from in our experiments
        # TODO: (Scheduler defined here are just examples, need to understand which scheduler to consider)

        if scheduler_type == "CosineAnnealingLR":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs, eta_min=0
            )
        elif scheduler_type == "StepLR":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=epochs // 2, gamma=0.1
            )
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}.")

        base_trainer_args = {
            "epochs": epochs,
            "loss_fn": loss_fn,
            "optimizer": optimizer,
            "scheduler": scheduler,
            "use_wandb": use_wandb,
            "metric_for_best_model": metric_for_best_model,
            "num_classes": num_classes,
            "checkpoint_dir": checkpoint_dir,
        }

        super().__init__(**base_trainer_args)

        print("Centralized Baseline Trainer initialized.")

        self.change_classifier_layer()
