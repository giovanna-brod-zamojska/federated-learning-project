import os
import wandb
import torch
from torch import nn
from time import time
from tqdm import tqdm
from typing import Optional
from torch.utils.data import DataLoader
from src.centralized_baseline.metrics import Metrics


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
        checkpoint_name: str = None,
        metric_for_best_model: str = "accuracy",
        unfreeze_at_epoch: int = None,
        accum_steps: int = 1,
    ):

        self.num_classes = num_classes
        self.use_wandb = use_wandb
        self.metric_for_best_model = metric_for_best_model

        self.unfreeze_at_epoch = unfreeze_at_epoch
        self.epochs = epochs
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler if scheduler else None
        self.accum_steps = accum_steps

        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_name = checkpoint_name
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        print(f"Checkpoint directory: {self.checkpoint_dir}")

    def _train_head_only(self):

        for param in self.model.parameters():
            param.requires_grad = False

        for param in self.model.head.parameters():
            param.requires_grad = True

        print("🔒 Encoder parameters frozen, only head parameters are trainable.")

    def _unfreeze_and_add_param_group(self):
        # 1. Unfreeze encoder parameters
        unfrozen = []
        for name, param in self.model.named_parameters():
            if "head" not in name and "heads" not in name:
                param.requires_grad = True
                unfrozen.append(name)

        print(f"🔓 Unfroze {len(unfrozen)} encoder parameters.")

        # 2. Collect already-optimized parameter IDs
        existing_param_ids = {
            id(p) for group in self.optimizer.param_groups for p in group["params"]
        }

        # 3. Find new parameters not in the optimizer
        new_params = [
            p
            for p in self.model.parameters()
            if p.requires_grad and id(p) not in existing_param_ids
        ]

        # 4. Add them to the optimizer
        if new_params:
            self.optimizer.add_param_group({"params": new_params})
            print(f"➕ Added {len(new_params)} new parameters to the optimizer.")
        else:
            print("ℹ️ No new parameters were added. All are already in the optimizer.")

    def change_classifier_layer(self, num_classes: int = None) -> None:
        """
        Change the classifier layer to match the dataset number of labels.
        """

        # DINO models use 'head' as an nn.Identity initially
        if hasattr(self.model, "head") and isinstance(self.model.head, nn.Identity):
            hidden_dim = self.model.blocks[0].mlp.fc1.in_features  # 384 for Vits16
            self.model.head = nn.Linear(384, num_classes)
            print(
                f"Replaced model.head (Identity) with Linear({hidden_dim}, {num_classes})"
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
                logits = self.model.head(features)
                loss = self.loss_fn(logits, y)

                # Backpropagation
                if is_train:
                    loss = loss / self.accum_steps if self.accum_steps > 1 else loss
                    loss.backward()

                    # Step optimizer and zero_grad every accum_steps or every batch if accum_steps=1 or not set
                    if (batch_idx + 1) % (
                        self.accum_steps if self.accum_steps > 1 else 1
                    ) == 0:
                        self.optimizer.step()
                        self.optimizer.zero_grad()

                # Metrics and logging
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

            print(f"\nEPOCH {epoch}")

            self.current_lr = self.optimizer.param_groups[0]["lr"]
            print(f"Using Learning Rate = {self.current_lr}")

            # Unfreeze after unfreeze_at_epoch
            if self.unfreeze_at_epoch is not None:
                if epoch == self.unfreeze_at_epoch:
                    print(
                        f"Unfreezing model parameters at epoch {self.unfreeze_at_epoch}."
                    )
                    self._unfreeze_and_add_param_group()

            train_metrics = Metrics(self.device, num_classes=self.num_classes)
            val_metrics = Metrics(self.device, num_classes=self.num_classes)

            train_loss, train_metrics = self._train_or_eval_loop(
                train_loader, is_train=True, metrics=train_metrics
            )
            val_loss, val_metrics = self._train_or_eval_loop(
                val_loader, is_train=False, metrics=val_metrics
            )

            # Update learning rate
            if self.scheduler is not None:
                self.scheduler.step()

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
                self.checkpoint_name,
            )

            if self.use_wandb:
                logs_for_wandb = {f"train_{k}": v for k, v in train_metrics.items()}
                logs_for_wandb.update({f"val_{k}": v for k, v in val_metrics.items()})
                logs_for_wandb.update(
                    {
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "epoch": epoch,
                        "current_learning_rate": self.current_lr,
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
        metrics = Metrics(self.device, num_classes=self.num_classes)

        test_loss, test_metrics = self._train_or_eval_loop(
            test_loader, is_train=False, metrics=metrics
        )

        print(
            f"  Test -> Loss: {test_loss:.4f}, {self.metric_for_best_model}: {test_metrics[self.metric_for_best_model]*100:.2f}%"
        )
        print(f"  Test Metrics: {test_metrics}")

        return test_metrics

    @staticmethod
    def save_checkpoint(state, is_best, checkpoint_dir, checkpoint_name: str = None):
        """Save checkpoint to disk"""
        last_checkpoint = (
            "model_last.pth"
            if not checkpoint_name
            else f"model_last_{checkpoint_name}.pth"
        )
        best_checkpoint = (
            "model_best.pth"
            if not checkpoint_name
            else f"model_best_{checkpoint_name}.pth"
        )

        filename = os.path.join(checkpoint_dir, last_checkpoint)
        torch.save(state, filename)
        print(f"Checkpoint saved to {filename}")
        if is_best:
            best_filename = os.path.join(checkpoint_dir, best_checkpoint)
            torch.save(state, best_filename)
            print(f"(Best) Checkpoint saved to {best_filename}")


class Trainer(BaseTrainer):

    def __init__(
        self,
        num_classes: int,
        scheduler_type: str = "CosineAnnealingLR",
        epochs: int = 1,
        optimizer_type: str = "SGD",
        use_wandb: bool = False,
        metric_for_best_model: str = "accuracy",
        checkpoint_dir: str = "./checkpoints",
        **kwargs,
    ):
        self.model = torch.hub.load("facebookresearch/dino:main", "dino_vits16")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        print(f"DINO ViT-S/16 model instantied. Using device: {self.device}")

        self.change_classifier_layer(num_classes)

        unfreeze_at_epoch = kwargs.get("unfreeze_at_epoch", None)
        self._train_head_only()

        lr = kwargs.get("lr", 1e-2)
        mom = kwargs.get("momentum", 0.9)
        wd = kwargs.get("weight_decay", 0)

        if optimizer_type == "SGD":
            optimizer = torch.optim.SGD(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=lr,
                momentum=mom,
                weight_decay=wd,
            )
        elif optimizer_type == "AdamW":
            optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=lr,
                weight_decay=wd,
                betas=kwargs.get("betas", (0.9, 0.999)),
            )
        else:
            raise ValueError(
                f"Missing configuration for optimizer type: {optimizer_type}."
            )

        if scheduler_type == "CosineAnnealingLR":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=kwargs.get("T_max", epochs), eta_min=0
            )
        else:
            raise ValueError(
                f"Missing configuration for scheduler type: {scheduler_type}."
            )

        loss_fn = nn.CrossEntropyLoss()

        base_trainer_args = {
            "epochs": epochs,
            "loss_fn": loss_fn,
            "optimizer": optimizer,
            "scheduler": scheduler,
            "use_wandb": use_wandb,
            "metric_for_best_model": metric_for_best_model,
            "num_classes": num_classes,
            "checkpoint_dir": checkpoint_dir,
            "unfreeze_at_epoch": unfreeze_at_epoch,
            "accum_steps": kwargs.get("accum_steps", 1),
        }

        super().__init__(**base_trainer_args)

        print("Trainer initialized.")
