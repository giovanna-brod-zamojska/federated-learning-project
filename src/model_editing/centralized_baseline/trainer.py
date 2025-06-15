from src.centralized_baseline.trainer import BaseTrainer
import torch
import torch.nn as nn

from ..sparse_sgd import SparseSGD


class ModelEditingTrainer(BaseTrainer):

    def __init__(
        self,
        num_classes: int,
        scheduler_type: str = "CosineAnnealingLR",
        epochs: int = 1,
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

        mask = kwargs.get("mask", None)
        for name, param in self.model.named_parameters():
            if name in mask and mask[name].sum() == 0:
                param.requires_grad = False
            else:
                param.requires_grad = True  # Trainable

        unfreeze_at_epoch = kwargs.get("unfreeze_at_epoch", None)

        lr = kwargs.get("lr", 1e-2)
        mom = kwargs.get("momentum", 0.9)
        wd = kwargs.get("weight_decay", 0)

        optimizer = SparseSGD(
            params=filter(lambda p: p.requires_grad, self.model.parameters()),
            named_params={
                n: p for n, p in self.model.named_parameters() if p.requires_grad
            },
            lr=lr,
            momentum=mom,
            weight_decay=wd,
            mask=mask,
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
