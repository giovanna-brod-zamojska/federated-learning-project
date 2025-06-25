import json
import torch
import wandb
import random
import numpy as np
from datetime import date
from torch.utils.data import DataLoader
from typing import Dict, Any, List, Tuple

from src.centralized_baseline.trainer import BaseTrainer
from src.centralized_baseline.dataset import CIFAR100Dataset
from src.model_editing.mask_calibration import calibrate_mask


class ExperimentManager:
    """
    Manages multiple training experiments with different hyperparameter configurations,
    logs results to Weights & Biases (WandB), and tracks the best performing setup.
    """

    def __init__(
        self,
        param_grid: List[Dict[str, Any]],
        use_wandb: bool = False,
        project_name: str = "federated-learning-project-TEST",
        group_name: str = "centralized_baseline",
        checkpoint_dir: str = "./checkpoints",
    ):
        self.param_grid = param_grid
        self.use_wandb = use_wandb
        self.project_name = project_name
        self.group_name = group_name
        self.checkpoint_dir = checkpoint_dir

    @staticmethod
    def set_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def setup_dataset(
        self, dataset: CIFAR100Dataset, config
    ) -> Tuple[CIFAR100Dataset, DataLoader, DataLoader, DataLoader]:

        train_loader, valid_loader, test_loader = dataset.get_dataloaders(
            batch_size=config["batch_size"],
            num_workers=config["num_workers"],
            pin_memory=True,
            seed=config["seed"],
        )

        print("Data loaders created.\n")

        return dataset, train_loader, valid_loader, test_loader

    def run(
        self,
        trainer_class: BaseTrainer,
        dataset_class: CIFAR100Dataset,
        run_name: str,
        run_tags: List[str],
        metric_for_best_config: str = "accuracy",
        resume_training_from_config: int = None,
        model_editing: bool = False,
        test: bool = False,
    ) -> Tuple[Dict[str, Any], float]:

        results = []
        best_metric = 0.0
        best_config = None
        metric_for_best_model = metric_for_best_config

        start_idx = (
            resume_training_from_config - 1
            if resume_training_from_config is not None
            else 0
        )

        for idx in range(start_idx, len(self.param_grid)):

            config = self.param_grid[idx]

            self.set_seed(config["seed"])

            print(config)

            dataset = dataset_class(
                seed=config["seed"], augment=config.get("augment", False)
            )

            # summarize the config params into a str to have a detailed run description
            notes = ", ".join(f"{k}={v}" for k, v in config.items())
            checkpoint_name = "-".join(f"{k}{v}" for k, v in config.items())

            print(
                f"\nRunning experiment {idx + 1}/{len(self.param_grid)} with config: {config}"
            )

            run_tags.extend(
                [
                    f"bs{config['batch_size']}",
                    f"lr{config['lr']}",
                    f"Tmax{config.get('Tmax', config['epochs'])}",
                    f"ep{config['epochs']}",
                    f"wd{config['weight_decay']}",
                    f"accum_steps{config['accum_steps']}",
                    f"momentum{config['momentum']}",
                    f"seed{config['seed']}",
                    f"optimizer_type{config['optimizer_type']}",
                    f"augment{config['augment']}",
                ]
            )

            if self.use_wandb:
                wandb.init(
                    project=self.project_name,
                    group=self.group_name,  # Group runs under this name
                    name=f"run_{run_name}_{checkpoint_name}",  # Name of the run
                    notes=notes,
                    config=config,
                    tags=run_tags,
                    dir="./wandb_logs",  # Directory to save logs
                    reinit=True,  # Reinitialize WandB for each run
                )

            _, train_loader, val_loader, test_loader = self.setup_dataset(
                dataset, config
            )

            if model_editing:
                # Load model for mask calibration
                model = torch.hub.load("facebookresearch/dino:main", "dino_vits16")
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model.to(device)

                sparsity = config.get("sparsity", 0.9)
                num_batches = config.get("num_batches", None)
                strategy = config.get("strategy", "train_least_important")
                rounds = config.get("rounds", 5)
                approximate_fisher = config.get("approximate_fisher", False)

                mask = calibrate_mask(
                    model,
                    train_loader,
                    device=device,
                    sparsity=sparsity,
                    rounds=rounds,
                    num_batches=num_batches,
                    strategy=strategy,
                    approximate_fisher=approximate_fisher,
                )
                config["mask"] = mask
                run_tags.extend(
                    [
                        f"sparsity{sparsity}",
                        f"num_batches{num_batches}",
                        f"strategy{strategy}",
                        f"rounds{rounds}",
                        f"approximateFisher{approximate_fisher}",
                    ]
                )

            trainer = trainer_class(
                **config,
                num_classes=dataset.get_num_labels(),
                use_wandb=self.use_wandb,
                metric_for_best_model=metric_for_best_model,
                checkpoint_dir=self.checkpoint_dir,
                checkpoint_name=checkpoint_name,
            )

            metric = trainer.train(
                train_loader,
                val_loader,
            )

            if self.use_wandb:
                wandb.finish()

            if model_editing is True:
                del config["mask"]

            result = {"config": config, "val_metric": metric}
            results.append(result)
            results = sorted(results, key=lambda x: x["val_metric"], reverse=True)

            if metric and metric > best_metric:
                best_metric = metric
                best_config = config

            if test:
                test_metrics = trainer.test(test_loader)

            print(
                f"🏆Best config up to now: {json.dumps(best_config, indent=4)} with validation {metric_for_best_config}: {best_metric*100:.2f}%"
            )

        results = sorted(results, key=lambda x: x["val_metric"], reverse=True)

        print(
            f"🏆Best config: {json.dumps(best_config, indent=4)} with validation {metric_for_best_config}: {best_metric*100:.2f}%"
        )
        return best_config, best_metric, results
