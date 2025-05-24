import json
import torch
import wandb
import random
import numpy as np
from copy import deepcopy
from datetime import datetime
from torch.utils.data import DataLoader
from typing import Dict, Any, List, Optional, Tuple

from src.classes.trainer import BaseTrainer
from src.classes.cifar100_dataset import CIFAR100Dataset_v2 as CIFAR100Dataset

from datetime import date


class ExperimentManager:
    """
    Manages multiple training experiments with different hyperparameter configurations,
    logs results to Weights & Biases (WandB), and tracks the best performing setup.
    """

    def __init__(
        self,
        base_config: Dict[str, Any],
        param_grid: List[Dict[str, Any]],
        use_wandb: bool = False,
        project_name: str = "federated-learning-project-TEST",
        group_name: str = "centralized_baseline",
        do_test: bool = False,
        checkpoint_dir: str = "./checkpoints",
    ):
        self.do_test = do_test

        self.base_config = base_config
        self.param_grid = param_grid

        self.use_wandb = use_wandb
        self.project_name = project_name
        self.group_name = group_name
        self.checkpoint_dir = checkpoint_dir

    @staticmethod
    def worker_init_fn(worker_id):

        seed = torch.initial_seed() % 2**32  # Each worker gets a different seed
        np.random.seed(seed)
        random.seed(seed)

        print(
            f"Initial seed: {torch.initial_seed()}. Setting up seed={seed} for worker {worker_id}"
        )

    def setup_dataset(
        self, dataset: CIFAR100Dataset, config
    ) -> Tuple[CIFAR100Dataset, DataLoader, DataLoader, DataLoader]:

        train_loader, valid_loader, test_loader = dataset.get_dataloaders(
            batch_size=config["batch_size"],
            num_workers=config["num_workers"],
            pin_memory=True,
            worker_init_fn=self.worker_init_fn,
        )

        print("Data loaders created.\n")

        return dataset, train_loader, valid_loader, test_loader

    def run(
        self,
        trainer_class: BaseTrainer,
        dataset: CIFAR100Dataset,
        run_name: str,
        run_tags: List[str],
        resume: Optional[str] = None,
        metric_for_best_config: str = "accuracy",
        resume_training_from_config: int = None,
    ) -> Tuple[Dict[str, Any], float]:

        results = []
        best_metric = 0.0
        best_config = None
        metric_for_best_model = metric_for_best_config
        today = date.today()

        start_idx = (
            resume_training_from_config - 1
            if resume_training_from_config is not None
            else 0
        )

        for idx in range(start_idx, len(self.param_grid)):
            params = self.param_grid[idx]
            config = deepcopy(self.base_config)
            config.update(params)

            # summarize the config params into a str to have a detailed run description
            notes = ", ".join(f"{k}={v}" for k, v in config.items())

            print(
                f"\nRunning experiment {idx + 1}/{len(self.param_grid)} with config: {params}"
            )

            if self.use_wandb:
                run = wandb.init(
                    project=self.project_name,
                    group=self.group_name,  # Group runs under this name
                    name=f"run_{today}_{run_name}_config{idx + 1}",  # Name of the run
                    notes=notes,
                    config=config,
                    tags=run_tags,
                    dir="./wandb_logs",  # Directory to save logs
                    reinit=True,  # Reinitialize WandB for each run
                )

            _, train_loader, val_loader, test_loader = self.setup_dataset(
                dataset, config
            )

            trainer = trainer_class(
                **config,
                num_classes=dataset.get_num_labels(),
                use_wandb=self.use_wandb,
                metric_for_best_model=metric_for_best_model,
                checkpoint_dir=self.checkpoint_dir,
            )

            if resume:
                metric = trainer.train(
                    train_loader,
                    val_loader,
                    resume=resume,
                )
            else:
                metric = trainer.train(
                    train_loader,
                    val_loader,
                )

            if self.do_test:
                test_metrics = trainer.test(test_loader)
                if self.use_wandb:
                    wandb.log({f"test_{k}": v for k, v in test_metrics.items()})

            if self.use_wandb:
                wandb.finish()

            result = {"config": config, "val_metric": metric}
            results.append(result)
            results = sorted(results, key=lambda x: x["val_metric"], reverse=True)

            if metric and metric > best_metric:
                best_metric = metric
                best_config = config

            print(
                f"🏆Best config up to now: {json.dumps(best_config, indent=4)} with validation {metric_for_best_config}: {best_metric*100:.2f}%"
            )

        results = sorted(results, key=lambda x: x["val_metric"], reverse=True)

        print(
            f"🏆Best config: {json.dumps(best_config, indent=4)} with validation {metric_for_best_config}: {best_metric*100:.2f}%"
        )
        return best_config, best_metric, results
