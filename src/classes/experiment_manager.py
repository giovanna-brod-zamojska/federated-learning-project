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
from src.classes.cifar100_dataset import CIFAR100Dataset

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
    ):
        self.do_test = do_test

        self.base_config = base_config
        self.param_grid = param_grid

        self.use_wandb = use_wandb
        self.project_name = project_name
        self.group_name = group_name

    @staticmethod
    def worker_init_fn(worker_id):
        seed = torch.initial_seed() % 2**32  # Each worker gets a different seed
        np.random.seed(seed)
        random.seed(seed)

    def setup_dataset(
        self, dataset: CIFAR100Dataset, config
    ) -> Tuple[CIFAR100Dataset, DataLoader, DataLoader, DataLoader]:

        dataset_dict = dataset.get_split(split_type="classic")

        train_loader = DataLoader(
            dataset_dict["train"],
            batch_size=config["batch_size"],
            shuffle=True,
            num_workers=config["num_workers"],
            pin_memory=True,
            worker_init_fn=self.worker_init_fn,
        )
        valid_loader = DataLoader(
            dataset_dict["val"],
            batch_size=config["batch_size"],
            shuffle=False,
            num_workers=config["num_workers"],
            pin_memory=True,
            worker_init_fn=self.worker_init_fn,
        )
        test_loader = DataLoader(
            dataset_dict["test"],
            batch_size=config["batch_size"],
            shuffle=False,
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
    ) -> Tuple[Dict[str, Any], float]:

        results = []
        best_metric = 0.0
        best_config = None
        metric_for_best_model = metric_for_best_config
        today = date.today()

        for idx, params in enumerate(self.param_grid):
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

            if metric and metric > best_metric:
                best_metric = metric
                best_config = config

        print(
            f"🏆Best config: {json.dumps(best_config, indent=4)} with validation {metric_for_best_config}: {best_metric:.2f}%"
        )
        return best_config, best_metric, results
