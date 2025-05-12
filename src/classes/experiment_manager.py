import wandb
from typing import Dict, Any, List, Optional, Tuple
from copy import deepcopy
from torch.utils.data import DataLoader

# NOT ready - work in progress


class ExperimentManager:
    """
    Manages multiple training experiments with different hyperparameter configurations,
    logs results to Weights & Biases (WandB), and tracks the best performing setup.
    """

    def __init__(
        self,
        base_config: Dict[str, Any],
        param_grid: List[Dict[str, Any]],
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: Optional[DataLoader] = None,
        use_wandb: bool = False,
        project_name: str = "vit_baseline_experiments",
    ):
        """
        Initializes the experiment manager.

        Args:
            base_config (dict): Default configuration for trainer.
            param_grid (list of dicts): List of parameter combinations to run.
            train_loader (DataLoader): Training data loader.
            val_loader (DataLoader): Validation data loader.
            test_loader (DataLoader, optional): Optional test data loader.
            use_wandb (bool): Whether to use WandB for logging.
            project_name (str): WandB project name.
        """
        self.base_config = base_config
        self.param_grid = param_grid
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.use_wandb = use_wandb
        self.project_name = project_name
        self.results: List[Dict[str, Any]] = []

    def run(self, trainer_class) -> Tuple[Dict[str, Any], float]:
        """
        Runs all experiments in the parameter grid and tracks the best configuration.

        Returns:
            Tuple of best configuration and best validation accuracy.
        """
        best_acc = 0.0
        best_config = None

        for idx, params in enumerate(self.param_grid):
            config = deepcopy(self.base_config)
            config.update(params)

            print(
                f"\n🔬 Running experiment {idx + 1}/{len(self.param_grid)} with config: {params}"
            )

            if self.use_wandb:
                wandb.init(project=self.project_name, config=config, reinit=True)
                wandb.run.name = f"run_{idx + 1}"

            trainer = trainer_class(
                model=None,
                lr=config["lr"],
                momentum=config["momentum"],
                weight_decay=config["weight_decay"],
                epochs=config["epochs"],
            )
            trainer.change_classifier_layer(num_classes=config["num_classes"])

            trainer.train(self.train_loader, self.val_loader)

            test_acc = None
            if self.test_loader:
                test_acc = trainer.test(self.test_loader)
                if self.use_wandb:
                    wandb.log({"test_accuracy": test_acc})

            if self.use_wandb:
                wandb.finish()

            result = {"config": config, "val_accuracy": test_acc}
            self.results.append(result)

            if test_acc and test_acc > best_acc:
                best_acc = test_acc
                best_config = config

        print(f"\n🏆 Best configuration achieved accuracy {best_acc:.2f}%")
        return best_config, best_acc

    def get_all_results(self) -> List[Dict[str, Any]]:
        """
        Returns:
            List of all experiment results with config and performance.
        """
        return self.results
