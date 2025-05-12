import torch
import argparse
import numpy as np
from src.classes.centralized_baseline_trainer import CentralizedBaselineTrainer
from src.classes.cifar100_dataset import CIFAR100Dataset
from torch.utils.data import DataLoader
from src.classes.experiment_manager import ExperimentManager
from src.classes.metrics import Metrics
from src.classes.experiment_manager import ExperimentManager


def parse_args():
    parser = argparse.ArgumentParser(
        description="Training script for image classification"
    )
    # parser.add_argument('--config', type=str, default='configs/config.yaml', help='Path to config file')
    # parser.add_argument('--data_dir', type=str, default='data/tiny-imagenet-200', help='Path to dataset')
    # parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Directory to save checkpoints')
    parser.add_argument(
        "--resume", type=str, default=None, help="Path to checkpoint to resume from"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def set_seed(seed):
    """Set random seed for reproducibility"""
    print(f"Setting random seed to {seed}")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_dataset():
    dataset = CIFAR100Dataset()
    print("Dataset instantiated.\n")

    dataset_dict = dataset.get_split(split_type="classic")

    train_loader = DataLoader(dataset_dict["train"], batch_size=32, shuffle=True)
    valid_loader = DataLoader(dataset_dict["val"], batch_size=32, shuffle=False)
    test_loader = DataLoader(dataset_dict["test"], batch_size=32, shuffle=False)
    print("Data loaders created.\n")

    return dataset, train_loader, valid_loader, test_loader


def run_experiments():
    raise NotImplementedError("Experiment manager is not ready yet.")

    dataset, train_loader, valid_loader, test_loader = setup_dataset()

    manager = ExperimentManager(
        base_config={
            "model": "vit",
            "loss_fn": "cross_entropy",
            "optimizer_class": torch.optim.Adam,
            "lr": 0.001,
            "momentum": 0.9,
            "weight_decay": 0.0,
            "epochs": 10,
        },
        param_grid=[
            {"batch_size": 32, "num_workers": 4},
            {"batch_size": 64, "num_workers": 8},
        ],
        train_loader=train_loader,
        val_loader=valid_loader,
        test_loader=test_loader,
        use_wandb=True,
        project_name="vit_baseline_experiments_TEST",
    )
    best_config, best_acc = manager.run(CentralizedBaselineTrainer)
    print(f"Best configuration: {best_config}")
    print(f"Best validation accuracy: {best_acc}")
    print("Experiments completed.\n")


def train(resume: str = None, seed: int = 42, epochs: int = 10):

    set_seed(seed)

    if resume:
        print(f"Resuming training from checkpoint: {resume}")

    trainer = CentralizedBaselineTrainer(epochs=epochs)
    print("Trainer instantiated.")

    dataset, train_loader, valid_loader, test_loader = setup_dataset()

    trainer.change_classifier_layer(dataset.get_num_labels())
    print("Model head before training:", trainer.model.head)

    trainer.train(train_loader, valid_loader, resume=resume)
    print("Training completed.\n")

    trainer.test(test_loader)
    print("Testing completed.\n")


if __name__ == "__main__":

    args = parse_args()

    resume = args.resume
    seed = args.seed

    if resume:
        train(resume=resume, seed=seed)
    else:
        train(seed=seed)
