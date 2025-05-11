import torch
import argparse
import numpy as np
from src.classes.CentralizedBaselineTrainer import CentralizedBaselineTrainer
from src.classes.CIFAR100Dataset import CIFAR100Dataset
from torch.utils.data import DataLoader

def parse_args():
    parser = argparse.ArgumentParser(description='Training script for image classification')
    # parser.add_argument('--config', type=str, default='configs/config.yaml', help='Path to config file')
    # parser.add_argument('--data_dir', type=str, default='data/tiny-imagenet-200', help='Path to dataset')
    # parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    return parser.parse_args()

def set_seed(seed):
    """Set random seed for reproducibility"""
    print(f"Setting random seed to {seed}")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train(resume: str = None, seed: int = 42):

    set_seed(seed)

    if resume:
        print(f"Resuming training from checkpoint: {resume}")
   
    dataset = CIFAR100Dataset()
    print("Dataset instantiated.\n")

    trainer = CentralizedBaselineTrainer(epochs=1)
    print("Trainer instantiated.")

    trainer.change_classifier_layer(dataset.get_num_labels())
    print("Last layer changed.\n")

    dataset_dict = dataset.get_split(split_type="classic")
    train_loader = DataLoader(dataset_dict["train"], batch_size=32, shuffle=True)
    valid_loader = DataLoader(dataset_dict["val"], batch_size=32, shuffle=False)
    test_loader = DataLoader(dataset_dict["test"], batch_size=32, shuffle=False)
    print("Data loaders created.\n")

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