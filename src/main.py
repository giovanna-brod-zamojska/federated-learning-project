import os
import torch
import numpy as np
from classes.CIFAR100Dataset import CIFAR100Dataset
from classes.CentralizedBaselineTrainer import CentralizedBaselineTrainer

def set_seed(seed):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():

    # Load Dataset
    manager = CIFAR100Dataset()
    manager.print_stats()
    
    # TODO:
    # classic = manager.get_split(split_type='classic')
    # iid = manager.get_split(split_type='iid', num_clients=100)
    # noniid = manager.get_split(split_type='noniid', num_clients=100, nc=2)



if __name__ == "__main__":
    main()