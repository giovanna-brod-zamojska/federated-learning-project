import os
import torch
from classes.DataLoader import FederatedCIFAR100DataLoader
from classes.CentralizedBaselineTrainer import CentralizedBaselineTrainer


def main():

    # Load Dataset
    manager = FederatedCIFAR100DataLoader()
    manager.print_stats()
    
    # TODO:
    # classic = manager.get_split(split_type='classic')
    # iid = manager.get_split(split_type='iid', num_clients=100)
    # noniid = manager.get_split(split_type='noniid', num_clients=100, nc=2)



if __name__ == "__main__":
    main()