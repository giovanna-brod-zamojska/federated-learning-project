import os
import torch
from classes.DataLoader import FederatedCIFAR100DataLoader


def main():
    # Load DINO ViT-S/16 model
    # vits16 = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')

    # Load Dataset
    manager = FederatedCIFAR100DataLoader()
    manager.print_stats()
    # TODO:
    # classic = manager.get_split(split_type='classic')
    # iid = manager.get_split(split_type='iid', num_clients=100)
    # noniid = manager.get_split(split_type='noniid', num_clients=100, nc=2)



if __name__ == "__main__":
    main()