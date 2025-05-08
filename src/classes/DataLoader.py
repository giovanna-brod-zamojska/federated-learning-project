import os
import torch
import random
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple, Union
from torchvision import datasets, transforms
from torch.utils.data import Dataset, Subset, random_split


# EXAMPLE USAGE

# manager = FederatedCIFAR100DataLoader(data_dir='./data')
# manager.print_stats()
# classic = manager.get_split(split_type='classic')
# iid = manager.get_split(split_type='iid', num_clients=5)
# noniid = manager.get_split(split_type='noniid', num_clients=5, nc=2)

class FederatedCIFAR100DataLoader:
    """
    The CIFAR100 dataset is an image dataset made of 50000 images, and 100 labels ranging from class 0 to 99.
    Labels are equally distributed, therefore to each label correspond 500 images.
    """

    def __init__(self, data_dir='./data', val_split=0.2, seed=42):
        self.data_dir = data_dir
        self.val_split = val_split
        self.seed = seed
        self.transform = transforms.ToTensor()

        # Check if the dataset exists, if not, download it
        self.dataset = self._load_or_download_dataset()

        self.train_data, self.test_data = self.dataset['train'], self.dataset['test']
        self.class_count = 100

    def _load_or_download_dataset(self):
        # Define dataset paths
        train_path = os.path.join(self.data_dir, 'cifar-100-python', 'train')
        test_path = os.path.join(self.data_dir, 'cifar-100-python', 'test')

        # Check if the dataset already exists
        if os.path.exists(train_path) and os.path.exists(test_path):
            print(f"Dataset found at {self.data_dir}. Loading the dataset...")
            return {
                'train': datasets.CIFAR100(root=self.data_dir, train=True, download=False, transform=self.transform),
                'test': datasets.CIFAR100(root=self.data_dir, train=False, download=False, transform=self.transform)
            }
        else:
            print(f"Dataset not found in {self.data_dir}. Downloading the dataset...")
            # If the dataset doesn't exist, download it
            return {
                'train': datasets.CIFAR100(root=self.data_dir, train=True, download=True, transform=self.transform),
                'test': datasets.CIFAR100(root=self.data_dir, train=False, download=True, transform=self.transform)
            }

    def _split_train_val(self):
        total_len = len(self.train_data)
        val_len = int(self.val_split * total_len)
        train_len = total_len - val_len
        return random_split(self.train_data, [train_len, val_len], generator=torch.Generator().manual_seed(self.seed))

    def get_data_statistics(self, dataset):
        label_counts = defaultdict(int)
        for _, label in dataset:
            label_counts[label] += 1
        return dict(sorted(label_counts.items()))

    def print_stats(self):
        print(f"Train size: {len(self.train_data)}, Test size: {len(self.test_data)}")
        print("Train label distribution:", self.get_data_statistics(self.train_data))

    def get_split(self, split_type='classic', num_clients=10, nc=2):
        if split_type == 'classic':
            train_data, val_data = self._split_train_val()
            return {'train': train_data, 'val': val_data, 'test': self.test_data}
        elif split_type == 'iid':
            return self._iid_split(num_clients)
        elif split_type == 'noniid':
            return self._noniid_split(num_clients, nc)
        else:
            raise ValueError(f"Unknown split_type: {split_type}")

    def _iid_split(self, num_clients: int) -> Dict[int, Subset]:
        """
        Create a IID split of the CIFAR-100 dataset.
        Each of K clients is given an approximately equal number of training 
        samples uniformly distributed over the class labels.
        
        Arguments:
        - num_clients (int): Number of clients to split the dataset into
        
        Returns:
        - A dictionary mapping client IDs to their respective subsets of data.
        """
        raise NotImplementedError
        

    def _noniid_split(self, num_clients: int, nc: int) -> Dict[int, Subset]:
        """
        Create a non-IID split of the CIFAR-100 dataset.
        Each client is given an approximately equal number of training 
        samples, belonging to Nc distinct classes, where Nc is an hyperparameter you will use 
        to control the severity of the induced dataset heterogeneity. 
        For example, if Nc=1, then each client has samples belonging to one class only.
        
        Arguments:
        - num_clients (int): Number of clients to split the dataset into
        - nc (int): Number of distinct classes per client
        
        Returns:
        - A dictionary mapping client IDs to their respective subsets of data.
        """
        raise NotImplementedError

