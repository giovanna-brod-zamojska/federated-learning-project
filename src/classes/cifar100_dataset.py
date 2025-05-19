import os
import torch
import random
import numpy as np
from typing import Dict
from torch.utils.data import Subset

from typing import Dict, Optional, List
from collections import defaultdict
from torchvision import datasets, transforms
from torch.utils.data import Dataset, Subset, random_split, DataLoader


class CIFAR100Dataset:
    """
    The CIFAR100 dataset is an image dataset made of 50000 images, and 100 labels ranging from class 0 to 99.
    Labels are equally distributed, therefore to each label correspond 500 images.
    """

    def __init__(self, data_dir="./data", val_split=0.2, seed=42):
        self.data_dir = data_dir
        self.val_split = val_split
        self.seed = seed

        self.transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
            ]
        )

        # Check if the dataset exists, if not, download it
        self.dataset = self._load_or_download_dataset()

        self.train_data, self.test_data = self.dataset["train"], self.dataset["test"]
        self.class_count = len(self.train_data.classes)

    def get_num_labels(self) -> int:
        """
        Returns the number of classes in the dataset.
        """
        return self.class_count

    def _load_or_download_dataset(self):
        # Define dataset paths
        train_path = os.path.join(self.data_dir, "cifar-100-python", "train")
        test_path = os.path.join(self.data_dir, "cifar-100-python", "test")

        # Check if the dataset already exists
        if os.path.exists(train_path) and os.path.exists(test_path):
            print(f"Dataset found at {self.data_dir}. Loading the dataset...")
            return {
                "train": datasets.CIFAR100(
                    root=self.data_dir,
                    train=True,
                    download=False,
                    transform=self.transform,
                ),
                "test": datasets.CIFAR100(
                    root=self.data_dir,
                    train=False,
                    download=False,
                    transform=self.transform,
                ),
            }
        else:
            print(f"Dataset not found in {self.data_dir}. Downloading the dataset...")
            # If the dataset doesn't exist, download it
            return {
                "train": datasets.CIFAR100(
                    root=self.data_dir,
                    train=True,
                    download=True,
                    transform=self.transform,
                ),
                "test": datasets.CIFAR100(
                    root=self.data_dir,
                    train=False,
                    download=True,
                    transform=self.transform,
                ),
            }

    def _split_train_val(self):
        total_len = len(self.train_data)
        val_len = int(self.val_split * total_len)
        train_len = total_len - val_len
        return random_split(
            self.train_data,
            [train_len, val_len],
            generator=torch.Generator().manual_seed(self.seed),
        )

    def get_data_statistics(self, dataset):
        label_counts = defaultdict(int)
        for _, label in dataset:
            label_counts[label] += 1
        return dict(sorted(label_counts.items()))

    def print_stats(self):
        print(f"Train size: {len(self.train_data)}, Test size: {len(self.test_data)}")
        print("Train label distribution:", self.get_data_statistics(self.train_data))

    def get_split(self, split_type="classic", num_clients=10, nc=2):
        if split_type == "classic":
            train_data, val_data = self._split_train_val()
            return {"train": train_data, "val": val_data, "test": self.test_data}
        elif split_type == "iid":
            return self._iid_split(num_clients)
        elif split_type == "noniid":
            return self._noniid_split(num_clients, nc)
        else:
            raise ValueError(f"Unknown split_type: {split_type}")

    def _iid_split(self, num_clients: int) -> Dict[int, Subset]:
        """
        Create an I.I.D. split of the CIFAR-100 training set.
        Each client receives the same number of randomly selected samples,
        uniformly distributed across all classes.
        """
        num_samples = len(self.train_data)
        samples_per_client = num_samples // num_clients

        # Shuffle the indices
        indices = torch.randperm(num_samples, generator=torch.Generator().manual_seed(self.seed)).tolist()

        client_dict = {}
        for i in range(num_clients):
            start = i * samples_per_client
            end = (i + 1) * samples_per_client
            client_indices = indices[start:end]
            client_dict[i] = Subset(self.train_data, client_indices)

        return client_dict



    def _noniid_split(self, num_clients: int, nc: int) -> Dict[int, Subset]:
        """
        Create a non-I.I.D. split of the CIFAR-100 training set.
        Each client receives samples belonging to Nc random classes only.
        """
        random.seed(self.seed)

        # Group indices by class
        class_to_indices = defaultdict(list)
        for idx, (_, label) in enumerate(self.train_data):
            class_to_indices[label].append(idx)

        # Shuffle each class' indices
        for indices in class_to_indices.values():
            random.shuffle(indices)

        all_labels = list(class_to_indices.keys())
        client_dict = {}

        for client_id in range(num_clients):
            selected_classes = random.sample(all_labels, nc)
            client_indices = []

            for cls in selected_classes:
                take = min(len(class_to_indices[cls]) // num_clients, 20)
                client_indices.extend(class_to_indices[cls][:take])
                class_to_indices[cls] = class_to_indices[cls][take:]  # remove used

            client_dict[client_id] = Subset(self.train_data, client_indices)

        return client_dict



class TransformedSubset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, idx):
        x, y = self.subset[idx]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)


class CIFAR100Dataset_v2:

    def __init__(
        self,
        data_dir="./data",
        val_split=0.2,
        seed=42,
        num_clients: Optional[int] = 100,
        nc: Optional[float] = None,
    ):

        self.data_dir = data_dir
        self.val_split = val_split
        self.seed = seed

        self.train_transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
            ]
        )
        self.test_transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

        self.dataset = self._load_or_download_dataset()
        self.train_data, self.test_data = self.dataset["train"], self.dataset["test"]
        self.class_count = len(self.train_data.classes)  # 100 classes

        self.base_partition = self.create_base_partition()

        # We will precompute client splits here for train
        self.num_clients = num_clients
        self.nc = nc

        self.iid_partitions = self.create_iid_splits(num_clients=num_clients)
        self.noniid_partitions = self.create_noniid_splits(
            num_clients=num_clients, nc=nc
        )

    def _load_or_download_dataset(self):

        download = True
        if os.path.exists(
            os.path.join(self.data_dir, "cifar-100-python", "train")
        ) and os.path.exists(os.path.join(self.data_dir, "cifar-100-python", "test")):
            print(f"Dataset found at {self.data_dir}. Loading...")
            download = False
        else:
            print(f"Dataset not found at {self.data_dir}. Downloading...")

        return {
            "train": datasets.CIFAR100(
                root=self.data_dir,
                train=True,
                download=download,
                transform=None,  # for train-val split we will need to apply separate transforms later
            ),
            "test": datasets.CIFAR100(
                root=self.data_dir,
                train=False,
                download=download,
                transform=self.test_transform,
            ),
        }

    def get_data_statistics(self, dataset):
        counts = defaultdict(int)
        for _, label in dataset:
            counts[label] += 1
        return dict(sorted(counts.items()))

    def print_stats(self):
        print(f"Train size: {len(self.train_data)}, Test size: {len(self.test_data)}")
        print("Train label distribution:", self.get_data_statistics(self.train_data))

    def get_num_labels(self) -> int:
        return self.class_count

    def create_iid_splits(self, num_clients: int) -> List[Dict[str, Dataset]]:
        """
        Create an I.I.D. split of the CIFAR-100 training set:
        each client receives an equal number of samples randomly drawn 
        from the full dataset (uniformly across all classes).
        
        Each client's data is also split into a local train/val subset.
        """
        num_samples = len(self.train_data)
        samples_per_client = num_samples // num_clients

        # Generate shuffled indices for reproducibility
        indices = torch.randperm(
            num_samples, generator=torch.Generator().manual_seed(self.seed)
        ).tolist()

        iid_partitions = []

        for i in range(num_clients):
            start = i * samples_per_client
            end = (i + 1) * samples_per_client if i < num_clients - 1 else num_samples
            client_indices = indices[start:end]

            subset = Subset(self.train_data, client_indices)
            train_subset, val_subset = self._train_test_split(subset)

            iid_partitions.append({"train": train_subset, "val": val_subset})

        return iid_partitions


    def create_noniid_splits(self, num_clients: int, nc: int) -> List[Dict[str, Dataset]]:
        """
        Create a non-IID split of the CIFAR-100 training set:
        each client receives samples from exactly Nc randomly selected classes.
        """
        random.seed(self.seed)

        label_to_indices = defaultdict(list)
        for idx, (_, label) in enumerate(self.train_data):
            label_to_indices[label].append(idx)

        # Shuffle each class's indices
        for indices in label_to_indices.values():
            random.shuffle(indices)

        all_labels = list(label_to_indices.keys())
        noniid_partitions = []

        for _ in range(num_clients):
            selected_classes = random.sample(all_labels, nc)
            client_indices = []

            for label in selected_classes:
                take = min(len(label_to_indices[label]) // num_clients, 20)
                client_indices.extend(label_to_indices[label][:take])
                label_to_indices[label] = label_to_indices[label][take:]

            subset = Subset(self.train_data, client_indices)
            train_subset, val_subset = self._train_test_split(subset)

            noniid_partitions.append({"train": train_subset, "val": val_subset})

        return noniid_partitions


    def create_base_partition(self) -> Dict[str, Dataset]:
        train_subset, val_subset = self._train_test_split()

        return {"train": train_subset, "val": val_subset}

    def _train_test_split(self, subset=None):
        """ "
        Split the dataset into train and val subsets.
        If subset is provided, it will be used instead of the full dataset (to be used for federated partitioning).
        It will then apply self.train_transform and self.test_transform to the obtained train and validiation splits.
        """

        dataset = self.dataset["train"]
        if subset:
            dataset = subset  # client partition

        # Split dataset into train/val subsets
        total_len = len(dataset)
        val_len = int(self.val_split * total_len)
        train_len = total_len - val_len

        train_sub, val_sub = random_split(
            dataset,
            [train_len, val_len],
            generator=torch.Generator().manual_seed(self.seed),
        )

        # Wrap with transforms
        train_sub = TransformedSubset(train_sub, transform=self.train_transform)
        val_sub = TransformedSubset(val_sub, transform=self.test_transform)

        return train_sub, val_sub

    def get_dataloaders(
        self,
        client_id: Optional[int],
        split_type: Optional[str],
        batch_size: int = 32,
        pin_memory=True,
        worker_init_fn=None,
        num_workers=4,
    ):
        """
        Return Train Val and Test DataLoaders for a client partition, if specified,
        otherwise for the base partition.
        split_type: "iid" or "noniid"
        """
        dataset = self.base_partition
        if client_id:
            if split_type == "iid":
                dataset = self.iid_partitions[client_id]

            elif split_type == "noniid":
                dataset = self.noniid_partitions[client_id]

            else:
                raise ValueError("Unknown split_type")

        train_dataloader = DataLoader(
            dataset["train"],
            batch_size=batch_size,
            shuffle=True,
            pin_memory=pin_memory,
            num_workers=num_workers,
            worker_init_fn=worker_init_fn,
        )
        val_dataloader = DataLoader(
            dataset["val"],
            batch_size=batch_size,
            shuffle=False,
            pin_memory=pin_memory,
            num_workers=num_workers,
            worker_init_fn=worker_init_fn,
        )
        test_dataloader = DataLoader(
            self.dataset["test"],
            batch_size=batch_size,
            shuffle=False,
            pin_memory=pin_memory,
            num_workers=num_workers,
            worker_init_fn=worker_init_fn,
        )
        return train_dataloader, val_dataloader, test_dataloader
