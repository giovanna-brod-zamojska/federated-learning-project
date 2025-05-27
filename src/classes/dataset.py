import os
import torch
import random
from collections import defaultdict
from typing import Dict, Optional, List
from torchvision import datasets, transforms
from torch.utils.data import Dataset, Subset, random_split, DataLoader



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


class CIFAR100Dataset:

    def __init__(
        self,
        data_dir="./data",
        val_split=0.2,
        seed=42,
        num_clients: Optional[int] = 100,
        nc: Optional[float] = None,
    ):

        self.data_dir = data_dir
        self.resized_data_dir = "./data_resized"
        self.val_split = val_split
        self.seed = seed

        self.train_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        self.test_transform = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
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
            print(f"Dataset not found at {self.data_dir}.")

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

    def get_num_labels(self, split: str = None, client_id: int = None) -> int:

        if client_id is not None:
            if split == "iid":
                return len(self.iid_partitions[client_id]["train"].classes)
            elif split == "noniid":
                return len(self.noniid_partitions[client_id]["train"].classes)
            else:
                raise ValueError("Unknown split type")
            
        return len(self.train_data.classes) 

    def create_iid_splits(self, num_clients: int) -> List[Dict[str, Dataset]]:

        # THE FOLLOWING IS AUTOMATICALLY GENERATED CODE
        # TODO: implement/correct but keeping the usage of self._train_test_split and the output signature

        # Similar to _iid_split, but store splits internally
        num_samples = len(self.dataset["train"])
        indices = torch.randperm(
            num_samples, generator=torch.Generator().manual_seed(self.seed)
        ).tolist()
        split_size = num_samples // num_clients

        iid_partitions = []

        for i in range(num_clients):
            start = i * split_size
            end = num_samples if i == num_clients - 1 else (i + 1) * split_size
            client_indices = indices[start:end]

            subset = Subset(self.dataset["train"], client_indices)
            # Split train subset into train/val per client
            train_subset, val_subset = self._train_test_split(subset)

            iid_partitions.append({"train": train_subset, "val": val_subset})

        return iid_partitions

    def create_noniid_splits(
        self, num_clients: int, nc: int
    ) -> List[Dict[str, Dataset]]:

        return None
        # THE FOLLOWING IS AUTOMATICALLY GENERATED CODE
        # TODO: implement/correct but keeping the usage of self._train_test_split and the output signature

        # Similar to _noniid_split but store splits internally, each subset split into train/val
        # Map label to indices, assign classes, then split train/val per client
        label_to_indices = defaultdict(list)
        for idx, (_, label) in enumerate(self.dataset["train"]):
            label_to_indices[label].append(idx)

        labels = list(label_to_indices.keys())
        random.seed(self.seed)
        random.shuffle(labels)
        classes_per_client = nc
        labels_per_client = [
            labels[i * classes_per_client : (i + 1) * classes_per_client]
            for i in range(num_clients)
        ]

        noniid_partitions = []

        for client_labels in labels_per_client:
            client_indices = []
            for lbl in client_labels:
                client_indices.extend(label_to_indices[lbl])
            subset = Subset(self.dataset["train"], client_indices)

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
        client_id: Optional[int] = None,
        split_type: Optional[str] = None,
        batch_size: int = 32,
        pin_memory=True,
        num_workers=4,
        seed=None,
    ):
        """
        Return Train Val and Test DataLoaders for a client partition, if specified,
        otherwise for the base partition.
        split_type: "iid" or "noniid"
        """
        seed = seed if seed else self.seed
        g = torch.Generator()
        g.manual_seed(seed)

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
            generator=g,
        )
        val_dataloader = DataLoader(
            dataset["val"],
            batch_size=batch_size,
            shuffle=False,
            pin_memory=pin_memory,
            num_workers=num_workers,
            generator=g,
        )
        test_dataloader = DataLoader(
            self.dataset["test"],
            batch_size=batch_size,
            shuffle=False,
            pin_memory=pin_memory,
            num_workers=num_workers,
            generator=g,
        )
        return train_dataloader, val_dataloader, test_dataloader
