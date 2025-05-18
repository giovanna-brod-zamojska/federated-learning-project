import torch
from torchvision import datasets, transforms
from torch.utils.data import random_split
from collections import defaultdict
import random
import json
import os

def split_train_val(train_ratio=0.8):
    # Transformation basique (à adapter plus tard si besoin)
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # Téléchargement du dataset CIFAR-100
    full_trainset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)

    # Taille du split
    train_size = int(train_ratio * len(full_trainset))
    val_size = len(full_trainset) - train_size

    train_set, val_set = random_split(full_trainset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

    print(f"Train set size: {len(train_set)}")
    print(f"Validation set size: {len(val_set)}")
    return train_set, val_set
def generate_iid_clients(train_dataset, num_clients=100):
    """
    Répartit aléatoirement les indices du dataset entre les clients (i.i.d.)
    """
    num_items_per_client = len(train_dataset) // num_clients
    all_indices = torch.randperm(len(train_dataset))

    client_dict = {}

    for i in range(num_clients):
        start = i * num_items_per_client
        end = start + num_items_per_client
        client_indices = all_indices[start:end].tolist()
        client_dict[i] = client_indices

    print(f"Répartition i.i.d. : {num_clients} clients avec {num_items_per_client} images chacun.")
    return client_dict

def generate_noniid_clients(train_dataset, num_clients=100, Nc=2, seed=42):
    """
    Distribue les données de manière non-i.i.d. :
    chaque client reçoit des images appartenant à Nc classes aléatoires.
    """
    random.seed(seed)

    # Regroupe les indices par label
    class_to_indices = defaultdict(list)
    for idx, (_, label) in enumerate(train_dataset):
        class_to_indices[label].append(idx)

    # Mélange les indices dans chaque classe
    for indices in class_to_indices.values():
        random.shuffle(indices)

    client_dict = {i: [] for i in range(num_clients)}
    all_classes = list(class_to_indices.keys())

    # Répartition non-i.i.d. : chaque client reçoit Nc classes
    for client_id in range(num_clients):
        chosen_classes = random.sample(all_classes, Nc)
        for c in chosen_classes:
            take = min(len(class_to_indices[c]) // num_clients, 20)  # nombre d'images par classe/client
            client_dict[client_id].extend(class_to_indices[c][:take])
            class_to_indices[c] = class_to_indices[c][take:]

    print(f"Répartition non-i.i.d. : {num_clients} clients, {Nc} classes chacun.")
    return client_dict

def save_client_dict(client_dict, filename="client_partition.json", folder="partitions"):
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, filename)
    with open(path, "w") as f:
        json.dump(client_dict, f)
    print(f"✅ Sauvegardé : {path}")

if __name__ == "__main__":
    # 1. Split train / val
    train_set, val_set = split_train_val()

    # 2. I.I.D. partitioning
    client_data_iid = generate_iid_clients(train_set)
    save_client_dict(client_data_iid, filename="iid_clients.json")

    # 3. Non-I.I.D. partitioning for multiple Nc values
    for Nc in [1, 5, 10, 50]:
        client_data_noniid = generate_noniid_clients(train_set, Nc=Nc)
        save_client_dict(client_data_noniid, filename=f"noniid_clients_Nc{Nc}.json")

