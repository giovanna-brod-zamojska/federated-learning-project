import torch
import numpy as np
from typing import List
from collections import OrderedDict
from flwr.server.strategy import FedAvg
from flwr.client import Client, NumPyClient
from flwr.server import ServerConfig, ServerAppComponents
from flwr.common import ndarrays_to_parameters, Context

from src.classes.trainer import Trainer
from src.classes.cifar100_dataset import CIFAR100Dataset_v2


# 1) Define Flower client
class FLClient(NumPyClient):
    def __init__(
        self, cid: str, trainer: Trainer, train_loader, val_loader, test_loader
    ):
        self.cid = cid
        self.trainer = trainer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        print(f"Client {self.cid} instantiated.")

    def get_parameters(self) -> List[np.ndarray]:
        print(f"[Client {self.cid}] get_parameters")

        # Return model weights as a list of NumPy arrays
        return [val.cpu().numpy() for _, val in self.trainer.model.state_dict().items()]

    def set_parameters(self, parameters: List[np.ndarray]):
        print(f"[Client {self.cid}] set_parameters")

        # Load weights from server
        params_dict = zip(self.trainer.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        self.trainer.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        print(f"[Client {self.cid}] fit, config: {config}")

        # Receive initial weights
        self.set_parameters(parameters)

        # Local training
        results = self.trainer.train(self.train_loader, self.val_loader)

        # Return updated weights and number of examples
        return self.get_parameters(), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):

        print(f"[Client {self.cid}] evaluate, config: {config}")

        # Receive global weights
        self.set_parameters(parameters)

        # Local evaluation
        metrics = self.trainer.test(self.test_loader)

        return (
            None,
            len(self.test_loader.dataset),
            {"accuracy": metrics["accuracy"]},
        )


# 2) Helper to create clients
def client_fn(
    context: Context, dataset: CIFAR100Dataset_v2, split_type="iid", **trainer_params
):

    # Note: each client gets a different trainloader/valloader, so each client
    # will train and evaluate on their own unique data partition
    # Read the node_config to fetch data partition associated to this node
    cid = context.node_config["partition-id"]

    print(f"Calling client_fn for client {cid}")

    train_loader, val_loader, test_loader = dataset.get_dataloaders(
        client_id=cid,
        split_type=split_type,
        batch_size=32,
    )

    # Instantiate Trainer (loads ViT)
    trainer = Trainer(num_classes=dataset.get_num_labels(), **trainer_params)

    # Create a single Flower client representing a single organization
    # FlowerClient is a subclass of NumPyClient, so we need to call .to_client()
    # to convert it to a subclass of `flwr.client.Client`
    return FLClient(cid, trainer, train_loader, val_loader, test_loader).to_client()


# 3) Server-side strategy
def server_fn(context: Context, num_rounds: int, **kwargs):

    print("Calling server_fn")

    strategy = FedAvg(
        fraction_fit=kwargs.get("fraction_fit", 1),
        fraction_evaluate=kwargs.get("fraction_fit", 1),
        min_fit_clients=kwargs.get("fraction_fit", 2),
        min_evaluate_clients=kwargs.get("fraction_fit", 2),
        min_available_clients=kwargs.get("min_available_clients", 2),
        evaluate_fn=kwargs.get("evaluate_fn", None),
    )

    config = ServerConfig(num_rounds=num_rounds)
    return ServerAppComponents(strategy=strategy, config=config)

