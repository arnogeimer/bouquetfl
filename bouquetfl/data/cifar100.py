# Some general parameters we need
import sys
from typing import List, OrderedDict, Tuple

import datasets
import numpy as np
import timm
import torch
import torch.nn as nn
import torchvision
import tqdm
import yaml
from data.cuda_vision import CUDA_VisionDataSet
from datasets import Dataset
from flwr.common.typing import NDArrays
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import (DirichletPartitioner, LinearPartitioner,
                                       SizePartitioner, SquarePartitioner)
from torch.utils.data import DataLoader


# Loading the model (Called when initializing FlowerClient and when testing)
def get_model() -> torch.nn.Module:
    return timm.create_model("resnet18").cuda()


transform_train = torchvision.transforms.Compose(
    [
        torchvision.transforms.RandomCrop(32, padding=4),
        torchvision.transforms.RandomRotation(15),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            (0.5071, 0.4867, 0.4408),
            (0.2675, 0.2565, 0.2761),
        ),
    ]
)

transform_test = torchvision.transforms.Compose(
    [
        # torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            (0.5071, 0.4867, 0.4408),
            (0.2675, 0.2565, 0.2761),
        ),
    ]
)


def size_based_split(num_clients):
    partitioner = SizePartitioner(
        partition_sizes=[int(50000 / num_clients) for _ in range(num_clients)],
    )
    return partitioner


def dirichlet_based_split(num_clients, alpha):
    partitioner = DirichletPartitioner(
        num_partitions=num_clients,
        partition_by="fine_label",
        alpha=alpha,
        min_partition_size=30,
    )
    return FederatedDataset(
        dataset="uoft-cs/cifar100",
        partitioners={"train": partitioner},
        trust_remote_code=True,
    )


def load_data(
    partition_id: int, num_clients: int = 36, num_workers: int = 4, batch_size: int = 64
) -> DataLoader:
    partitioner = size_based_split(num_clients)
    fds = FederatedDataset(
        dataset="uoft-cs/cifar100",
        partitioners={"train": partitioner},
        trust_remote_code=True,
    )

    dataset = fds.load_partition(partition_id=partition_id)

    def apply_transforms(batch):
        """Apply transforms to the partition from FederatedDataset."""
        batch["img"] = [transform_train(img) for img in batch["img"]]
        return batch

    dataset = dataset.with_transform(apply_transforms)

    trainloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
    )
    return trainloader


# Load the (global) test dataset
def load_global_test_data(batch_size: int = 64) -> DataLoader:
    testset = fds.load_split("test")

    def apply_test_transform(batch):
        """Apply transforms to the partition from FederatedDataset."""
        batch["img"] = [transform_test(img) for img in batch["img"]]
        return batch

    testset = testset.with_transform(apply_test_transform)

    testloader = DataLoader(
        testset, batch_size=batch_size, shuffle=True, drop_last=True
    )
    return testloader


# Train and test on a trainloader and testloader
def train(
    model: nn.Module,
    trainloader: DataLoader,
    epochs: int = 5,
    device: str = "cuda",
    **kwargs,
):
    """Train the model on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    print("Starting training")
    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3)
    for _ in tqdm.trange(epochs):
        for batch in trainloader:
            images, labels = batch["img"].to(device), batch["fine_label"].to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()


def test(
    model: nn.Module,
    testloader: DataLoader,
    device: str = "cuda",
) -> Tuple[float, float]:
    """Validate the model on the entire test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    if len(testloader) == 0:
        return np.inf, 0
    with torch.no_grad():
        for batch in testloader:
            images, labels = batch["img"].to(device), batch["fine_label"].to(device)
            outputs = model(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    del (testloader, model)
    torch.cuda.empty_cache()
    return loss, accuracy


def ndarrays_from_model(model: torch.nn.ModuleList) -> NDArrays:
    """Get model weights as a list of NumPy ndarrays."""
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def ndarrays_to_model(model: torch.nn.ModuleList, params: NDArrays):
    """Set model weights from a list of NumPy ndarrays."""
    params_dict = zip(model.state_dict().keys(), params)
    state_dict = OrderedDict({k: torch.from_numpy(np.copy(v)) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)


# All training calls, to be sent to the server and clients


from flwr.common import Context, ndarrays_to_parameters
from flwr.server.strategy import FedAvg


def evaluate_fn(server_round, weights_aggregated, dict, **kwargs):
    model = get_model()
    ndarrays_to_model(model, weights_aggregated)
    loss, accuracy = test(model, load_global_test_data())
    del model
    torch.cuda.empty_cache()
    return -loss, {"accuracy": accuracy}


def get_initial_parameters():
    init_model = get_model()
    initial_parameters = ndarrays_to_parameters(ndarrays_from_model(init_model))
    del init_model
    torch.cuda.empty_cache()
    return initial_parameters


client_resources: dict = {
    "num_cpus": 1,
    "num_gpus": 0.1,
}


def estimate_training_memory_usage(batch_size: int = 64):
    # https://discuss.pytorch.org/t/how-to-deal-with-excessive-memory-usages-of-pytorch/126098/5
    acts = []
    model = get_model()
    for name, module in model.named_modules():
        if name == "classifier" or name == "features":
            continue
        module.register_forward_hook(
            lambda m, input, output: acts.append(output.detach())
        )

    def get_model_size(model: torch.nn) -> float:
        "Get PyTorch model size in byte"
        size_model = 0
        for param in model.parameters():
            if param.data.is_floating_point():
                size_model += param.numel() * torch.finfo(param.data.dtype).bits
            else:
                size_model += param.numel() * torch.iinfo(param.data.dtype).bits
        return size_model / 8

    # execute single training step
    X, y_true = next(iter(load_global_test_data()))
    # Forward pass
    y_hat = model(X)
    criterion = torch.nn.CrossEntropyLoss()
    loss = criterion(y_hat, y_true)
    # Backward pass
    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # approximate memory requirements
    model_param_size_byte = get_model_size(model=model)
    grad_size_byte = model_param_size_byte
    batch_size_byte = 32 * 32 * 3 * batch_size
    optimizer_size_byte = sum(
        [
            (
                p.nelement() * torch.finfo(p.data.dtype).bits / 8
                if p.data.is_floating_point()
                else p.nelement() * torch.iinfo(p.data.dtype).bits / 8
            )
            for p in optimizer.param_groups[0]["params"]
        ]
    )
    act_size_byte = sum(
        [
            (
                a.nelement() * torch.finfo(a.data.dtype).bits / 8
                if a.data.is_floating_point()
                else a.nelement() * torch.iinfo(a.data.dtype).bits / 8
            )
            for a in acts
        ]
    )

    total_nb_elements = (
        model_param_size_byte
        + grad_size_byte
        + batch_size_byte
        + optimizer_size_byte
        + act_size_byte
    )
    total_mb = total_nb_elements / 1e6
    print(f"Training will use a total of around {total_mb:.2f} MB")
    return total_mb


def estimate_training_memory_usage_on_gpu(batch_size: int = 64, partition_id: int = 0):
    # https://discuss.pytorch.org/t/how-to-deal-with-excessive-memory-usages-of-pytorch/126098/5
    acts = []
    model = get_model()
    for name, module in model.named_modules():
        if name == "classifier" or name == "features":
            continue
        module.register_forward_hook(
            lambda m, input, output: acts.append(output.detach())
        )

    def get_model_size(model: torch.nn) -> float:
        "Get PyTorch model size in byte"
        size_model = 0
        for param in model.parameters():
            if param.data.is_floating_point():
                size_model += param.numel() * torch.finfo(param.data.dtype).bits
            else:
                size_model += param.numel() * torch.iinfo(param.data.dtype).bits
        return size_model / 8

    # execute single training step
    X, y_true = next(iter(load_global_test_data_onto_gpu()))
    # Forward pass
    y_hat = model(X)
    criterion = torch.nn.CrossEntropyLoss()
    loss = criterion(y_hat, y_true)
    # Backward pass
    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # approximate memory requirements
    model_param_size_byte = get_model_size(model=model)
    grad_size_byte = model_param_size_byte
    batch_size_byte = (
        32
        * 32
        * 3
        * batch_size
        * len(load_data_onto_gpu(partition_id=partition_id, batch_size=batch_size))
    )
    optimizer_size_byte = sum(
        [
            (
                p.nelement() * torch.finfo(p.data.dtype).bits / 8
                if p.data.is_floating_point()
                else p.nelement() * torch.iinfo(p.data.dtype).bits / 8
            )
            for p in optimizer.param_groups[0]["params"]
        ]
    )
    act_size_byte = sum(
        [
            (
                a.nelement() * torch.finfo(a.data.dtype).bits / 8
                if a.data.is_floating_point()
                else a.nelement() * torch.iinfo(a.data.dtype).bits / 8
            )
            for a in acts
        ]
    )

    total_nb_elements = (
        model_param_size_byte
        + grad_size_byte
        + batch_size_byte
        + optimizer_size_byte
        + act_size_byte
    )
    total_mb = total_nb_elements / 1e6
    print(f"Training will use a total of around {total_mb:.2f} MB")
    return total_mb
