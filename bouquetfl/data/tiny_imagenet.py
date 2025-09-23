# Some general parameters we need
from typing import List, OrderedDict, Tuple

import numpy as np
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
    return torch.hub.load(
        "pytorch/vision:v0.10.0", "densenet121", weights=None, verbose=False
    ).cuda()


transform_train = torchvision.transforms.Compose(
    [
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)

transform_test = torchvision.transforms.Compose(
    [
        torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)


def size_based_split(num_clients, num_high_quality):
    size_min, size_max = [
        max(256, 100000 / (num_clients * 3)),
        int(100000 / (num_clients * 2)),
    ]
    sizes = [np.random.randint(size_min, size_max) for _ in range(num_clients)]
    remaining_points = 100000 - np.sum(sizes[num_high_quality:])
    for i in range(num_high_quality):
        sizes[i] = int(remaining_points / num_high_quality)
    partitioner = SizePartitioner(
        partition_sizes=sizes,
    )
    return FederatedDataset(
        dataset="zh-plus/tiny-imagenet",
        partitioners={"train": partitioner},
        trust_remote_code=True,
    )


def dirichlet_based_split(num_clients, alpha):
    partitioner = DirichletPartitioner(
        num_partitions=num_clients,
        partition_by="label",
        alpha=alpha,
        min_partition_size=30,
    )
    return FederatedDataset(
        dataset="zh-plus/tiny-imagenet",
        partitioners={"train": partitioner},
        trust_remote_code=True,
    )


with open("./config/tiny_imagenetsplit_cfg.yaml", "r") as f:
    split_config = yaml.safe_load(f)

if split_config["split"] == "size":
    fds = size_based_split(split_config["num_clients"], split_config["num_quality"])

SIZES = [
    len(fds.load_partition(partition_id=partition_id))
    for partition_id in range(split_config["num_clients"])
]


def load_data(partition_id: int, batch_size: int = 64) -> DataLoader:
    dataset = fds.load_partition(partition_id=partition_id)

    def apply_transforms(batch):
        """Apply transforms to the partition from FederatedDataset."""
        batch["image"] = [
            torchvision.transforms.ToTensor(transform_train(img))
            for img in batch["image"]
        ]
        return batch

    dataset = dataset.with_transform(apply_transforms)

    trainloader = DataLoader(
        dataset=dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )
    return trainloader


def load_data_onto_gpu(partition_id: int, batch_size: int = 64) -> DataLoader:
    dataset = fds.load_partition(partition_id=partition_id)
    trainloader = DataLoader(
        CUDA_VisionDataSet(
            dataset,
            transform=transform_train,
            image_name="image",
            target_name="label",
        ),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )
    return trainloader


# Load the (global) test dataset
def load_global_test_data(batch_size: int = 64) -> DataLoader:
    testset = fds.load_split("valid")

    def apply_test_transform(batch):
        """Apply transforms to the partition from FederatedDataset."""
        batch["image"] = [
            torchvision.transforms.ToTensor(transform_test(img))
            for img in batch["image"]
        ]
        return batch

    testset = testset.with_transform(apply_test_transform)

    testloader = DataLoader(
        testset, batch_size=batch_size, shuffle=True, drop_last=True
    )
    return testloader


def load_global_test_data_onto_gpu(batch_size: int = 64) -> DataLoader:
    testset = fds.load_split("valid")
    testloader = DataLoader(
        CUDA_VisionDataSet(
            testset,
            transform=transform_test,
            image_name="image",
            target_name="label",
        ),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
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
            images, labels = batch["image"].to(device), batch["label"].to(device)
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
            images, labels = batch["image"].to(device), batch["label"].to(device)
            outputs = model(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    del (testloader, model)
    torch.cuda.empty_cache()
    return loss, accuracy


def train_on_gpu(model: nn.Module, trainloader: DataLoader, epochs: int = 5, **kwargs):
    """Train the model on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    print("Starting training")
    current_round = 1
    optimizer = torch.optim.Adam(
        params=model.parameters(), lr=1e-3 * 0.99**current_round
    )
    for _ in tqdm.trange(epochs):
        for _, (images, labels) in enumerate(trainloader):
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()


def test_on_gpu(
    model: nn.Module,
    testloader: DataLoader,
) -> Tuple[float, float]:
    """Validate the model on the entire test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    if len(testloader) == 0:
        return np.inf, 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0], data[1]
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


from flwr.common import ndarrays_to_parameters


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


def fit_config(server_round: int):
    """Generate training configuration for each round."""
    # Create the configuration dictionary
    config = {
        "current_round": server_round,
    }
    return config


client_resources: dict = {
    "num_cpus": 1,
    "num_gpus": 0.33333,
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
