# For faster computation, we load datasets to the GPU as a dedicated CUDA_VisionDataSet

import logging
import time
from typing import Callable

logger = logging.getLogger(__name__)
import matplotlib.pyplot as plt
import numpy as np
import torch
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import (DirichletPartitioner,
                                       ExponentialPartitioner, SizePartitioner)

# We use flwr-datasets to create our federated dataset


# ['mnist', 'ylecun/mnist', 'cifar10', 'uoft-cs/cifar10', 'fashion_mnist', 'zalando-datasets/fashion_mnist', 'sasha/dog-food', 'zh-plus/tiny-imagenet', 'scikit-learn/adult-census-income', 'cifar100',
# 'uoft-cs/cifar100', 'svhn', 'ufldl-stanford/svhn', 'sentiment140', 'stanfordnlp/sentiment140', 'speech_commands', 'LIUM/tedlium', 'flwrlabs/femnist', 'flwrlabs/ucf101', 'flwrlabs/ambient-acoustic-context', 'jlh/uci-mushrooms', 'Mike0307/MNIST-M', 'flwrlabs/usps']


def generate_datasplit(num_clients, dataset, target_name):
    fds = FederatedDataset(
        dataset=dataset,
        partitioners={
            "train": SizePartitioner(
                [int(60000 / num_clients) for _ in range(num_clients)]
            ),
        },
    )
    logging.info("Dataset successfully split.")
    return fds


from dataclasses import dataclass


@dataclass
class TrainingCalls:
    get_model: Callable = None
    train: Callable = None
    test: Callable = None
    get_parameters: Callable = None
    set_parameters: Callable = None
    load_data: Callable = None
    load_global_test_data: Callable = None
    get_initial_parameters: Callable = None


@dataclass
class GlobalArgs:
    save_name: str = time.time()
    num_clients: int = 2
    epochs: int = 5
    seed: int = 0

    max_rounds: int = 5
    k: int = 5


def client_threshold(num_clients, min_clients, current_round):
    def sigm(x):
        return 1 / (1 + np.exp(-x))

    k = min_clients + int((num_clients) - ((num_clients) * sigm(current_round / 5)))
    return k


import time

from datasets import Dataset


def flip_indices(
    dataset: Dataset, target_name: str = "label", perc_malicious: float = 0.35
):
    t = time.time()
    old_labels = dataset[target_name]
    size, targets = len(old_labels), list(set(old_labels))
    malicious_indices = np.random.choice(
        size, size=int(perc_malicious * size), replace=False
    )
    for i in malicious_indices:
        old_labels[i] = np.random.choice([x for x in targets if x != old_labels[i]])
    print(f"generating new took {time.time() - t} seconds")
    dataset = (
        dataset.remove_columns(target_name)
        .add_column(target_name, old_labels)
        .cast(dataset.features)
    )
    return dataset


from typing import List, OrderedDict


def ndarrays_from_model(model: torch.nn.ModuleList) -> List[np.ndarray]:
    """Get model weights as a list of NumPy ndarrays."""
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def ndarrays_to_model(model: torch.nn.ModuleList, params: List[np.ndarray]):
    """Set model weights from a list of NumPy ndarrays."""
    params_dict = zip(model.state_dict().keys(), params)
    state_dict = OrderedDict({k: torch.from_numpy(np.copy(v)) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)


import matplotlib.pyplot as plt
from flwr_datasets.visualization import plot_label_distributions


def plot_distributions(partitioner, labelname: str, dsname: str):
    fig, ax, df = plot_label_distributions(
        partitioner,
        label_name=labelname,
        plot_type="bar",
        size_unit="absolute",
        partition_id_axis="x",
        legend=True,
        verbose_labels=True,
        title="Per Partition Labels Distribution",
    )

    plt.savefig(f"./plots/label_distribution_{dsname}.pdf")
