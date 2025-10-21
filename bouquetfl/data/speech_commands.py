"google/speech_commands"

# Some general parameters we need
from typing import List, OrderedDict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torchvision
import yaml
from datasets import Dataset
from flwr.common.typing import NDArrays
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import (
    DirichletPartitioner,
    LinearPartitioner,
    PathologicalPartitioner,
    SizePartitioner,
    SquarePartitioner,
)
from torch.utils.data import DataLoader

# import data_utils


# Loading the model (Called when initializing FlowerClient and when testing)
def get_model() -> torch.nn.Module:
    return torch.hub.load(
        "pytorch/vision:v0.10.0", "Wav2Vec2Model", weights=None, verbose=False
    ).cuda()


def size_based_split(num_clients, num_high_quality):
    size_min, size_max = [
        max(256, 50000 / (num_clients * 3)),
        int(50000 / (num_clients * 2)),
    ]
    sizes = [np.random.randint(size_min, size_max) for _ in range(num_clients)]
    remaining_points = 50000 - np.sum(sizes[num_high_quality:])
    for i in range(num_high_quality):
        sizes[i] = int(remaining_points / num_high_quality)
    partitioner = SizePartitioner(
        partition_sizes=sizes,
    )
    fds = FederatedDataset(
        dataset="google/speech_commands",
        partitioners={"train": partitioner},
        trust_remote_code=True,
        subset="v0.01",
    )
    partitioner = fds.partitioners["train"]

    def plot_distributions(partitioner, labelname: str):
        fig, ax, df = plot_label_distributions(
            partitioner,
            label_name=labelname,
            plot_type="bar",
            size_unit="absolute",
            partition_id_axis="x",
            legend=True,
            verbose_labels=True,
            title="Per Partition speaker Distribution",
        )
        plt.savefig(f"./speech_distribution.pdf")

    plot_distributions(partitioner, "label")
    return fds


import matplotlib.pyplot as plt
from flwr_datasets.visualization import plot_label_distributions


def pathological_split(num_clients, alpha):
    partitioner = PathologicalPartitioner(
        num_partitions=num_clients,
        num_classes_per_partition=5,
        partition_by="speaker_id",
    )

    fds = FederatedDataset(
        dataset="google/speech_commands",
        partitioners={"train": partitioner},
        trust_remote_code=True,
    )
    partitioner = fds.partitioners["train"]

    def plot_distributions(partitioner, labelname: str):
        fig, ax, df = plot_label_distributions(
            partitioner,
            label_name=labelname,
            plot_type="bar",
            size_unit="absolute",
            partition_id_axis="x",
            legend=True,
            verbose_labels=True,
            title="Per Partition speaker Distribution",
        )
        plt.savefig(f"./speech_distribution.pdf")

    plot_distributions(partitioner, "speaker_id")
    return fds


size_based_split(10, 3)
