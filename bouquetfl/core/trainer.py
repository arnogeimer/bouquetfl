import argparse
import os
import time
import timeit

import numpy as np
import pandas as pd
import torch
from flwr.common.parameter import parameters_to_ndarrays

from bouquetfl.utils import power_clock_tools as pct
from bouquetfl.utils.filesystem import (
    load_client_hardware_config,
    save_load_and_training_times,
    save_ndarrays,
)

os.environ["HF_DATASETS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import importlib

import pyarrow as pa

pa.set_cpu_count(1)

# Arguments passed by client.py

parser = argparse.ArgumentParser(
    description="Train a client-specific model with specific hardware settings."
)
args_list = [
    (
        "--experiment",
        {"type": str, "default": "cifar100", "help": "Dataset to use for training."},
    ),
    ("--client_id", {"type": int, "default": 0, "help": "Client ID."}),
    ("--round", {"type": int, "default": 1, "help": "    Round number."}),
    ("--num_rounds", {"type": int, "default": 1, "help": "Total number of rounds."}),
]
for arg, kwargs in args_list:
    parser.add_argument(arg, **kwargs)
args = parser.parse_args()

# Load dataset-specific configurations and training calls

modules = {
    "cifar100": "task.cifar100",
    "flowertune_llm": "task.flowertune_llm",
    "tiny_imagenet": "task.tiny_imagenet",
}

if args.experiment in modules:
    flower_baseline = importlib.import_module(modules[args.experiment])
else:
    raise ValueError("Please specify a dataset and model.")

####################################
############# Training #############
####################################


def train_model():
    client_id = args.client_id
    gpu, cpu, _ = load_client_hardware_config(client_id)
    # Load model and apply global parameters
    model = flower_baseline.get_model()
    try:
        ndarrays_original = np.load(
            f"checkpoints/global_params_round_{args.round}.npz",
            allow_pickle=True,
        )
        ndarrays_original = [ndarrays_original[key] for key in ndarrays_original]
    except FileNotFoundError:
        model_parameters = flower_baseline.get_initial_parameters()
        ndarrays_original = parameters_to_ndarrays(model_parameters)

    # Set hardware limits (Ram limit was set in the subprocess environement)
    pct.set_physical_gpu_limits(gpu)
    num_cpu_cores = pct.set_cpu_limit(cpu)
    # Give some time for the limits to take effect
    time.sleep(0.5)

    # Load data (on CPU)
    start_data_load_time = timeit.default_timer()
    trainloader = flower_baseline.load_data(client_id, num_workers=num_cpu_cores)
    data_load_time = timeit.default_timer() - start_data_load_time

    # Train model (on GPU)
    start_train_time = timeit.default_timer()
    flower_baseline.ndarrays_to_model(model, ndarrays_original)
    flower_baseline.train(
        model=model,
        trainloader=trainloader,
        epochs=5,
        device=("cuda" if torch.cuda.is_available() and gpu != "None" else "cpu"),
    )
    train_time = timeit.default_timer() - start_train_time

    # Save updated model parameters
    save_ndarrays(
        flower_baseline.ndarrays_from_model(model),
        f"checkpoints/params_updated_{client_id}.npz",
    )

    # Save load and training times
    save_load_and_training_times(
        client_id=client_id,
        round=args.round,
        gpu=gpu,
        cpu=cpu,
        data_load_time=data_load_time,
        train_time=train_time,
        num_rounds=args.num_rounds,
    )


train_model()
