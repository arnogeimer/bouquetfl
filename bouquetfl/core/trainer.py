import argparse
import importlib
import json
import os
import time
import timeit

import numpy as np
import pandas as pd
import pyarrow as pa
import torch
from flwr.common.parameter import parameters_to_ndarrays

from bouquetfl.utils import power_clock_tools as pct
from bouquetfl.utils.filesystem import (load_client_hardware_config,
                                        save_load_and_training_times,
                                        save_ndarrays)

os.environ["HF_DATASETS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
pa.set_cpu_count(1)

# Arguments passed by client.py

parser = argparse.ArgumentParser(
    description="Train a client-specific model with specific hardware settings."
)
args_list = [
    ("--client_id", {"type": int, "default": 0, "help": "Client ID."}),
    ("--config", {"type": str, "default": "", "help": "Path to config from pyproject.toml"})
]
for arg, kwargs in args_list:
    parser.add_argument(arg, **kwargs)
args = parser.parse_args()

with open(args.config) as f:
    cfg = json.load(f)

# Load dataset-specific configurations and training calls

modules = {
    "cifar100": "task.cifar100",
    "flowertune_llm": "task.flowertune_llm",
    "tiny_imagenet": "task.tiny_imagenet",
}

if cfg["experiment"] in modules:
    flower_baseline = importlib.import_module(modules[cfg["experiment"]])
else:
    raise ValueError("Please specify a dataset and model.")

from task import cifar100 as mltask
####################################
############# Training #############
####################################


def train_model():
    client_id = args.client_id
    gpu, cpu, _ = load_client_hardware_config(client_id)
    # Load model and apply global parameters
    model = mltask.get_model()
    try:
        ndarrays_original = np.load(
            f"checkpoints/global_params_round_{cfg["server_round"]}.npz",
            allow_pickle=True,
        )
        ndarrays_original = [ndarrays_original[key] for key in ndarrays_original]
    except FileNotFoundError:
        model_parameters = mltask.get_initial_parameters()
        ndarrays_original = parameters_to_ndarrays(model_parameters)

    # Set hardware limits (Ram limit was set in the subprocess environement)
    pct.set_physical_gpu_limits(gpu)
    num_cpu_cores = pct.set_cpu_limit(cpu)
    # Give some time for the limits to take effect
    time.sleep(0.5)

    # Load data (on CPU)
    start_data_load_time = timeit.default_timer()
    trainloader = mltask.load_data(client_id, num_clients = cfg["num-clients"], num_workers=num_cpu_cores, batch_size=cfg["batch_size"])
    data_load_time = timeit.default_timer() - start_data_load_time

    # Train model (on GPU)
    start_train_time = timeit.default_timer()
    mltask.ndarrays_to_model(model, ndarrays_original)
    try:
        mltask.train(
            model=model,
            trainloader=trainloader,
            epochs=cfg["local-epochs"],
            device=("cuda" if torch.cuda.is_available() and gpu != "None" else "cpu"),
            lr=cfg["learning_rate"],
        )
        train_time = timeit.default_timer() - start_train_time
        # Save updated model parameters
        save_ndarrays(
            mltask.ndarrays_from_model(model),
            f"/tmp/params_updated_{client_id}.npz",
        )

    except torch.OutOfMemoryError:
        print(f"Client {client_id} has encountered an out-of-memory error.")
        train_time = None
        save_ndarrays(
            [],
            f"/tmp/params_updated_{client_id}.npz",
        )


    # Save load and training times
    save_load_and_training_times(
        client_id=client_id,
        round=cfg["server_round"],
        gpu=gpu,
        cpu=cpu,
        data_load_time=data_load_time,
        train_time=train_time,
        num_rounds=cfg["num-server-rounds"],
    )


train_model()
