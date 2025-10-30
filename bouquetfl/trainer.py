import argparse
import os
import time
import timeit

import pandas as pd
import numpy as np
import torch
from flwr.common.parameter import parameters_to_ndarrays

from bouquetfl import power_clock_tools as pct

os.environ["HF_DATASETS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import pyarrow as pa

pa.set_cpu_count(1)

# Arguments passed by client.py

parser = argparse.ArgumentParser(
    description="Train a client-specific model with specific hardware settings."
)
parser.add_argument(
    "--experiment", type=str, default="cifar100", help="Dataset to use for training."
)
parser.add_argument("--client_id", type=int, default=0, help="Client ID.")
parser.add_argument(
    "--global_model_load_path",
    type=str,
    default="model.npz",
    help="Path to load the global model.",
)
parser.add_argument(
    "--local_model_save_path",
    type=str,
    default="client_model.npz",
    help="Path to save the local model.",
)
parser.add_argument(
    "--gpu_name",
    type=str,
    default="GeForce RTX 3050 8 GB",
    help="Name of the GPU to simulate (must be in hardwareconf/gpus.csv).",
)
parser.add_argument(
    "--cpu_name",
    type=str,
    default="Ryzen 3 1200",
    help="Name of the CPU to simulate (must be in hardwareconf/cpus.csv).",
)
parser.add_argument(
    "--round",
    type=int,
    default=1,
    help="Round number.",
)
parser.add_argument(
    "--num_rounds",
    type=int,
    default=1,
    help="Total number of rounds.",
)
# Load dataset-specific configurations and training calls

args = parser.parse_args()
if args.experiment == "cifar100":
    from bouquetfl.data import cifar100 as flower_baseline

elif args.experiment == "flowertune_llm":
    from bouquetfl.data import flowertune_llm as flower_baseline


elif args.experiment == "tiny_imagenet":
    from bouquetfl.data import tiny_imagenet as flower_baseline

else:
    raise ValueError("Please specify a dataset and model.")

####################################
############# Training #############
####################################


def train_model():
    client_id = args.client_id
    global_model_load_path = args.global_model_load_path
    local_model_save_path = args.local_model_save_path

    # Load model and apply global parameters
    model = flower_baseline.get_model()
    try:
        ndarrays_original = np.load(
            "./bouquetfl/checkpoints/global_params.npz", allow_pickle=True
        )
        ndarrays_original = [ndarrays_original[key] for key in ndarrays_original]
    except FileNotFoundError:
        model_parameters = flower_baseline.get_initial_parameters()
        ndarrays_original = parameters_to_ndarrays(model_parameters)

    # Set hardware limits (Ram limit was set in the subprocess environement)
    pct.set_physical_gpu_limits(args.gpu_name)
    num_cpu_cores = pct.set_cpu_limit(args.cpu_name)

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
        epochs=1,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    train_time = timeit.default_timer() - start_train_time

    # Reset hardware limits
    pct.reset_all_limits()

    # Save updated model parameters
    ndarrays_new = flower_baseline.ndarrays_from_model(model)
    np.savez(f"./bouquetfl/checkpoints/params_updated_{client_id}.npz", *ndarrays_new)

    # Save load and training times
    print(f"Data loading time: {data_load_time} seconds")
    print(f"Training time: {train_time} seconds")
    try:
        df = pd.read_pickle("./bouquetfl/checkpoints/load_and_training_times.pkl")
    except FileNotFoundError:
        df = pd.DataFrame(
            columns=[
                "gpu",
                "cpu"
            ]
            + [f"load_time_{i}" for i in range(1, args.num_rounds + 1)] 
            + [f"train_time_{i}" for i in range(1, args.num_rounds + 1)]
        )
    df.at[client_id, "gpu"] = args.gpu_name
    df.at[client_id, "cpu"] = args.cpu_name
    df.at[client_id, f"load_time_{args.round}"] = data_load_time
    df.at[client_id, f"train_time_{args.round}"] = train_time
    df.to_pickle("./bouquetfl/checkpoints/load_and_training_times.pkl")

train_model()
