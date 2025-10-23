import argparse
import os
import time
import timeit

import numpy as np
from bouquetfl import power_clock_tools as pct
import torch
from flwr.common.parameter import parameters_to_ndarrays

import bouquetfl.resource_utils as resource_utils
from bouquetfl.data import data_utils

os.environ["HF_DATASETS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import pyarrow as pa

from bouquetfl.task import Net, get_weights
pa.set_cpu_count(1)
# Arguments:

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
    "--model_save_path",
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
# Load dataset-specific configurations and training calls

global_args = data_utils.GlobalArgs()

args = parser.parse_args()
if args.experiment == "cifar100":
    from bouquetfl.data import cifar100 as flower_baseline

    global_args.num_clients = 36
    global_args.max_rounds = 80
    global_args.draw_threshold = 0.025
    global_args.min_clients = 6

elif args.experiment == "flowertune_llm":
    from bouquetfl.data import flowertune_llm as flower_baseline

    global_args.num_clients = 8
    global_args.max_rounds = 12
    global_args.draw_threshold = 0.001
    global_args.min_clients = 4

elif args.experiment == "tiny_imagenet":
    from bouquetfl.data import tiny_imagenet as flower_baseline

    global_args.num_clients = 60
    global_args.max_rounds = 35
    global_args.draw_threshold = 0.025
    global_args.min_clients = 10

else:
    raise ValueError("Please specify a dataset and model.")

####################################
############# Training #############
####################################


def train_model():
    client_id = args.client_id
    global_model_load_path = args.global_model_load_path
    model_save_path = args.model_save_path

    model = flower_baseline.get_model()
    try:
        model_parameters = np.load(
            "./bouquetfl/checkpoints/global_params.npy", allow_pickle=True
        )
    except FileNotFoundError:
        model_parameters = flower_baseline.get_initial_parameters()
    # model_parameters = [model_parameters[key] for key in model_parameters.keys()]
    model_parameters = parameters_to_ndarrays(model_parameters)
    pct.set_physical_gpu_limits(args.gpu_name)
    num_cpu_cores = pct.set_cpu_limit(args.cpu_name)
    # set_ram_limit(args.ram_size)
    # resource_utils.start_collection()
    time.sleep(0.5)
    start_data_load_time = timeit.default_timer()
    trainloader = flower_baseline.load_data(client_id, num_workers=num_cpu_cores)
    data_load_time = timeit.default_timer() - start_data_load_time
    print(f"Data loading time: {data_load_time} seconds")

    start_train_time = timeit.default_timer()
    flower_baseline.ndarrays_to_model(model, model_parameters)
    flower_baseline.train(
        model=model,
        trainloader=trainloader,
        epochs=25,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    model_parameters = flower_baseline.ndarrays_from_model(model)
    train_time = timeit.default_timer() - start_train_time

    print(f"Training time: {train_time} seconds")
    # resources = resource_utils.stop_collection()
    # np.savez(f"./bouquetfl/checkpoints/resources_client_{client_id}.npz", resources)
    # print(f"Resources used: {resources}")
    pct.reset_all_limits()
    if not model_save_path:
        np.savez(
            f"./bouquetfl/checkpoints/params_updated_{client_id}.npz", *model_parameters
        )
    else:
        np.savez(model_save_path, *model_parameters)


train_model()
