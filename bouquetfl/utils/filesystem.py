import os

import numpy as np
import pandas as pd
import yaml
from flwr.common import Code, Status, ndarrays_to_parameters
from flwr.common.typing import Parameters, FitRes


def delete_unused_files():

    files = os.listdir("checkpoints/")
    for file in files:
        if file.endswith(".npz"):
            os.remove(os.path.join("checkpoints/", file))

    files = os.listdir("config/")
    for file in files:
        if file.endswith(".yaml"):
            os.remove(os.path.join("config/", file))

    files = os.listdir("plots/")
    for file in files:
        os.remove(os.path.join("plots/", file))


def save_ndarrays(ndarrays: list[np.ndarray], savename: str) -> None:
    #if not os.path.exists(savename):
    np.savez(
        savename,
        *ndarrays,
    )


def load_new_client_parameters(client_id: int) -> tuple[Status, Parameters]:
    """Load the updated model parameters for a given client after local training. Found in FlowerClient.fit"""
    local_save_path = f"/tmp/params_updated_{client_id}.npz"
    try:
        ndarrays_new = np.load(local_save_path, allow_pickle=True)
        ndarrays_new = [ndarrays_new[key] for key in ndarrays_new]
        os.remove(local_save_path)
        if len(ndarrays_new) == 0:
            # If OutOfMemory
            print(f"Client {client_id} has encountered an out-of-memory error.")
            status = Status(code=Code.FIT_NOT_IMPLEMENTED, message="Training failed.")
            return status, Parameters(tensor_type="", tensors=[])
        # Build and return response
        status = Status(code=Code.OK, message="Success")
        # Serialize ndarray's into a Parameters object
        parameters_updated = ndarrays_to_parameters(ndarrays_new)

    except FileNotFoundError:
        status = Status(code=Code.FIT_NOT_IMPLEMENTED, message="Training failed.")
        parameters_updated = Parameters(tensor_type="", tensors=[])
    return status, parameters_updated


def load_client_hardware_config(client_id: int) -> tuple[str, str, int]:
    """Load the hardware configuration for a given client from YAML file. Found in FlowerClient.fit and trainer.py"""
    try:
        with open("config/federation_client_hardware.yaml", "r") as f:
            client_config = yaml.safe_load(f)
            gpu = client_config[f"client_{client_id}"]["gpu"]
            cpu = client_config[f"client_{client_id}"]["cpu"]
            ram = client_config[f"client_{client_id}"]["ram_gb"]
    except FileNotFoundError:
        raise ValueError("Client hardware configuration file not found.")
    print(f"Client {client_id} hardware: GPU={gpu}, CPU={cpu}, RAM={ram}GB")
    return gpu, cpu, ram


def save_load_and_training_times(
    client_id: int,
    round: int,
    gpu: str,
    cpu: str,
    data_load_time: float,
    train_time: float,
    num_rounds: int,
) -> None:
    """Save the data load and training times for a given client and round to a pickle file. Found in trainer.py"""
    try:
        df = pd.read_pickle("checkpoints/load_and_training_times.pkl")
    except FileNotFoundError:
        df = pd.DataFrame(
            index=range(0, 100),
            columns=["gpu", "cpu"]
            + [f"load_time_{i}" for i in range(1, num_rounds + 1)]
            + [f"train_time_{i}" for i in range(1, num_rounds + 1)],
        )
    df.at[client_id, "gpu"] = gpu
    df.at[client_id, "cpu"] = cpu
    df.at[client_id, f"load_time_{round}"] = data_load_time
    df.at[client_id, f"train_time_{round}"] = train_time
    df.to_pickle("checkpoints/load_and_training_times.pkl")
