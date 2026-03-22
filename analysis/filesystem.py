import os

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import yaml
from flwr.common import Code, Status
from flwr.common.typing import Parameters
import torch

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


def load_new_client_state_dict(client_id: int) -> tuple[Status, Parameters]:
    """Load the updated model state dict for a given client after local training. Found in FlowerClient.fit"""
    local_save_path = f"/tmp/params_updated_{client_id}.tp"
    try:
        state_dict_new = torch.load(local_save_path, weights_only=True)
        os.remove(local_save_path)
        status = Status(code=Code.OK, message="Success")

    except FileNotFoundError:
        state_dict_new = None
        status = Status(code=Code.FIT_NOT_IMPLEMENTED, message="Training failed.")
    return status, state_dict_new


def load_client_hardware_config(client_id: int) -> tuple[str, str, int]:
    """Load the hardware configuration for a given client from YAML file. Found in FlowerClient.fit and trainer.py"""
    try:
        with open("federation_client_hardware.toml", "rb") as f:
            client_config = yaml.safe_load(f)
            gpu = client_config[f"client_{client_id}"]["gpu"]
            cpu = client_config[f"client_{client_id}"]["cpu"]
            ram = client_config[f"client_{client_id}"]["ram_gb"]
    except FileNotFoundError:
        raise ValueError("Client hardware configuration file not found.")
    print(f"{"\033[31m"}Client {client_id} hardware{"\033[0m"}: GPU={gpu}, CPU={cpu}, RAM={ram}GB")
    return gpu, cpu, ram


