import logging
import time

import h5py
import numpy
import pandas as pd
import power_clock_tools as pct
import torch
from data import cifar100

logger = logging.getLogger(__name__)
import os
import subprocess

import numpy as np
import power_clock_tools as pct
import torch
from flwr.client import Client
from flwr.common import (
    Code,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Status,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)

from .data import data_utils

#####################################
############# GPU tools #############
#####################################


def set_gpu_limit(gpu_name: str):
    gpu_info = pct.get_gpu_info(gpu_name)
    if not gpu_info:
        raise ValueError(f"GPU {gpu_name} not found in database.")
    current_gpu_info = pct.get_current_gpu_info()

    if gpu_info["memory"] > int(current_gpu_info["memory"]):
        raise ValueError(
            f"GPU {gpu_name} has more memory ({gpu_info['memory']} GB) than the current GPU ({current_gpu_info['memory']} GB)."
        )
    if gpu_info["clock speed"] > int(current_gpu_info["clock speed"]):
        raise ValueError(
            f"GPU {gpu_name} has a higher clock speed ({gpu_info['clock speed']} MHz) than the current GPU ({current_gpu_info['clock speed']} MHz)."
        )
    if gpu_info["memory speed"] > int(current_gpu_info["memory speed"]):
        raise ValueError(
            f"GPU {gpu_name} has a higher memory speed ({gpu_info['memory speed']} MHz) than the current GPU ({current_gpu_info['memory speed']} MHz)."
        )

    pct.set_gpu_memory_limit(gpu_info["memory"], 0)
    pct.lock_gpu_clocks(
        0, int(gpu_info["clock speed"]), int(gpu_info["clock speed"] + 10)
    )
    pct.lock_gpu_memory_clocks(
        0, int(gpu_info["memory speed"]), int(gpu_info["memory speed"] + 10)
    )
    print(f"Set memory limit to {gpu_info['memory']} GB")
    print(f"Set clock speed to {gpu_info['clock speed']} MHz")
    print(f"Set memory speed to {gpu_info['memory speed']} MHz")


def create_cuda_restricted_env(gpu_name: str, current_cores: int):

    gpu_info = pct.get_gpu_info(gpu_name)
    target_cores = gpu_info["cuda cores"]
    pct.start_mps()
    env = os.environ.copy()
    env["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = str(100 * target_cores / current_cores)
    return env


######################################
############# CPU tools #############
######################################


def set_cpu_limit(cpu_name: str):
    cpu_info = pct.get_cpu_info(cpu_name)
    if not cpu_info:
        raise ValueError(f"CPU {cpu_name} not found in database.")
    current_cpu_info = pct.get_current_cpu_info()

    if cpu_info["cores"] > int(current_cpu_info["cores"]):
        raise ValueError(
            f"CPU {cpu_name} has more cores ({cpu_info["cores"]}) than the current CPU ({current_cpu_info['cores']})."
        )
    if cpu_info["turbo clock"] > int(current_cpu_info["clock speed"]):
        raise ValueError(
            f"CPU {cpu_name} has a higher clock speed ({cpu_info['turbo clock']} MHz) than the current CPU ({current_cpu_info['clock speed']} MHz)."
        )

    pct.set_cpu_limit(int(cpu_info["turbo clock"]))
    print(f"Set CPU clock speed to {cpu_info['turbo clock']} MHz")
    return cpu_info["cores"]


######################################
############# RAM tools #############
######################################


def set_ram_limit(ram_size: int):
    pct.limit_ram(ram_size)
    print(f"Set RAM limit to {ram_size} GB")


######################################
########## Reset all limits ##########
######################################


def reset_all_limits():
    pct.reset_gpu_memory_limit(), pct.reset_gpu_clocks(0), pct.reset_gpu_memory_clocks(
        0
    ), pct.reset_ram_limit(), pct.reset_cpu_limit()
    print("Reset memory limit and clock speeds to default")


######################################
########## Nvidia MPS tools ##########
######################################


def start_mps():
    mps_proc = subprocess.Popen(["nvidia-cuda-mps-control", "-d"])
    mps_proc.wait()
    print("MPS server has started.")


def stop_mps():
    mps_proc = subprocess.Popen("echo quit | nvidia-cuda-mps-control")
    mps_proc.wait()
    print("MPS server has exited.")


#####################################
########## Flower Client ############
#####################################


class FlowerClient(Client):
    def __init__(
        self,
        client_id: int,
        training_calls: data_utils.TrainingCalls,
        oracle: bool = False,
    ) -> None:
        self.client_id = client_id
        self.training_calls = training_calls
        self.oracle = (oracle,)
        if not self.oracle:
            self.num_examples = 1
        else:
            self.num_examples = len(self.training_calls.load_data(self.client_id))

        self.round = 0
        self.gpu_name: str = ""
        self.cpu_name: str = ""
        self.ram_size: int = 0
        self.current_cores: int = 1
        self.model_load_path: str = "params.h5"
        self.model_save_path: str = f"model_client_{self.client_id}.h5"

    def fit(self, ins: FitIns) -> FitRes:

        # Deserialize parameters to NumPy ndarray's
        parameters_original = ins.parameters
        ndarrays_original = parameters_to_ndarrays(parameters_original)

        with h5py.File(self.model_load_path, "w") as f:
            f.create_dataset("params", data=ndarrays_original)

        def limit_resources():
            set_gpu_limit(self.gpu_name)
            set_ram_limit(self.ram_size)
            set_cpu_limit(self.cpu_name)

        env = create_cuda_restricted_env(self.gpu_name, self.current_cores)

        start_mps()
        subprocess.run(
            [
                "uv run",
                "mpstrainer.py",
                "--experiment",
                "cifar100",
                "--client_id",
                f"{self.client_id}",
                "--model_load_path",
                self.model_load_path,
                "--model_save_path",
                self.model_save_path,
            ],
            env=env,
            preexec_fn=limit_resources,
        )

        stop_mps()
        with h5py.File(self.model_save_path, "r") as f:
            ndarrays_updated = f["params"][:]
        # Serialize ndarray's into a Parameters object
        parameters_updated = ndarrays_to_parameters(ndarrays_updated)
        # Build and return response
        status = Status(code=Code.OK, message="Success")
        logging.info(f"Client {self.client_id} successfully trained.")
        return FitRes(
            status=status,
            parameters=parameters_updated,
            num_examples=self.num_examples,
            metrics={"client_id": self.client_id},
        )

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        status = Status(code=Code.OK, message="Success")
        return EvaluateRes(
            status=status,
            loss=np.inf,
            num_examples=self.num_examples,
            metrics={"accuracy": 0},
        )
