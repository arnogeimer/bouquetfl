"""bouquetfl: A Flower / PyTorch app."""

import csv
import logging
import time

import h5py
import numpy as np
import pandas as pd
from bouquetfl import power_clock_tools as pct
import torch
from flwr.client import ClientApp, Client
from flwr.common import (
    Code,
    Context,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Status,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)

logger = logging.getLogger(__name__)
import os
import subprocess
from datetime import datetime

import psutil
import torch

"""from pynvml import (nvmlDeviceGetComputeRunningProcesses, nvmlDeviceGetCount,
                    nvmlDeviceGetHandleByIndex, nvmlDeviceGetName, nvmlInit,
                    nvmlShutdown)"""


def create_cuda_restricted_env(gpu_name: str, current_cores: int):

    gpu_info = pct.get_gpu_info(gpu_name)
    target_cores = gpu_info["cuda cores"]
    env = os.environ.copy()
    env["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = str(100 * target_cores / current_cores)
    return env


######################################
########## Nvidia MPS tools ##########
######################################


def start_mps():
    mps_proc = subprocess.Popen(["nvidia-cuda-mps-control", "-d"])
    mps_proc.wait()
    print("MPS server has started.")


def stop_mps():
    mps_proc = subprocess.Popen(["echo", "quit", "|", "nvidia-cuda-mps-control"])
    mps_proc.wait()
    print("MPS server has exited.")


#####################################
########## Flower Client ############
#####################################


# Define Flower Client and client_fn
class FlowerClient(Client):
    def __init__(
        self,
        client_id: int,
    ) -> None:
        self.client_id = client_id
        self.num_examples = 1
        self.gpu_name: str = "GeForce 210"
        self.cpu_name: str = "Ryzen 3 1200"
        self.ram_size: int = 2
        self.current_cores: int = 10240
        self.global_model_load_path: str = ""
        self.local_model_save_path: str = (
            f"./bouquetfl/checkpoints/params_updated_{client_id}.npz"
        )

    # def fit(self, ins: FitIns) -> FitRes:
    def fit(self, ins: FitIns) -> FitRes:
        # Deserialize parameters to NumPy ndarray's
        ndarrays_original = parameters_to_ndarrays(ins.parameters)
        np.savez(
            "./bouquetfl/checkpoints/global_params.npz", *ndarrays_original
        )

        env = create_cuda_restricted_env(self.gpu_name, self.current_cores)

        start_mps()
        # We run trainer.py in a separate process with systemd-run using set CUDA_MPS_ACTIVE_THREAD_PERCENTAGE.
        # Anything else (CPU throttling, RAM limiting, GPU memory and clock limiting) could be done without a separate process.
        # Theoretically, one could implement ones own model in cuda/triton, physically setting the grid on which the model is allowed to run.
        # This way, one could limit the GPU usage without MPS. However, this would require a lot of work and is not implemented here.
        # We take advantage of systemd-run to limit the RAM usage of the process.

        child = subprocess.Popen(
            [
                "systemd-run",
                "--user",
                "--scope",
                "-p",
                f"MemoryMax={self.ram_size}G",
                "uv",  # <--- Change this to "python3" or corresponding if you don't have uv installed
                "run",
                "./bouquetfl/trainer.py",
                "--experiment",
                "cifar100",
                "--client_id",
                f"{self.client_id}",
                "--global_model_load_path",
                self.global_model_load_path,
                "--local_model_save_path",
                self.local_model_save_path,
                "--gpu_name",
                f"{self.gpu_name}",
                "--cpu_name",
                f"{self.cpu_name}",
            ],
            env=env,
        )
        pid = child.pid
        print("Child PID:", pid)

        child.wait()
        try:
            ndarrays_new = np.load(self.local_model_save_path, allow_pickle=True)
            ndarrays_new = [ndarrays_new[key] for key in ndarrays_new]
            os.remove(self.local_model_save_path)
            # Serialize ndarray's into a Parameters object
            #parameters_updated = ndarrays_to_parameters(ndarrays_updated)
            # Build and return response
            status = Status(code=Code.OK, message="Success")
            logging.info(f"Client {self.client_id} successfully trained.")

        except FileNotFoundError:
            logging.info(
                f"Model file {self.local_model_save_path} not found. Training on client {self.client_id} might have failed."
            )
            status = Status(code=Code.FIT_NOT_IMPLEMENTED, message="Training failed.")
            parameters_updated = None
        parameters_updated = ndarrays_to_parameters(ndarrays_new)
        #print("HERE    >>>>>>>",ndarrays_new, self.num_examples, {})
        #return ndarrays_new, self.num_examples, {}
        return FitRes(
            status=status,
            parameters=parameters_updated,
            num_examples=self.num_examples,
            metrics={"client_id": self.client_id},
        )

    def evaluate(self, ins: EvaluateIns):
        from bouquetfl.data import cifar100 as flower_baseline
        testset = flower_baseline.load_global_test_data()
        model = flower_baseline.get_model()
        ndarrays = parameters_to_ndarrays(ins.parameters)
        flower_baseline.ndarrays_to_model(model, ndarrays)
        loss, accuracy = flower_baseline.test(
            model=model,
            testloader=testset,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        status = Status(code=Code.OK, message="Success")
        return EvaluateRes(
            status=status,
            loss=loss,
            num_examples=self.num_examples,
            metrics={"accuracy": accuracy}
        )


def client_fn(context: Context):
    # Return Client instance
    print(context)
    return FlowerClient(client_id=context.node_config['partition-id']).to_client()

# Flower ClientApp
app = ClientApp(
    client_fn,
)

'''
gpus = [
    # "GeForce RTX 3070",
    "GeForce GTX 1660 SUPER",
]
ram_sizes = [1, 0.5, 0.25]  # in GB

for i in range(len(gpus)):
    print(f"Starting client {i} with GPU {gpus[i]}")
    x = FlowerClient(i)
    x.client_id = i
    x.gpu_name = gpus[i]
    x.cpu_name = "Ryzen 3 1200"
    x.ram_size = ram_sizes[i]
    # IMPORTANT: FIND OUT CURRENT CORES OF THE GPU
    x.current_cores = 10240
    parameters = cifar100.get_initial_parameters()
    x.fit(parameters)
'''