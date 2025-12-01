"""bouquetfl: A Flower / PyTorch app."""


import logging

import numpy as np
from flwr.client import Client, ClientApp
from flwr.common import (Code, Context, EvaluateIns, EvaluateRes, FitIns,
                         FitRes, Status, ndarrays_to_parameters,
                         parameters_to_ndarrays)

from bouquetfl import power_clock_tools as pct

logger = logging.getLogger(__name__)
import os
import subprocess
import yaml
import torch
from hardwareconf.sampler import generate_hardware_sample

def create_cuda_restricted_env(gpu_name: str, current_cores: int):

    gpu_info = pct.get_gpu_info(gpu_name)
    target_cores = gpu_info["cuda cores"]
    env = os.environ.copy()
    env["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = str(100 * target_cores / current_cores)
    print(f"Creating CUDA MPS env with {target_cores} cores out of {current_cores} ({str(100 * target_cores / current_cores)}%)")
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
        self.current_cores: int = 7168 # Set to physical GPU's CUDA cores, e.g., 7168 for RTX 4070 Super

    def fit(self, ins: FitIns) -> FitRes:
        # Save the global model parameters to a file to be loaded by trainer.py

        if not os.path.exists(f"./bouquetfl/checkpoints/global_params_round_{ins.config['server_round']}.npz"):
            ndarrays_original = parameters_to_ndarrays(ins.parameters)
            np.savez(f"./bouquetfl/checkpoints/global_params_round_{ins.config['server_round']}.npz", *ndarrays_original)
        try:
            with open("./bouquetfl/hardwareconf/client_hardware.yaml", "r") as f:
                client_config = yaml.safe_load(f)
                gpu_name = client_config[f"client_{self.client_id}"]["gpu"]
                cpu_name = client_config[f"client_{self.client_id}"]["cpu"]
                ram_size = client_config[f"client_{self.client_id}"]["ram_gb"]
        except FileNotFoundError:
            gpu_name, cpu_name, ram_size = generate_hardware_sample()
        env = create_cuda_restricted_env(gpu_name, self.current_cores)

        start_mps()

        # We run trainer.py as a separate process with systemd-run using a set CUDA_MPS_ACTIVE_THREAD_PERCENTAGE.
        # Anything else (CPU throttling, RAM limiting, GPU memory and clock limiting) could be done without a separate process.

        # We take advantage of systemd-run to limit the RAM usage of the process.

        child = subprocess.Popen(
            [
                "systemd-run",
                "--user",
                "--scope",
                "-p",
                f"MemoryMax={ram_size}G",
                "uv",  # <--- Change this to "python3", "poetry", or corresponding if you don't have uv installed
                "run",
                "./bouquetfl/trainer.py",
                "--experiment",
                f"{'cifar100'}",
                "--client_id",
                f"{self.client_id}",
                "--gpu_name",
                f"{gpu_name}",
                "--cpu_name",
                f"{cpu_name}",
                "--round",
                f"{ins.config['server_round']}",
                "--num_rounds",
                f"{ins.config['num_rounds']}"
            ],
            env=env,
        )
        pid = child.pid
        print("Child PID:", pid)

        # Important: wait for the subprocess to finish before spawning the next one
        child.wait()
        stop_mps()

        # Get new stored model parameters and return to server
        local_save_path = f"./bouquetfl/checkpoints/params_updated_{self.client_id}.npz"
        try:
            ndarrays_new = np.load(local_save_path, allow_pickle=True)
            ndarrays_new = [ndarrays_new[key] for key in ndarrays_new]
            os.remove(local_save_path)
            # Build and return response
            status = Status(code=Code.OK, message="Success")
            logging.info(f"Client {self.client_id} successfully trained.")
            # Serialize ndarray's into a Parameters object
            parameters_updated = ndarrays_to_parameters(ndarrays_new)

        except FileNotFoundError:
            logging.info(
                f"Model file {local_save_path} not found. Training on client {self.client_id} seems to have failed."
            )
            status = Status(code=Code.FIT_NOT_IMPLEMENTED, message="Training failed.")
            parameters_updated = None

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
            metrics={"accuracy": accuracy},
        )


def client_fn(context: Context) -> Client:
    # Return Client instance
    return FlowerClient(client_id=context.node_config["partition-id"]).to_client()


# Flower ClientApp
app = ClientApp(
    client_fn,
)
