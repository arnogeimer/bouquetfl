"""bouquetfl: A Flower / PyTorch app."""

import logging

from flwr.client import Client, ClientApp
from flwr.common import (Code, Context, EvaluateIns, EvaluateRes, FitIns,
                         FitRes, Status, parameters_to_ndarrays)

from bouquetfl.utils import power_clock_tools as pct
from bouquetfl.utils.filesystem import (load_client_hardware_config,
                                        load_new_client_parameters,
                                        save_ndarrays)

logger = logging.getLogger(__name__)
import os
import subprocess

import torch
import yaml

with open("./config/local_hardware_parameters.yaml", "r") as stats_file:
    hardware_stats = yaml.safe_load(stats_file)
    current_cores = hardware_stats.get("gpu_cores", None)


def create_cuda_restricted_env(gpu_name: str):
    gpu_info = pct.get_gpu_info(gpu_name)
    target_cores = gpu_info["cuda cores"]
    env = os.environ.copy()
    env["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = str(100 * target_cores / current_cores)
    print(
        f"Creating CUDA MPS env with {target_cores} cores out of {current_cores} ({str(100 * target_cores / current_cores)}%)"
    )
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

    def fit(self, ins: FitIns) -> FitRes:
        # Save the global model parameters to a file to be loaded by trainer.py
        pct.reset_all_limits()
        save_ndarrays(
            parameters_to_ndarrays(ins.parameters),
            f"checkpoints/global_params_round_{ins.config['server_round']}.npz",
        )
        gpu, _, ram = load_client_hardware_config(self.client_id)

        env = create_cuda_restricted_env(gpu)
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
                f"MemoryMax={ram}G",
                "uv",  # <--- Change this to "python3", "poetry", or corresponding if you don't have uv installed
                "run",
                "./bouquetfl/core/trainer.py",
                "--experiment",
                f"{'cifar100'}",
                "--client_id",
                f"{self.client_id}",
                "--round",
                f"{ins.config['server_round']}",
                "--num_rounds",
                f"{ins.config['num_rounds']}",
            ],
            env=env,
        )
        pid = child.pid

        # Important: wait for the subprocess to finish before spawning the next one
        child.wait()
        stop_mps()
        pct.reset_all_limits()

        # Get new stored model parameters and return to server

        status, parameters_updated = load_new_client_parameters(self.client_id)

        return FitRes(
            status=status,
            parameters=parameters_updated,
            num_examples=self.num_examples,
            metrics={"client_id": self.client_id},
        )

    def evaluate(self, ins: EvaluateIns):
        from task import cifar100 as flower_baseline

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
