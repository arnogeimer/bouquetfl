import os
import subprocess

import yaml
from flwr.common import Status, parameters_to_ndarrays
from flwr.common.typing import Parameters

from bouquetfl.utils import power_clock_tools as pct
from bouquetfl.utils.filesystem import (
    load_client_hardware_config,
    load_new_client_parameters,
    save_ndarrays,
)


def _create_cuda_restricted_env(gpu_name: str):
    gpu_info = pct.get_gpu_info(gpu_name)

    with open("./config/local_hardware.yaml", "r") as stats_file:
        hardware_stats = yaml.safe_load(stats_file)
    current_cores = hardware_stats.get("gpu_cores", None)
    target_cores = gpu_info["cuda cores"]
    env = os.environ.copy()
    env["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = str(100 * target_cores / current_cores)
    print(
        f"Creating CUDA MPS env with {target_cores} cores out of {current_cores} ({str(100 * target_cores / current_cores)}%)"
    )
    return env


def _start_mps():
    mps_proc = subprocess.Popen(["nvidia-cuda-mps-control", "-d"])
    mps_proc.wait()
    print("MPS server has started.")


def _stop_mps():
    mps_proc = subprocess.Popen(["echo", "quit", "|", "nvidia-cuda-mps-control"])
    mps_proc.wait()
    print("MPS server has exited.")


def run_training_process_in_env(client_id: int, ins) -> tuple[Status, Parameters]:

    save_ndarrays(
        parameters_to_ndarrays(ins.parameters),
        f"checkpoints/global_params_round_{ins.config['server_round']}.npz",
    )

    gpu, _, ram = load_client_hardware_config(client_id)

    env = _create_cuda_restricted_env(gpu)
    _start_mps()

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
            "uv",  # <--- Change this to "python3", "poetry", "pyenv" or corresponding if you don't have uv installed
            "run",
            "bouquetfl/core/trainer.py",
            "--experiment",
            f"{'cifar100'}",
            "--client_id",
            f"{client_id}",
            "--round",
            f"{ins.config['server_round']}",
            "--num_rounds",
            f"{ins.config['num_rounds']}",
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
        env=env,
    )

    # Wait for the subprocess to finish before spawning the next one
    child.wait()
    _stop_mps()
    pct.reset_all_limits()

    # Get new stored model parameters and return to server

    status, parameters_updated = load_new_client_parameters(client_id)
    return status, parameters_updated
