import os
import subprocess

import yaml
from flwr.common import FitIns, Status, parameters_to_ndarrays
from flwr.common.typing import Parameters

from bouquetfl.utils import power_clock_tools as pct
from bouquetfl.utils.filesystem import (load_client_hardware_config,
                                        load_new_client_state_dict,
                                        save_ndarrays)

import json
import subprocess
import tempfile
from torch.utils.tensorboard import SummaryWriter
from flwr.app import Context, Message
writer = SummaryWriter("logs/bouquetrun")

def _create_cuda_restricted_env(gpu_name: str):
    gpu_info = pct.get_gpu_info(gpu_name)

    with open("./config/local_hardware.yaml", "r") as stats_file:
        hardware_stats = yaml.safe_load(stats_file)
    current_cores = hardware_stats.get("gpu_cores", None)
    target_cores = gpu_info["cuda cores"]
    if current_cores < target_cores:
        print("Emulating a GPU which has more cores than local is impossible.")
        target_cores = current_cores
    os.environ["CUDA_MPS_LOG_DIRECTORY"] = "/tmp/nvidia-mps"
    env = os.environ.copy()
    env["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = str(100 * target_cores / current_cores)
    print(
        f"Creating CUDA MPS env with {target_cores} cores out of {current_cores} ({round(100 * target_cores / current_cores, 2)}%)"
    )
    return env


def _start_mps():
    mps_proc = subprocess.Popen(["nvidia-cuda-mps-control", "-d"])
    mps_proc.wait()
    print("MPS server has started.")


def _stop_mps(env):
    # Send "quit" to the control daemon via stdin (NOT via "|")
    p = subprocess.run(
        ["nvidia-cuda-mps-control"],
        input="quit\n",
        text=True,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return p.returncode, p.stdout, p.stderr


def run_training_process_in_env(msg: Message, context: Context) -> tuple[Status, Parameters]:

    gpu, cpu, ram = load_client_hardware_config(context.node_config['partition-id'])

    client_i = {
    "gpu_name": gpu,
    "cpu_name": cpu,
    "ram_gb": ram,
}
    step = 0

    writer.add_text(
        f"clients/{context.node_config['partition-id']}/info_json",
        "```json\n" + json.dumps(client_i, indent=2) + "\n```",
        step,
    )

    env = _create_cuda_restricted_env(gpu)
    _start_mps()

    # We run trainer.py as a separate process with systemd-run using a set CUDA_MPS_ACTIVE_THREAD_PERCENTAGE.
    # Anything else (CPU throttling, RAM limiting, GPU memory and clock limiting) could be done without a separate process.

    # We take advantage of systemd-run to limit the RAM usage of the process.

    cfg = dict(context.run_config)
    cfg.update(msg.content["config"])

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(cfg, f)
        cfg_path = f.name

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
            "--client_id",
            f"{context.node_config['partition-id']}",
            "--config", 
            f"{cfg_path}",
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
        env=env,
    )

    # Wait for the subprocess to finish before spawning the next one
    child.wait()
    #_stop_mps(env)
    pct.reset_all_limits()

    # Get new stored model parameters and return to server

    status, state_dict_updated = load_new_client_state_dict(context.node_config['partition-id'])
    return status, state_dict_updated