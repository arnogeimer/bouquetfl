import getpass
import os
import shutil
import subprocess

import pandas as pd
import psutil
import torch

password = getpass.getpass(
    prompt="Enter your sudo password (needed for some GPU/CPU operations): "
)


def run(cmd):
    command = " ".join(cmd)
    # print(">> " + command)
    os.system("echo %s | %s" % (password, command))


def require_sudo():
    if shutil.which("sudo") is None:
        print("sudo not found. On some systems you must run as root.")
    return []


#####################################
############# GPU tools #############
#####################################

"""
def get_memory_usage(gpu_index: int):
    cmd = ["nvidia-smi", "-i", str(gpu_index), "--query-gpu=memory.used,memory.total"]
    run(cmd)
"""


def set_gpu_memory_limit(value: int, gpu_index: int):
    "Sets total available memory of the process to <value> GB"
    total_memory = torch.cuda.get_device_properties(0).total_memory
    memory_fraction = min(1, float(value * 1024**3) / total_memory)
    torch.cuda.memory.set_per_process_memory_fraction(memory_fraction, gpu_index)


def reset_gpu_memory_limit(gpu_index: int):
    torch.cuda.memory.set_per_process_memory_fraction(1.0, gpu_index)


"""
def set_power_limit_watts(gpu_index: int, watts: int):
    # Requires sudo and within allowed range (see `nvidia-smi -q -d POWER`)
    cmd = ["sudo -S", "nvidia-smi", "-i", str(gpu_index), "-pl", str(watts)]
    run(cmd)
"""

"""
def show_power_limits(gpu_index: int):
    cmd = [
        "nvidia-smi",
        "-i",
        str(gpu_index),
        "--query-gpu=power.min_limit,power.max_limit,power.limit,power.draw",
        "--format=csv",
    ]
    run(cmd)
"""


def lock_gpu_clocks(gpu_index: int, min_mhz: int, max_mhz: int):
    # Requires sudo; only supported on Volta+.
    cmd = [
        "sudo -S",
        "nvidia-smi",
        "-i",
        str(gpu_index),
        f"--lock-gpu-clocks={min_mhz},{max_mhz}",
    ]
    run(cmd)


def reset_gpu_clocks(gpu_index: int):
    # Requires sudo; only supported on Volta+.
    cmd = ["sudo -S", "nvidia-smi", "-i", str(gpu_index), "--reset-gpu-clocks"]
    run(cmd)


def lock_gpu_memory_clocks(gpu_index: int, min_mhz: int, max_mhz: int):
    # Requires sudo; only supported on Volta+.
    cmd = [
        "sudo -S",
        "nvidia-smi",
        "-i",
        str(gpu_index),
        f"--lock-memory-clocks={min_mhz},{max_mhz}",
    ]
    run(cmd)


def reset_gpu_memory_clocks(gpu_index: int):
    # Requires sudo; only supported on Volta+.
    cmd = ["sudo -S", "nvidia-smi", "-i", str(gpu_index), "--reset-memory-clocks"]
    run(cmd)


def get_current_gpu_info():
    query = "name,memory.total,clocks.max.graphics,clocks.max.memory"
    result = subprocess.run(
        ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
        stdout=subprocess.PIPE,
        text=True,
    )
    name, mem_total, max_graphics, max_mem = result.stdout.strip().split(", ")
    return {
        "name": name,
        "memory": int(int(mem_total) / 1024),
        "clock speed": int(max_graphics),
        "memory speed": int(max_mem),
    }


def enable_persistence():
    # Optional: keeps driver state loaded; helpful for NVML operations
    cmd = ["sudo -S", "nvidia-smi", "-pm", "1"]
    out = run(cmd)
    print(out.stdout or out.stderr)


def nvml_lock_app_clocks(gpu_index: int, mem_mhz: int, graphics_mhz: int):
    """Try to lock application clocks via NVML using pynvml (needs root on many systems)."""
    ### FLAG: DOES REQUIRE SUDO EXECUTION ###
    try:
        import pynvml
    except ImportError:
        print("pynvml not installed. `pip install nvidia-ml-py3`")
        return
    pynvml.nvmlInit()
    try:
        h = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
        # Prefer GPU locked clocks if available (Volta+). Fallback to applications clocks.
        try:
            fn = getattr(pynvml, "nvmlDeviceSetGpuLockedClocks", None)
            if fn is not None:
                fn(h, graphics_mhz, graphics_mhz)
                print(
                    f"Locked GPU clocks at {graphics_mhz} MHz via NVML (nvmlDeviceSetGpuLockedClocks)."
                )
            else:
                pynvml.nvmlDeviceSetApplicationsClocks(h, mem_mhz, graphics_mhz)
                print(
                    f"Set application clocks mem={mem_mhz} MHz, graphics={graphics_mhz} MHz."
                )
        finally:
            pynvml.nvmlShutdown()
    except Exception as e:
        print("NVML clock lock failed:", e)


def get_gpu_info(gpu_name: str):
    gpu_info = None
    with open("./bouquetfl/hardwareconf/gpus.csv") as file:
        gpus = pd.read_csv(file, header=None).to_numpy()
    for gpu in gpus:
        if gpu[0] == gpu_name:
            gpu_info = {
                "name": gpu[0],
                "memory": float(gpu[1]),
                "memory type": gpu[2],
                "memory bandwidth": float(gpu[3]),
                "clock speed": float(gpu[4]),
                "memory speed": float(gpu[5]),
                "cuda cores": int(gpu[6]),
            }

    if not gpu_info:
        raise ValueError(f"GPU {gpu_name} not found in database.")

    return gpu_info


#####################################
############# RAM tools #############
#####################################

import resource


def limit_ram(maxsize: float):
    "Sets memory limit of the process to <maxsize> GB"
    _, hard = resource.getrlimit(resource.RLIMIT_AS)
    resource.setrlimit(resource.RLIMIT_AS, (maxsize * (1024**3), hard))


def reset_ram_limit():
    resource.setrlimit(
        resource.RLIMIT_AS, (resource.RLIM_INFINITY, resource.RLIM_INFINITY)
    )


#####################################
############# CPU tools #############
#####################################


def set_cpu_limit(value: int):
    "Sets clock speed of the process to <value> MHz"
    cmd = [
        "sudo -S",
        "cpupower",
        "frequency-set",
        "-g",
        "performance",
        "-u",
        f"{value}GHz",
    ]
    run(cmd)


def reset_cpu_limit():
    # IMPORTANT: FIND CURRENT MIN MAX WITH >>>`cpupower frequency-info`
    # Output should be something like: hardware limits: 800 MHz - 3.60 GHz
    MIN = 0.8
    MAX = 3.6
    cmd = [
        "sudo -S",
        "cpupower",
        "frequency-set",
        "-d",
        f"{MIN}GHz",
        "-u",
        f"{MAX}GHz",
    ]
    run(cmd)


def get_cpu_info(cpu_name: str):
    cpu_info = None
    with open("./bouquetfl/hardwareconf/cpus.csv") as file:
        cpus = pd.read_csv(file, header=None).to_numpy()
    for cpu in cpus:
        if cpu[0] == cpu_name:
            if len(cpu[1].split(" ")) > 1:
                num_cores = cpu[1].split(" ")[0]
            else:
                num_cores = cpu[1]
            base_clock = cpu[2].split(" ")[0]
            turbo_clock = base_clock
            if len(cpu[2].split(" ")) > 2:
                turbo_clock = cpu[2].split(" ")[2]
            cpu_info = {
                "name": cpu[0],
                "cores": int(num_cores),
                "base clock": 1000 * float(base_clock),  # GHz to MHz
                "turbo clock": 1000 * float(turbo_clock),  # GHz to MHz
            }

    if not cpu_info:
        raise ValueError(f"CPU {cpu_name} not found in database.")

    return cpu_info


def get_current_cpu_info():
    cpu_info = {}
    cpu_info["cores"] = psutil.cpu_count(logical=False)
    cpu_info["clock speed"] = psutil.cpu_freq().max
    return cpu_info
