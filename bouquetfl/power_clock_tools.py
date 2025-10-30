import getpass
import os
import shutil
import subprocess

import pandas as pd
import psutil
import torch

import keyring

# This is required when running on Ubuntu-servers without a GUI, else just us PlaintextKeyring from keyring
from keyrings.alt.file import PlaintextKeyring
keyring.set_keyring(PlaintextKeyring())

service = "power_clock_tools_service"
username = "local_user"

# Before using the tools, run this once to store your sudo password securely.
#password = input("Enter password: ")

#keyring.set_password(service, username, password)
#print("Password saved securely.")

password = keyring.get_password(service, username)
if password is None:
    raise RuntimeError("No password found in keyring — run setup script first.")

#####################################
############# Auxiliary #############
#####################################


def run(cmd):
    command = " ".join(cmd)
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


def set_physical_gpu_limits(gpu_name: str):
    gpu_info = get_gpu_info(gpu_name)
    if not gpu_info:
        raise ValueError(f"GPU {gpu_name} not found in database.")
    current_gpu_info = get_current_gpu_info()

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

    set_gpu_memory_limit(gpu_info["memory"], 0)
    print(f"Set GPU memory limit to {gpu_info['memory']} GB")

    lock_gpu_clocks(0, int(gpu_info["clock speed"]), int(gpu_info["clock speed"]))
    print(f"Set GPU clock speed to {gpu_info['clock speed']} MHz")

    lock_gpu_memory_clocks(
        0, int(gpu_info["memory speed"]), int(gpu_info["memory speed"])
    )
    print(f"Set GPU memory speed to {gpu_info['memory speed']} MHz")


#####################################
############# CPU tools #############
#####################################


def set_cpu_limit(cpu_name: str):
    cpu_info = get_cpu_info(cpu_name)
    if not cpu_info:
        raise ValueError(f"CPU {cpu_name} not found in database.")
    current_cpu_info = get_current_cpu_info()

    if cpu_info["cores"] > int(current_cpu_info["cores"]):
        raise ValueError(
            f"CPU {cpu_name} has more cores ({cpu_info["cores"]}) than the current CPU ({current_cpu_info['cores']})."
        )
    if cpu_info["turbo clock"] > int(current_cpu_info["clock speed"]):
        raise ValueError(
            f"CPU {cpu_name} has a higher clock speed ({cpu_info['turbo clock']} MHz) than the current CPU ({current_cpu_info['clock speed']} MHz)."
        )

    cmd = [
        "sudo -S",
        "cpupower",
        "frequency-set",
        "-u",
        f"{cpu_info['turbo clock']}GHz",
    ]
    run(cmd)
    print(f"Set CPU clock speed to {cpu_info['turbo clock']} MHz")
    return cpu_info["cores"]


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


def reset_cpu_limit():
    # IMPORTANT: FIND CURRENT MIN MAX WITH >>>`cpupower frequency-info`
    # Output should be something like: hardware limits: 800 MHz - 3.60 GHz
    cmd = [
        "sudo -S",
        "cpupower",
        "frequency-set",
        "-g",
        "performance",
    ]
    run(cmd)


def reset_all_limits():
    reset_cpu_limit()
    reset_gpu_memory_limit(0)
    reset_gpu_clocks(0)
    reset_gpu_memory_clocks(0)
    print("Reset memory limit and clock speeds to default")
