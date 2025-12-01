import numpy as np
import pandas as pd
from bouquetfl.power_clock_tools import get_available_gpu_cores, get_available_cpu_cores
# Steam Hardware Survey based sampling for GPUs and CPUs: hardware stats for Windows Computers (94.84% of total)
# Source: https://store.steampowered.com/hwsurvey/processormfg/   (October 2025)

def generate_gpu_sample() -> list[str]:
    gpu_cores = get_available_gpu_cores()
    gpu_df = pd.read_csv("./bouquetfl/hardwareconf/gpus.csv")
    probabilities = gpu_df["shss"].astype(float)
    probabilities = probabilities / np.sum(probabilities)
    sample_compatible = False
    while not sample_compatible:
        sampled_gpu = np.random.choice(
            gpu_df["gpu name"], p=probabilities, replace=True
        )
        print(sampled_gpu)
        gpu_info = gpu_df[gpu_df["gpu name"] == sampled_gpu].iloc[0]
        if gpu_info["CUDA cores"] <= gpu_cores:
            sample_compatible = True
    return sampled_gpu

def generate_cpu_sample() -> list[str]:
    cpu_cores = get_available_cpu_cores()
    cpu_df = pd.read_csv("./bouquetfl/hardwareconf/cpus.csv")
    probabilities = cpu_df["shss"].astype(float)
    probabilities = probabilities / np.sum(probabilities)
    sample_compatible = False
    while not sample_compatible:
        sampled_cpu = np.random.choice(
            cpu_df["cpu name"], p=probabilities, replace=True
        )
        cpu_info = cpu_df[cpu_df["cpu name"] == sampled_cpu].iloc[0]
        if len(cpu_info["cores"].split(" ")) > 1:
            num_cores = cpu_info["cores"].split(" ")[0]
        else:
            num_cores = cpu_info["cores"]
        if int(num_cores) <= cpu_cores:
            sample_compatible = True
    return sampled_cpu

def generate_ram_sample() -> int:
    ram_options = [4, 8, 12, 16, 24, 32, 48, 64]
    probabilities = np.array([.0162, .0838, .0261, .4149, .0185, .3593, .0109, .0435])
    probabilities = probabilities / np.sum(probabilities)
    sampled_ram = np.random.choice(
        ram_options, p=probabilities, replace=True
    ).tolist()
    return sampled_ram

def generate_hardware_sample() -> tuple[str, str, int]:
    gpu = generate_gpu_sample()
    cpu = generate_cpu_sample()
    ram = generate_ram_sample()
    print(gpu, cpu, ram)
    return (gpu, cpu, ram)

for i in range(10):
    generate_hardware_sample()