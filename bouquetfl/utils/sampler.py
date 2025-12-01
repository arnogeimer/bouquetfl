import numpy as np
import pandas as pd
import yaml

from bouquetfl.utils.localinfo import get_all_local_info

# Steam Hardware Survey based sampling for GPUs and CPUs: hardware stats for Windows Computers (94.84% of total)
# Source: https://store.steampowered.com/hwsurvey/processormfg/   (October 2025)
# To generate samples, we first check the current hardware capabilities of the physical machine
# Then, we sample hardware profiles from the database until we find a compatible one
# This ensures that the sampled hardware can be realistically simulated on the physical machine


try:
    with open("config/local_hardware_parameters.yaml", "r") as stats_file:
        hardware_stats = yaml.safe_load(stats_file)
    gpu_mem, gpu_clock, gpu_mem_speed, gpu_cores = (
        hardware_stats.get("gpu_memory", None),
        hardware_stats.get("gpu_clock_speed", None),
        hardware_stats.get("gpu_memory_speed", None),
        hardware_stats.get("gpu_cores", None),
    )
    cpu_cores, cpu_clock = hardware_stats.get("cpu_cores", None), hardware_stats.get(
        "cpu_clock_speed", None
    )
    ram = hardware_stats.get("ram_gb", None)
except FileNotFoundError:
    local_info = get_all_local_info()
    with open("config/local_hardware_parameters.yaml", "w") as stats_file:
        yaml.dump(local_info, stats_file)
    gpu_mem, gpu_clock, gpu_mem_speed, gpu_cores = (
        local_info.get("gpu_memory", None),
        local_info.get("gpu_clock_speed", None),
        local_info.get("gpu_memory_speed", None),
        local_info.get("gpu_cores", None),
    )
    cpu_cores, cpu_clock = local_info.get("cpu_cores", None), local_info.get(
        "cpu_clock_speed", None
    )
    ram = local_info.get("ram_gb", None)
print(
    f"Local hardware capabilities: GPU cores={gpu_cores}, GPU clock={gpu_clock} MHz, GPU memory={gpu_mem} GB, GPU memory speed={gpu_mem_speed} MHz, CPU cores={cpu_cores}, CPU clock={cpu_clock} GHz, RAM={ram} GB"
)


def _generate_gpu_sample() -> list[str]:
    gpu_df = pd.read_csv("hardwareconf/gpus.csv")
    probabilities = gpu_df["shss"].astype(float)
    probabilities = probabilities / np.sum(probabilities)
    sample_compatible = False
    tries = 0
    while not sample_compatible:
        tries += 1
        sampled_gpu = np.random.choice(
            gpu_df["gpu name"], p=probabilities, replace=True
        )
        gpu_info = gpu_df[gpu_df["gpu name"] == sampled_gpu].iloc[0]
        if (
            (gpu_info["CUDA cores"] <= gpu_cores)
            and (gpu_info["Clock speed"] <= gpu_clock)
            and (gpu_info["Memory (GB)"] <= gpu_mem)
            and (gpu_info["Memory Speed"] <= gpu_mem_speed)
        ):
            sample_compatible = True
        if tries > 50:
            print("Could not find compatible GPU after 50 tries, using fallback GPU.")
            sampled_gpu = "GeForce GTX 1050"  # Fallback GPU
            sample_compatible = True
    return sampled_gpu


def _generate_cpu_sample() -> list[str]:
    cpu_df = pd.read_csv("hardwareconf/cpus.csv")
    probabilities = cpu_df["shss"].astype(float)
    probabilities = probabilities / np.sum(probabilities)
    sample_compatible = False
    tries = 0
    while not sample_compatible:
        tries += 1
        sampled_cpu = np.random.choice(
            cpu_df["cpu name"], p=probabilities, replace=True
        )
        cpu_info = cpu_df[cpu_df["cpu name"] == sampled_cpu].iloc[0]
        if len(cpu_info["cores"].split(" ")) > 1:
            num_cores = cpu_info["cores"].split(" ")[0]
        else:
            num_cores = cpu_info["cores"]
        if len(cpu_info["core clock"].split(" ")) > 1:
            clock_speed = 1000 * float(cpu_info["core clock"].split(" ")[0])
        else:
            clock_speed = 1000 * float(cpu_info["core clock"])
        if int(num_cores) <= cpu_cores and float(clock_speed) <= cpu_clock:
            sample_compatible = True
        if tries > 50:
            print("Could not find compatible CPU after 50 tries, using fallback CPU.")
            sampled_cpu = "Ryzen 3 1200"  # Fallback CPU
            sample_compatible = True
    return sampled_cpu


def _generate_ram_sample() -> int:
    ram_options = [4, 8, 12, 16, 24, 32, 48, 64]
    probabilities = np.array(
        [0.0162, 0.0838, 0.0261, 0.4149, 0.0185, 0.3593, 0.0109, 0.0435]
    )
    probabilities = probabilities / np.sum(probabilities)
    sample_compatible = False
    while not sample_compatible:
        sampled_ram = np.random.choice(ram_options, p=probabilities, replace=True)
        if sampled_ram <= ram:
            sample_compatible = True
    return sampled_ram.tolist()


def generate_hardware_config(num_clients: int) -> None:
    client_hardware = {}
    for client_id in range(num_clients):
        gpu, cpu, ram = (
            _generate_gpu_sample(),
            _generate_cpu_sample(),
            _generate_ram_sample(),
        )
        client_hardware[f"client_{client_id}"] = {
            "gpu": gpu,
            "cpu": cpu,
            "ram_gb": ram,
        }
    with open("./config/federation_client_hardware.yaml", "w") as hardware_file:
        yaml.dump(client_hardware, hardware_file)
