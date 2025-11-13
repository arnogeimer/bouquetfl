import numpy as np
import pandas as pd

# Steam Hardware Survey based sampling for GPUs and CPUs: hardware stats for Windows Computers (94.84% of total)

def generate_gpu_samples(n_samples: int) -> list[str]:

    gpu_df = pd.read_csv("./bouquetfl/hardwareconf/gpus.csv")
    probabilities = gpu_df["shss"].astype(float)
    probabilities = probabilities / np.sum(probabilities)
    sampled_gpus = np.random.choice(
        gpu_df["gpu name"], size=n_samples, p=probabilities, replace=True
    )
    return sampled_gpus

def generate_cpu_samples(n_samples: int) -> list[str]:

    cpu_df = pd.read_csv("./bouquetfl/hardwareconf/cpus.csv")
    probabilities = cpu_df["shss"].astype(float)
    probabilities = probabilities / np.sum(probabilities)
    sampled_cpus = np.random.choice(
        cpu_df["cpu name"], size=n_samples, p=probabilities, replace=True
    )
    return sampled_cpus

def generate_ram_samples(n_samples: int) -> list[int]:
    ram_options = [4, 8, 12, 16, 24, 32, 48, 64]
    probabilities = np.array([.0162, .0838, .0261, .4149, .0185, .3593, .0109, .0435])
    probabilities = probabilities / np.sum(probabilities)
    sampled_rams = np.random.choice(
        ram_options, size=n_samples, p=probabilities, replace=True
    ).tolist()
    return sampled_rams

def generate_hardware_samples(n_samples: int) -> list[tuple[str, str]]:
    gpu_samples = generate_gpu_samples(n_samples)
    cpu_samples = generate_cpu_samples(n_samples)
    ram_samples = generate_ram_samples(n_samples)
    sample_configs = []
    for i, (gpu, cpu, ram) in enumerate(zip(gpu_samples, cpu_samples, ram_samples)):
        sample_configs.append({f"client_{i}": {"gpu": gpu, "cpu": cpu, "ram_gb": ram}})
    return sample_configs
