import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

d = {
    "id": np.linspace(0, 19, 20),
    "gpu": [
        "NVIDIA GeForce RTX 4090",
        "NVIDIA GeForce RTX 4080",
        "NVIDIA GeForce RTX 4070 Ti",
        "NVIDIA GeForce RTX 3090 Ti",
        "NVIDIA GeForce RTX 3080",
        "NVIDIA GeForce RTX 3070",
        "NVIDIA GeForce RTX 3060 Ti",
        "NVIDIA GeForce GTX 1660 Super",
        "NVIDIA GeForce GTX 1080 Ti",
        "NVIDIA Tesla V100",
        "NVIDIA A100",
        "NVIDIA H100",
        "AMD Radeon RX 7900 XTX",
        "AMD Radeon RX 7800 XT",
        "AMD Radeon RX 7700 XT",
        "AMD Radeon RX 6700 XT",
        "AMD Radeon RX 6600 XT",
        "Intel Arc A770",
        "Intel Arc A750",
        "Apple M3 GPU",
    ],
    "cpu": [
        "Intel Core i9-14900K",
        "Intel Core i7-14700K",
        "Intel Core i5-14600K",
        "Intel Core i9-13900KS",
        "Intel Core i7-13700K",
        "AMD Ryzen 9 7950X3D",
        "AMD Ryzen 9 7900X",
        "AMD Ryzen 7 7800X3D",
        "AMD Ryzen 7 7700",
        "AMD Ryzen 5 7600X",
        "AMD Ryzen 5 5600G",
        "Intel Core i9-12900K",
        "Intel Core i7-12700K",
        "Intel Core i5-12600K",
        "AMD Threadripper PRO 7995WX",
        "AMD EPYC 9654",
        "Intel Xeon W9-3495X",
        "Apple M3 Max",
        "Apple M2 Ultra",
        "Qualcomm Snapdragon X Elite",
    ],
    "load_time_1": np.random.random(20),
    "load_time_2": np.random.random(20),
    "load_time_3": np.random.random(20),
    "load_time_4": np.random.random(20),
    "train_time_1": 13 * np.random.random(20),
    "train_time_2": 13 * np.random.random(20),
    "train_time_3": 13 * np.random.random(20),
    "train_time_4": 13 * np.random.random(20),
}

results = pd.DataFrame(d)
results["total_train_time"] = [
    results["train_time_1"][i]
    + results["train_time_2"][i]
    + results["train_time_2"][i]
    + results["train_time_2"][i]
    for i in range(len(results["id"]))
]
results["total_load_time"] = [
    results["load_time_1"][i]
    + results["load_time_2"][i]
    + results["load_time_3"][i]
    + results["load_time_4"][i]
    for i in range(len(results["id"]))
]
print(results)


def gpu_times(df):
    df = df.sort_values(by=["train_time_1"], ignore_index=True, ascending=False)
    _, ax = plt.subplots()
    for i in range(len(df["train_time_1"])):
        ax.barh(i, df["train_time_1"][i], 0.4, left=0.001, color="royalblue")
    ax.set_yticks(range(len(df["gpu"])), df["gpu"])
    plt.title("Average training times per GPU")
    plt.tight_layout()
    plt.savefig("./gpu.png")


def cpu_times(df):
    df = df.sort_values(by=["load_time_1"], ignore_index=True, ascending=False)
    _, ax = plt.subplots()
    for i in range(len(df["load_time_1"])):
        ax.barh(i - 0.2, df["load_time_1"][i], 0.4, left=0.001, color="royalblue")
    ax.set_yticks(range(len(df["cpu"])), df["cpu"])
    plt.title("Average data loading times per CPU")
    plt.tight_layout()
    plt.savefig("./cpu.png")


def plot_federation_timeline(df):

    df = df.sort_values(by=["total_train_time"], ignore_index=True, ascending=False)
    _, ax = plt.subplots()
    starttime = 0.1
    for round in range(1, 5):
        max_time = 0
        for client in range(len(df["id"])):
            ax.barh(
                client,
                df[f"load_time_{round}"][client],
                0.4,
                left=starttime,
                color="red",
            )
            ax.barh(
                client,
                df[f"train_time_{round}"][client],
                0.4,
                left=starttime + df[f"load_time_{round}"][client],
                color="green",
            )
            if (
                df[f"load_time_{round}"][client] + df[f"train_time_{round}"][client]
                > max_time
            ):
                max_time = (
                    df[f"load_time_{round}"][client] + df[f"train_time_{round}"][client]
                )
        if round < 4:
            plt.vlines(
                starttime + max_time + 0.05,
                ymin=-0.2,
                ymax=len(df["id"]) - 0.8,
                color="black",
                linewidth=0.1,
            )
        starttime += max_time + 0.1
    y_ticks = [
        f"{df['gpu'][client]} \n {df['cpu'][client]}" for client in range(len(df["id"]))
    ]
    ax.set_yticks(range(len(df["id"])), y_ticks, fontsize=5)
    plt.xlabel("time (s)", loc="right")
    plt.title("Federation timetable")

    custom_lines = [
        Line2D([0], [0], color="red", lw=4),
        Line2D([0], [0], color="green", lw=4),
    ]

    ax.legend(custom_lines, ["load_time", "train_time"])

    plt.tight_layout()
    plt.savefig("./timetable.png")


gpu_times(results)
cpu_times(results)
plot_federation_timeline(results)
