import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

results = pd.read_pickle("./bouquetfl/checkpoints/load_and_training_times.pkl")

num_rounds = int(results.shape[1] / 2 - 1)


results["total_load_time"] = np.nan
results["total_train_time"] = np.nan

for i in range(results.shape[0]):
    train_col_list = [f"train_time_{j}" for j in range(1, num_rounds + 1)]
    load_col_list = [f"load_time_{j}" for j in range(1, num_rounds + 1)]
    results["total_train_time"] = results[train_col_list].sum(axis=1)
    results["total_load_time"] = results[load_col_list].sum(axis=1)
print(results)

def gpu_times(df):
    df = df.sort_values(by=["total_train_time"], ignore_index=True, ascending=False)
    _, ax = plt.subplots()
    for i in range(df.shape[0]):
        ax.barh(i, df["total_train_time"][i] / num_rounds, 0.4, left=0.001, color="royalblue")
    ax.set_yticks(range(df.shape[0]), df["gpu"])
    plt.title("Average training times per GPU")
    plt.tight_layout()
    plt.savefig("./bouquetfl/plots/gpu.png")


def cpu_times(df):
    df = df.sort_values(by=["total_load_time"], ignore_index=True, ascending=False)
    _, ax = plt.subplots()
    for i in range(df.shape[0]):
        ax.barh(i - 0.2, df["total_load_time"][i] / num_rounds, 0.4, left=0.001, color="royalblue")
    ax.set_yticks(range(df.shape[0]), df["cpu"])
    plt.title("Average data loading times per CPU")
    plt.tight_layout()
    plt.savefig("./bouquetfl/plots/cpu.png")


def plot_federation_timeline(df):

    df = df.sort_values(by=["total_train_time"], ignore_index=True, ascending=False)
    _, ax = plt.subplots()
    starttime = 0.1
    for round in range(1, num_rounds + 1):
        max_time = 0
        for client in range(df.shape[0]):
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
        if round < num_rounds - 1:
            plt.vlines(
                starttime + max_time + 0.05,
                ymin=-0.2,
                ymax=df.shape[0] - 0.8,
                color="black",
                linewidth=0.1,
            )
        starttime += max_time + 0.1
    y_ticks = [
        f"{df['gpu'][client]} \n {df['cpu'][client]}" for client in range(df.shape[0])
    ]
    ax.set_yticks(range(df.shape[0]), y_ticks, fontsize=5)
    plt.xlabel("time (s)", loc="right")
    plt.title("Federation timetable")

    custom_lines = [
        Line2D([0], [0], color="red", lw=4),
        Line2D([0], [0], color="green", lw=4),
    ]

    ax.legend(custom_lines, ["load_time", "train_time"])

    plt.tight_layout()
    plt.savefig("./bouquetfl/plots/timetable.png")


gpu_times(results)
cpu_times(results)
plot_federation_timeline(results)
