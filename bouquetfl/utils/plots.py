import altair as alt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from filesystem import delete_unused_files
from matplotlib.lines import Line2D

# delete_unused_files()


def plot_gpu_average_training_times():

    results = pd.read_pickle("checkpoints/load_and_training_times.pkl")
    results["total_load_time"] = np.nan
    results["total_train_time"] = np.nan

    results = results[:14]

    num_rounds = int(results.shape[1] / 2 - 2)

    time_errors = []
    for client_id in range(14):
        values = []
        for round in range(num_rounds):
            values.append(results[f"train_time_{round+1}"][client_id])

        time_errors.append(np.std(values))

    for i in range(num_rounds):
        results[f"{i+1}load_time"] = results[f"load_time_{i+1}"].astype(float)
        results[f"{i+1}train_time"] = results[f"train_time_{i+1}"].astype(float)
        del results[f"load_time_{i+1}"]
        del results[f"train_time_{i+1}"]

    for i in range(results.shape[0]):
        train_col_list = [f"{j}train_time" for j in range(1, num_rounds + 1)]
        load_col_list = [f"{j}load_time" for j in range(1, num_rounds + 1)]
        results["total_train_time"] = results[train_col_list].sum(axis=1)
        results["total_load_time"] = results[load_col_list].sum(axis=1)
    print(results["cpu"])
    print(results["gpu"])
    print("HERE", results["total_train_time"])
    # Identify duplicate GPUs and CPUs and average their times
    results["total_train_time"] = results.groupby("gpu")["total_train_time"].transform(
        "mean"
    )
    """
    results["total_load_time"] = results.groupby("cpu")["total_load_time"].transform(
        "mean"
    )"""

    results = results.drop_duplicates(subset=["gpu"]).reset_index(drop=True)
    results = results[0:14]
    # results = results.drop_duplicates(subset=["cpu"]).reset_index(drop=True)
    print(np.array(results["total_train_time"]))
    # results = zip(results["gpu"], results["total_train_time"] / max(results["total_train_time"]))

    _, ax = plt.subplots(figsize=(4, 4), constrained_layout=True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", labelsize=10)
    # Example data
    y_pos = np.arange(len(results["gpu"]))

    norm_times = results["total_train_time"] / max(results["total_train_time"])
    print(np.array(norm_times))
    bmarks = [
        np.array(
            [
                10.7,
                23.7,
                33.5,
                43,
                18.1,
                28.6,
                31.8,
                37.2,
                42.7,
                51.4,
                31,
                42,
                61.7,
                78.9,
            ]
        ),
        np.array(
            [
                5040,
                9821,
                13507,
                15614,
                7867,
                11631,
                12710,
                14113,
                16049,
                18653,
                10750,
                15223,
                22149,
                25046,
            ]
        ),
    ]
    norm_times = []
    for benchmark in bmarks:
        normalized_times = benchmark / max(benchmark)
        normalized_times = np.abs(
            normalized_times - max(normalized_times)
        )  # Invert the times for better visualization
        normalized_times += 1 - max(normalized_times)
        norm_times.append(normalized_times)

    norm_times = np.array(norm_times)
    print((norm_times[0] + norm_times[1]) / 2)

    ax.barh(
        y_pos,
        np.mean(norm_times, axis=0),
        xerr=np.std(norm_times, axis=0),
        capsize=4,
        edgecolor="black",
        linewidth=0.6,
        align="center",
        color="#4C72B0",
    )
    labels = []
    print(results["gpu"])
    for label in results["gpu"]:
        label = label.split()
        if len(label) == 3:
            labels.append(f"{label[1]} {label[2]}")
        elif len(label) > 3:
            labels.append(f"{label[1]} {label[2]} {label[3]}")
    ax.set_yticks(y_pos, labels=labels)
    ax.invert_yaxis()  # labels read top-to-bottom

    plt.show()
    plt.savefig("plots/normalized_benchmark_time_by_gpu.pdf")
#plot_gpu_average_training_times()

"""
    df_long = results.melt(
        id_vars=["gpu", "cpu", "total_load_time", "total_train_time", "1load_time"],
        var_name="segment",
        value_name="value",
    )

    df_long["segment"] = df_long["segment"].apply(
        lambda x: "load_time" if "load_time" in x else "train_time"
    )


    gpu_chart = (
        alt.Chart(df_long)
        .mark_bar(
            cornerRadiusBottomLeft=2,
            cornerRadiusTopLeft=2,
            cornerRadiusBottomRight=2,
            cornerRadiusTopRight=2,
            strokeWidth=0,
        )
        .encode(
            x=alt.X("total_train_time", title="training time (s)"),
            y=alt.Y("gpu").sort("-x"),
        )
    )

    gpu_chart.save("plots/timetable_gpu.pdf")
    gpu_chart.save("plots/timetable_gpu.png")

    cpu_chart = (
        alt.Chart(df_long)
        .mark_bar(
            cornerRadiusBottomLeft=2,
            cornerRadiusTopLeft=2,
            cornerRadiusBottomRight=2,
            cornerRadiusTopRight=2,
        )
        .encode(
            x=alt.X("value:Q", title="data load time (s)"),
            y=alt.Y("cpu").sort("-x"),
        )
    )

    cpu_chart.save("plots/timetable_cpu.pdf")
    cpu_chart.save("plots/timetable_cpu.png")

"""
# plot_gpu_average_training_times()
from scipy.stats import kendalltau, spearmanr


def generate_scatter_plot():
    gpus = [
        "GeForce GTX 1050",
        "GeForce GTX 1060",
        "GeForce GTX 1070",
        "GeForce GTX 1080",
        "GeForce GTX 1650",
        "GeForce GTX 1660",
        "GeForce GTX 1660 Ti",
        "GeForce RTX 2060",
        "GeForce RTX 2070",
        "GeForce RTX 2080",
        "GeForce RTX 3050",
        "GeForce RTX 3060",
        "GeForce RTX 3070",
        "GeForce RTX 3080",
    ]

    gpu_times, benchmark_times = (
        np.array([96.25812109, 39.12650584, 31.15540277, 24.54303205, 59.90921479, 37.90189175,
        33.42215541, 31.3451803,  27.05915711, 23.49124462, 28.8020803,  20.23592472,
        16.95063561, 15.94066052]),
        np.array([
            5040,
            9821,
            13507,
            15614,
            7867,
            11631,
            12710,
            14113,
            16049,
            18653,
            10750,
            15223,
            22149,
            25046,
        ])
    )

    gpu_times = gpu_times / np.mean(gpu_times)
    benchmark_times = np.abs(benchmark_times - max(benchmark_times) - min(benchmark_times))
    benchmark_times = benchmark_times / np.mean(benchmark_times)
    fig, ax = plt.subplots(figsize=(4, 4), constrained_layout=True)

    from matplotlib.ticker import AutoMinorLocator

    ax.set_axisbelow(True)
    ax.xaxis.set_minor_locator(AutoMinorLocator(4))
    ax.yaxis.set_minor_locator(AutoMinorLocator(4))
    # And a corresponding grid
    ax.grid(which="major", color="#CCCCCC", linestyle="--")
    #ax.grid(which="minor", color="#CCCCCC", linestyle=":")
    ax.plot([0, 1], [0, 1], color="black", linestyle="-", linewidth=0.8, zorder=1, alpha = .6)
    print(gpus)
    print("spearman", spearmanr(gpu_times, benchmark_times))
    print("kendall", kendalltau(gpu_times, benchmark_times))
    gpu_times = gpu_times[1:] / 2
    benchmark_times = benchmark_times[1:] / 2
    ax.scatter(
        gpu_times,
        benchmark_times,
        color="#76b900",
        edgecolor="black",
        linewidth=0.6,
        zorder=2,
    )
    for i, txt in enumerate(gpus[1:]):
        if i in [0, 2, 3, 6, 8, 9, 12]:
            print(txt)
            if i == 0:
                offset = (-10, 5)
            elif i == 2:
                offset = (-10, 5)
            elif i == 3:
                offset = (5, -2.5)
            elif i == 6:
                offset = (5, -5)
            elif i == 8:
                offset = (-25, -3)
            elif i == 9:
                offset = (-15.5, 4)
            elif i == 12:
                offset = (-5.5, 4)
            ax.annotate(
                txt.split()[-1],
                (gpu_times[i], benchmark_times[i]),
                xytext=offset,
                textcoords="offset points",
                fontsize=8,
            )
    # plt.title("Spearman correlation = 0.99", fontsize=12)
    plt.xlabel("Bouquet", fontsize=10)
    plt.ylabel("Benchmark", fontsize=10)

    plt.show()
    plt.savefig("plots/scatters.pdf")


def generate_line_plot():
    
    gpu_times, bmarks = (
        np.array([96.25812109, 39.12650584, 31.15540277, 24.54303205, 59.90921479, 37.90189175,
        33.42215541, 31.3451803,  27.05915711, 23.49124462, 28.8020803,  20.23592472,
        16.95063561, 15.94066052]),
        np.array([
            5040,
            9821,
            13507,
            15614,
            7867,
            11631,
            12710,
            14113,
            16049,
            18653,
            10750,
            15223,
            22149,
            25046,
        ])
    )
    gpu_times = gpu_times[1:]
    bmarks = bmarks[1:]
    bmarks = max(bmarks) - bmarks + min(bmarks)

    groups = [3, 6, 9, 13]
    start = 0
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list(
        "my_cmap", ["#76b900", "black"]  # start → end
    )

    fig, ax = plt.subplots(figsize=(4, 4), constrained_layout=True)
    for i, generation in enumerate(groups):
        current_gpus = gpu_times[start:generation]
        current_bmarks = bmarks[start:generation]
        current_gpus = np.array(current_gpus) / np.mean(current_gpus) / 2
        current_bmarks = np.array(current_bmarks) / np.mean(current_bmarks) / 2
        maximum = max(max(current_bmarks), max(current_gpus))
        current_gpus = current_gpus + 1 - maximum
        current_bmarks = current_bmarks + 1 - maximum
        if i == 2:
            ax.plot(
                np.linspace(start + i, generation + i, generation - start),
                current_gpus,
                marker="o",
                markerfacecolor="none",
                markeredgecolor=cmap((i / 3) ** 1.5),
                markeredgewidth=1.5,
                color=cmap((i / 3) ** 1.5),
                linestyle="--",
                label="Bouquet",
            )
            ax.plot(
                np.linspace(start + i, generation + i, generation - start),
                current_bmarks,
                marker="s",
                markerfacecolor="none",
                markeredgecolor=cmap((i / 3) ** 1.5),
                markeredgewidth=1.5,
                color=cmap((i / 3) ** 1.5),
                linestyle="--",
                label="Benchmark",
            )
        else:
            ax.plot(
                np.linspace(start + i, generation + i, generation - start),
                current_gpus,
                marker="o",
                markerfacecolor="none",
                markeredgecolor=cmap((i / 3) ** 1.5),
                markeredgewidth=1.5,
                color=cmap((i / 3) ** 1.5),
                linestyle="--",
            )
            ax.plot(
                np.linspace(start + i, generation + i, generation - start),
                current_bmarks,
                marker="s",
                markerfacecolor="none",
                markeredgecolor=cmap((i / 3) ** 1.5),
                markeredgewidth=1.5,
                color=cmap((i / 3) ** 1.5),
                linestyle="--",
            )

        ax.axvline(x=start + i, color="black", linestyle="--", linewidth=0.5, alpha=0.4)
        start = generation

    ax.set_xticks([0, 5, 9, 13])
    ax.set_xticklabels(["GTX 10xx", "GTX 16xx", "RTX 20xx", "RTX 30xx"], ha="left")

    for y in [0.4, 0.6, 0.8]:
        ax.axhline(y=y, color="gray", linestyle="--", linewidth=0.5, alpha=0.4)
    plt.xlabel(" ", fontsize=10)

    plt.legend(handlelength=0, handletextpad=1, borderpad=0.65, frameon=True)
    plt.show()
    plt.savefig("plots/line_plot.pdf")


generate_scatter_plot()
generate_line_plot()
