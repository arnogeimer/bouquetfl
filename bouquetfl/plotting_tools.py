import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

import altair as alt

def delete_unused_files():
    import os

    files = os.listdir("./bouquetfl/checkpoints/")
    for file in files:
        if file.endswith(".npz"):
            os.remove(os.path.join("./bouquetfl/checkpoints/", file))
delete_unused_files()


def plot_gpu_average_training_times():

    results = pd.read_pickle("./bouquetfl/checkpoints/load_and_training_times.pkl")
    results["total_load_time"] = np.nan
    results["total_train_time"] = np.nan
    num_rounds = int(results.shape[1] / 2 - 2)
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

    df_long = results.melt(
        id_vars=["gpu", "cpu", "total_load_time", "total_train_time", "1load_time"],
        var_name="segment",
        value_name="value"
    )

    df_long["segment"] = df_long["segment"].apply(lambda x: "load_time" if "load_time" in x else "train_time")

    gpu_chart = alt.Chart(df_long).mark_bar(
        cornerRadiusBottomLeft=2,
        cornerRadiusTopLeft=2,
        cornerRadiusBottomRight=2,
        cornerRadiusTopRight=2
    ).encode(
        x=alt.X('value:Q', title = "training time (s)"),
        y=alt.Y('gpu').sort('-x'),
    )

    gpu_chart.save('./bouquetfl/plots/timetable_gpu.pdf')
    gpu_chart.save('./bouquetfl/plots/timetable_gpu.png')

    cpu_chart = alt.Chart(df_long).mark_bar(
        cornerRadiusBottomLeft=2,
        cornerRadiusTopLeft=2,
        cornerRadiusBottomRight=2,
        cornerRadiusTopRight=2
    ).encode(
        x=alt.X('value:Q', title = "training time (s)"),
        y=alt.Y('cpu').sort('-x'),
    )

    cpu_chart.save('./bouquetfl/plots/timetable_cpu.pdf')
    cpu_chart.save('./bouquetfl/plots/timetable_cpu.png')

plot_gpu_average_training_times()
