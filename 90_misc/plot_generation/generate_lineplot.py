from pathlib import Path

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def generate_all_plots():
    results_path = Path(__file__).parent.parent / "results"
    csv_path = results_path / "csv"
    plots_path = results_path / "plots"

    custom_color = ["#0F4E6A", "#A1AFBB", "#156082", "#2396D1"]

    metrics = [
        "mAP50",
        "mAP50-95",
        "recall",
    ]

    plot_data = pd.read_csv(csv_path / "yolo11n_epochs" / "results.csv")

    plot_data = plot_data.melt(
        id_vars=["epoch"], var_name="metrics", value_name="value"
    )

    plot_data = plot_data[plot_data["metrics"].isin(metrics)]

    plt.figure(figsize=(10, 6))

    ax = sns.lineplot(
        plot_data,
        x="epoch",
        y="value",
        hue="metrics",
        estimator="avg",
        errorbar=None,
        palette=custom_color,
        zorder=2,
    )
    ax.yaxis.grid(True, linestyle="-", linewidth=0.7, alpha=0.7, zorder=1)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.tick_params(axis="x", labelsize=14)
    ax.tick_params(axis="y", labelsize=14)

    plt.ylim(0, 1)
    plt.ylabel("Performance", fontsize=18)

    plt.xlabel("Epochs", fontsize=18)

    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0, fontsize=15)

    plt.tight_layout()
    plt.savefig(plots_path / "yolo11n_epochs.png", bbox_inches="tight")
    plt.close()

    metrics = [
        "mAP50",
        "mAP50-95",
        "mAR50",
    ]

    plot_data = pd.read_csv(csv_path / "resnet_epochs" / "validation-favorable.csv")

    plot_data = plot_data.melt(
        id_vars=["model"], var_name="metrics", value_name="value"
    )

    plot_data = plot_data[plot_data["metrics"].isin(metrics)]
    plot_data["epoch"] = plot_data["model"].apply(
        lambda x: int(x.split("_")[-1].split(".")[0])
    )

    plt.figure(figsize=(10, 6))

    ax = sns.lineplot(
        plot_data,
        x="epoch",
        y="value",
        hue="metrics",
        errorbar=None,
        palette=custom_color,
        zorder=2,
    )
    ax.yaxis.grid(True, linestyle="-", linewidth=0.7, alpha=0.7, zorder=1)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.tick_params(axis="x", labelsize=14)
    ax.tick_params(axis="y", labelsize=14)

    plt.ylim(0, 1)
    plt.ylabel("Performance", fontsize=18)

    plt.xlabel("Epochs", fontsize=18)

    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0, fontsize=15)

    plt.tight_layout()
    plt.savefig(plots_path / "resnet_epochs.png", bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    generate_all_plots()
    print("All plots generated successfully!")
