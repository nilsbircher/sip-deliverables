from pathlib import Path

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from helper import extract_data_from_model_name


def generate_plot(
    experiment_name: str,
    extractor: str,
    metrics: list,
    csv_path: Path,
    plots_path: Path,
):
    favorable_df = pd.read_csv(csv_path / experiment_name / "validation-favorable.csv")
    random_df = pd.read_csv(csv_path / experiment_name / "validation-random.csv")

    custom_color = ["#0F4E6A", "#A1AFBB", "#156082", "#2396D1"]

    for dataset_type, plot_data in [("favorable", favorable_df), ("random", random_df)]:
        plot_data["model"] = plot_data.apply(
            lambda row: extract_data_from_model_name(row["model"], extractor), axis=1
        )
        plot_data = plot_data.melt(
            id_vars=["model"], var_name="metrics", value_name="value"
        )

        plot_data = plot_data[plot_data["metrics"].isin(metrics)]

        plt.figure(figsize=(10, 6))

        ax = sns.barplot(
            plot_data,
            x="metrics",
            y="value",
            hue="model",
            estimator="sum",
            errorbar=None,
            gap=0.2,
            palette=custom_color,
            zorder=2,
        )
        ax.yaxis.grid(True, linestyle="-", linewidth=0.7, alpha=0.7, zorder=1)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax.tick_params(axis="x", labelsize=14)
        ax.tick_params(axis="y", labelsize=14)

        plt.ylim(0, 1)
        plt.ylabel("Detection Performance", fontsize=18)

        plt.xlabel("Metrics", fontsize=18)

        plt.legend(
            bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0, fontsize=15
        )

        plt.tight_layout()
        plt.savefig(
            plots_path / f"{experiment_name}_{dataset_type}.png", bbox_inches="tight"
        )
        plt.close()


def generate_all_plots():
    results_path = Path(__file__).parent.parent / "results"
    csv_path = results_path / "csv"
    plots_path = results_path / "plots"

    metrics = [
        "precision",
        "recall",
        "F1",
    ]

    generate_plot(
        experiment_name="augmentation",
        extractor="augment",
        metrics=metrics,
        csv_path=csv_path,
        plots_path=plots_path,
    )

    generate_plot(
        experiment_name="data_amount",
        extractor="data_subset",
        metrics=metrics,
        csv_path=csv_path,
        plots_path=plots_path,
    )

    generate_plot(
        experiment_name="model_size",
        extractor="model_name",
        metrics=metrics,
        csv_path=csv_path,
        plots_path=plots_path,
    )

    generate_plot(
        experiment_name="model_version",
        extractor="model_name",
        metrics=metrics,
        csv_path=csv_path,
        plots_path=plots_path,
    )

    # Mix of ResNet and YOLO models, different metrics required for eval

    metrics = [
        "mAP50",
        "mAP50-95",
        "recall",
        "mAR50",
    ]

    generate_plot(
        experiment_name="model_architecture",
        extractor="model_name",
        metrics=metrics,
        csv_path=csv_path,
        plots_path=plots_path,
    )

    generate_plot(
        experiment_name="confidence_threshold",
        extractor="conf",
        metrics=metrics,
        csv_path=csv_path,
        plots_path=plots_path,
    )


if __name__ == "__main__":
    generate_all_plots()
    print("All plots generated successfully!")
