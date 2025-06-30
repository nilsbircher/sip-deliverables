from pathlib import Path
from ultralytics import YOLO


def generate_validation_csv(
    result_path: Path, val_data_types: list[str], csv_data: dict
):
    if not result_path.exists():
        result_path.mkdir(parents=True, exist_ok=True)

    for val_data_type in val_data_types:
        with open(result_path / f"{val_data_type}.csv", "w") as f:
            f.write(",".join(csv_data.keys()) + "\n")


def main():
    val_data_types = ["validation-favorable", "validation-random"]
    experiments = [
        "augmentation",
        "data_amount",
        "model_architecture",
        "model_size",
        "model_version",
        "confidence_threshold",
    ]
    experiments_path = Path("/workspaces/sip-deliverables") / \
        "10_data" / "models" / "experiments"
    results_path = Path("/workspaces/sip-deliverables") / \
        "10_data" / "results" / "csv"

    for experiment in experiments:
        if experiment in ["confidence_threshold", "model_architecture"]:
            csv_data = {
                "model": None,
                "precision": 0,
                "recall": 0,
                "mAP50": 0,
                "mAP50-95": 0,
                "mAR50": 0,
                "mAR100": 0,
                "fitness": 0,
                "F1": 0,
            }
            generate_validation_csv(
                results_path / experiment, val_data_types, csv_data)

        else:
            csv_data = {
                "model": None,
                "precision": 0,
                "recall": 0,
                "mAP50": 0,
                "mAP50-95": 0,
                "fitness": 0,
                "F1": 0,
            }

            generate_validation_csv(
                results_path / experiment, val_data_types, csv_data)

        experiment_path = experiments_path / experiment
        models = list(experiment_path.glob("*.pt"))

        for model_path in models:
            model = YOLO(model_path)

            for val_data_type in val_data_types:
                if experiment == "confidence_threshold":
                    for conf in [0.25, 0.5, 0.75]:
                        metrics = model.val(
                            data=f"/workspaces/sip-deliverables/10_data/{val_data_type}/data.yaml",
                            split="val",
                            conf=conf,
                        )

                        csv_data["model"] = f"conf-YOLO_{conf}"
                        csv_data["precision"] = metrics.results_dict[
                            "metrics/precision(B)"
                        ]
                        csv_data["recall"] = metrics.results_dict["metrics/recall(B)"]
                        csv_data["mAP50"] = metrics.results_dict["metrics/mAP50(B)"]
                        csv_data["mAP50-95"] = metrics.results_dict[
                            "metrics/mAP50-95(B)"
                        ]
                        csv_data["fitness"] = metrics.results_dict["fitness"]
                        csv_data["F1"] = (
                            2
                            * (csv_data["precision"] * csv_data["recall"])
                            / (csv_data["precision"] + csv_data["recall"])
                            if (csv_data["precision"] + csv_data["recall"]) > 0
                            else 0
                        )

                        with open(
                            results_path / experiment /
                                f"{val_data_type}.csv", "a"
                        ) as f:
                            f.write(
                                ",".join([str(v)
                                         for v in csv_data.values()]) + "\n"
                            )
                else:
                    metrics = model.val(
                        data=f"/workspaces/sip-deliverables/10_data/{val_data_type}/data.yaml",
                        split="val",
                        conf=0.25,
                    )

                    csv_data["model"] = model_path.name
                    csv_data["precision"] = metrics.results_dict["metrics/precision(B)"]
                    csv_data["recall"] = metrics.results_dict["metrics/recall(B)"]
                    csv_data["mAP50"] = metrics.results_dict["metrics/mAP50(B)"]
                    csv_data["mAP50-95"] = metrics.results_dict["metrics/mAP50-95(B)"]
                    csv_data["fitness"] = metrics.results_dict["fitness"]
                    csv_data["F1"] = (
                        2
                        * (csv_data["precision"] * csv_data["recall"])
                        / (csv_data["precision"] + csv_data["recall"])
                        if (csv_data["precision"] + csv_data["recall"]) > 0
                        else 0
                    )

                    with open(
                        results_path / experiment / f"{val_data_type}.csv", "a"
                    ) as f:
                        f.write(",".join([str(v)
                                for v in csv_data.values()]) + "\n")


if __name__ == "__main__":
    main()
