from pathlib import Path
from ultralytics import YOLO


def extract_data_from_model_name(model_name: str, extractor: str) -> str | None:
    parts = model_name.split("__")
    for part in parts:
        if extractor in part:
            value = part.split("-")[-1]
            return value
    return None


def main():
    val_data_types = ["validation-favorable", "validation-random"]
    experiments = [
        "augmentation",
        "data_amount",
        "model_architecture",
        "model_size",
        "model_version",
    ]

    extractors = [
        "augment",
        "data_subset",
        "model_name",
        "model_name",
        "model_name",
    ]

    experiments_path = Path("/workspaces/sip-deliverables") / \
        "10_data" / "models" / "experiments"
    results_path = Path("/workspaces/sip-deliverables") / \
        "10_data" / "results" / "images"

    for experiment, extractor in zip(experiments, extractors):
        experiment_path = experiments_path / experiment
        models = list(experiment_path.glob("*.pt"))

        for model_path in models:
            for val_data_type in val_data_types:
                model = YOLO(model_path)

                img_folder = Path("/workspaces/sip-deliverables") / \
                    "10_data" / val_data_type

                if not (results_path / experiment / val_data_type).exists():
                    (results_path / experiment / val_data_type).mkdir(
                        parents=True, exist_ok=True
                    )

                for img_path in img_folder.glob("**/*.jpg"):
                    _key = extract_data_from_model_name(
                        model_name=model_path.name, extractor=extractor
                    )

                    results = model.predict(source=img_path, conf=0.25)

                    output_path = (
                        results_path
                        / experiment
                        / val_data_type
                        / f"{img_path.stem}_{_key}.jpg"
                    )

                    for result in results:
                        result.save(filename=str(output_path))


if __name__ == "__main__":
    main()
