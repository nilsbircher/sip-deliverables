from helper import gen_run_name
import sys
from pathlib import Path

from ultralytics import YOLO

sys.path.append(Path(__file__).parent)

try:
    from create_train_val_split import create_train_val_split
except ImportError:
    print("Error importing modules")


def train(model_name: str, epochs: int, optimizer: str, augment: bool = True, **kwargs):
    run_name = gen_run_name(run_config)
    data_location = Path(
        f"/workspaces/sip-deliverables/10_data/runs/{run_name}/train")

    model = YOLO(f"{model_name}.pt")

    model.train(
        data=str(data_location / "data.yaml"),
        name=run_name,
        epochs=epochs,
        imgsz=640,
        batch=16,
        seed=42,
        optimizer=optimizer,
        augment=augment,
    )


if __name__ == "__main__":
    run_config = {
        "model_name": "yolo11n",
        "epochs": 500,
        "optimizer": "auto",
        "split_ratio": 0.8,
        "enable_threshold": False,
        "threshold": 0.01,
        "data_subset": 1,
        "augment": True,
    }
    create_train_val_split(run_config)
    train(**run_config)
