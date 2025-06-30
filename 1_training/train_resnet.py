import re
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn

sys.path.append(Path(__file__).parent)

try:
    from create_train_val_split import create_train_val_split
    from helper import gen_run_name
    from dataset import YoloDataset
except ImportError:
    print(
        "create_train_val_split module not found. Ensure it is in the same directory as this script."
    )


def train(params: dict):
    classes = ["tree"]
    run_name = gen_run_name(params)
    print(f"Run name: {run_name}")

    data_location = Path(
        f"/workspaces/sip-deliverables/10_data/runs/{run_name}/train")

    save_dir = Path(
        f"/workspaces/sip-deliverables/10_data/runs/{run_name}/train")
    save_dir.mkdir(parents=True, exist_ok=True)

    # Load your data
    train_dataset = YoloDataset(
        images_dir=data_location / "images" / "train",
        labels_dir=data_location / "labels" / "train",
        classes=classes,
    )
    train_loader = DataLoader(
        train_dataset, batch_size=2, shuffle=True, collate_fn=lambda x: tuple(zip(*x))
    )

    # Load model
    model = fasterrcnn_resnet50_fpn(num_classes=len(classes) + 1)
    model.train()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Optimizer
    optimizer_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        optimizer_params, lr=0.005, momentum=0.9, weight_decay=0.0005
    )

    # Resume logic
    start_epoch = 0
    checkpoint_files = list(save_dir.glob(f"{run_name}_model_*.pth"))

    if checkpoint_files:
        print(f"Found {len(checkpoint_files)} checkpoint files.")

        latest_checkpoint = max(
            checkpoint_files, key=lambda f: int(
                re.findall(r"_(\d+).pth", f.name)[0])
        )
        start_epoch = int(re.findall(
            r"_(\d+).pth", latest_checkpoint.name)[0]) + 1

        print(
            f"Resuming training from checkpoint: {latest_checkpoint.name}, starting at epoch {start_epoch}"
        )
        model.load_state_dict(torch.load(
            latest_checkpoint, map_location=device))
    else:
        print("No checkpoint found. Starting from scratch.")

    num_epochs = params["epochs"]
    for epoch in range(start_epoch, num_epochs):
        print(f"Epoch [{epoch}/{num_epochs}]")
        epoch_loss = 0.0

        start_time = time.time()

        for images, targets in train_loader:
            images = list(img.to(device) for img in images)
            targets = [{k: v.to(device) for k, v in t.items()}
                       for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

        print(
            f"Epoch [{epoch}] completed in {time.time() - start_time:.1f}s - "
            f"Loss: {epoch_loss / len(train_loader):.4f} - "
            f"LR: {optimizer.param_groups[0]['lr']:.6f}"
        )

        torch.save(
            model.state_dict(
            ), f"/workspaces/sip-deliverables/10_data/runs/{run_name}/train/{run_name}_model_{epoch}.pth"
        )
    torch.save(
        model, f"/workspaces/sip-deliverables/10_data/runs/{run_name}/train/{run_name}_model_raw.pth")
    torch.save(model.state_dict(
    ), f"/workspaces/sip-deliverables/10_data/runs/{run_name}/train/{run_name}_model.pth")


if __name__ == "__main__":
    run_config = {
        "model_name": "ResNet",
        "epochs": 100,
        "optimizer": "auto",
        "split_ratio": 0.8,
        "enable_threshold": False,
        "threshold": 0.01,
        "data_subset": 1,
        "augment": True,
    }
    create_train_val_split(run_config)
    train(run_config)
