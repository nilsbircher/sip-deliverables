from data_filters import filter_function_lookup
from helper import gen_run_name
import sys
import random
from shutil import copy2
from pathlib import Path

sys.path.append(Path(__file__).parent)


SEED = 42
random.seed(SEED)


def delete_previous_files(folders: list[Path]):
    for folder in folders:
        for file in list(folder.glob("*.jpg")) + list(folder.glob("*.txt")):
            file.unlink()


def create_train_val_split(params: dict):
    split_ratio: float = params.get("split_ratio")

    enable_threshold: bool = params.get("enable_threshold")
    threshold: float = params.get("threshold")
    data_subset: float = params.get("data_subset")

    if enable_threshold:
        filter_function_name = "threshold_filter"
        data_threshold: float = threshold
    else:
        filter_function_name = "subset_filter"
        data_threshold: float = data_subset

    filter_function: callable = filter_function_lookup[filter_function_name]

    input_image_folder = Path(
        "/workspaces/sip-deliverables/10_data/labels/images")
    input_label_folder = Path(
        "/workspaces/sip-deliverables/10_data/labels/labels")

    train_folder = Path(
        f"/workspaces/sip-deliverables/10_data/runs/{gen_run_name(params)}/train"
    )

    train_folder.mkdir(parents=True, exist_ok=True)

    image_train_folder = train_folder / "images" / "train"
    image_val_folder = train_folder / "images" / "val"
    label_train_folder = train_folder / "labels" / "train"
    label_val_folder = train_folder / "labels" / "val"

    with open(train_folder / "data.yaml", "w") as f:
        f.writelines(f"path: {train_folder}\n")
        f.writelines(f"train: {str(image_train_folder)}\n")
        f.writelines(f"val: {str(image_val_folder)}\n")
        f.writelines("names: ['tree']")

    image_train_folder.mkdir(parents=True, exist_ok=True)
    image_val_folder.mkdir(parents=True, exist_ok=True)
    label_train_folder.mkdir(parents=True, exist_ok=True)
    label_val_folder.mkdir(parents=True, exist_ok=True)

    all_files = list(input_image_folder.glob("*.jpg"))
    random.shuffle(all_files)

    all_files = (
        filter_function(all_files, input_label_folder, data_threshold)
        if filter_function
        else all_files
    )

    split_idx = int(len(all_files) * split_ratio)
    train_files = all_files[:split_idx]
    val_files = all_files[split_idx:]

    for file in train_files:
        copy2(file, image_train_folder)
        copy2(input_label_folder / f"{file.stem}.txt", label_train_folder)

    for file in val_files:
        copy2(file, image_val_folder)
        copy2(input_label_folder / f"{file.stem}.txt", label_val_folder)

    print(f"Train folder: {train_folder.parts[-2]}")
    print(f"Copied {len(train_files)} files to {str(image_train_folder)}")
    print(f"Copied {len(val_files)} files to {str(image_val_folder)}")
