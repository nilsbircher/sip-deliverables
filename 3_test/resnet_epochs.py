from pathlib import Path
from PIL import Image
from tqdm import tqdm

import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F

import torchmetrics.detection as detection_metrics

classes = ["tree"]
num_classes = len(classes) + 1

if torch.cuda.is_available():
    device = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
    torch.set_num_threads(12)
    torch.set_num_interop_threads(12)


def generate_validation_csv(
    result_path: Path, val_data_types: list[str], csv_data: dict, overwrite: bool
):
    if not result_path.exists():
        result_path.mkdir(parents=True, exist_ok=True)

    for val_data_type in val_data_types:
        if (result_path / f"{val_data_type}.csv").exists() and not overwrite:
            print(
                "Files was already generated and overwrite is set to false, writing results into it."
            )
        else:
            with open(result_path / f"{val_data_type}.csv", "w") as f:
                f.write(",".join(csv_data.keys()) + "\n")


def load_ground_truth(data_dir):
    """Load ground truth annotations from YOLO format labels."""
    gt_boxes = {}

    labels_dir = data_dir / "labels"
    if not labels_dir.exists():
        raise FileNotFoundError(f"Labels directory not found: {labels_dir}")

    images_dir = data_dir / "images"
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    for label_file in labels_dir.glob("*.txt"):
        image_file = images_dir / f"{label_file.stem}.jpg"

        with Image.open(image_file) as img:
            img_width, img_height = img.size

        boxes = []
        with open(label_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                class_id = int(parts[0])
                # YOLO format: class_id, x_center, y_center, width, height (normalized)
                x_center, y_center, width, height = map(float, parts[1:5])

                # Convert to absolute coordinates: xmin, ymin, xmax, ymax
                xmin = (x_center - width / 2) * img_width
                ymin = (y_center - height / 2) * img_height
                xmax = (x_center + width / 2) * img_width
                ymax = (y_center + height / 2) * img_height

                # Add to boxes list: [xmin, ymin, xmax, ymax, class_id]
                boxes.append([xmin, ymin, xmax, ymax, class_id])

        gt_boxes[image_file.name] = boxes

    return gt_boxes, images_dir


def prepare_data_for_torchmetrics(gt_data, pred_data):
    formatted_preds = []
    formatted_targets = []

    all_image_keys = sorted(
        list(set(gt_data.keys()).union(set(pred_data.keys()))))

    for img_key in all_image_keys:
        # Process ground truth
        gt_boxes = []
        gt_labels = []
        if img_key in gt_data:
            for bbox in gt_data[img_key]:
                gt_boxes.append(bbox[:4])  # xmin, ymin, xmax, ymax
                gt_labels.append(bbox[4])

        # If no ground truth boxes for an image, create empty tensors
        if not gt_boxes:
            gt_boxes_tensor = torch.empty((0, 4), dtype=torch.float32)
            gt_labels_tensor = torch.empty((0,), dtype=torch.int64)
        else:
            gt_boxes_tensor = torch.tensor(gt_boxes, dtype=torch.float32)
            gt_labels_tensor = torch.tensor(gt_labels, dtype=torch.int64)

        formatted_targets.append(
            {"boxes": gt_boxes_tensor, "labels": gt_labels_tensor})

        # Process predictions
        pred_boxes = []
        pred_scores = []
        pred_labels = []
        if img_key in pred_data:
            for bbox in pred_data[img_key]:
                pred_boxes.append(bbox[:4])  # xmin, ymin, xmax, ymax
                pred_labels.append(0)  # Single class
                pred_scores.append(bbox[4])  # Confidence

        # If no predicted boxes for an image, create empty tensors
        if not pred_boxes:
            pred_boxes_tensor = torch.empty((0, 4), dtype=torch.float32)
            pred_scores_tensor = torch.empty((0,), dtype=torch.float32)
            pred_labels_tensor = torch.empty((0,), dtype=torch.int64)
        else:
            pred_boxes_tensor = torch.tensor(pred_boxes, dtype=torch.float32)
            pred_scores_tensor = torch.tensor(pred_scores, dtype=torch.float32)
            pred_labels_tensor = torch.tensor(pred_labels, dtype=torch.int64)

        formatted_preds.append(
            {
                "boxes": pred_boxes_tensor,
                "scores": pred_scores_tensor,
                "labels": pred_labels_tensor,
            }
        )

    return formatted_preds, formatted_targets


def calculate_metrics(predictions, ground_truths, iou_threshold=0.5):
    """Calculate metrics using torchmetrics."""
    # Initialize torchmetrics detection metrics
    map_metric = detection_metrics.MeanAveragePrecision(
        box_format="xyxy",  # [x1, y1, x2, y2] format
        iou_type="bbox",
        iou_thresholds=[
            iou_threshold,
            0.55,
            0.6,
            0.65,
            0.7,
            0.75,
            0.8,
            0.85,
            0.9,
            0.95,
        ],
        max_detection_thresholds=[50, 75, 100],  # Maximum detections per image
        class_metrics=True,
    ).to(device)

    formatted_preds, formatted_targets = prepare_data_for_torchmetrics(
        ground_truths, predictions
    )
    map_metric.update(formatted_preds, formatted_targets)

    # Compute metrics
    result = map_metric.compute()

    map50 = float(result["map_50"].cpu().item())
    map50_95 = float(result["map"].cpu().item())
    mar50 = float(result["mar_50"].cpu().item())  # Most similar to YOLO recall
    mar100 = float(result["mar_100"].cpu().item())

    return {
        "mAP50": map50,
        "mAP50-95": map50_95,
        "mAR50": mar50,
        "mAR100": mar100,
    }


def validate_model(model_path: Path, data_dir: str, confidence_threshold=0.25):
    """Validate a model on the given dataset."""
    print(f"Validating model: {model_path}")

    model = fasterrcnn_resnet50_fpn(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    ground_truths, images_dir = load_ground_truth(
        Path(__file__).parent.parent / data_dir
    )

    predictions = {}

    for img_path in tqdm(list(images_dir.glob("*.jpg"))):
        img_name = img_path.name

        img = Image.open(img_path).convert("RGB")
        img_tensor = F.to_tensor(img).to(device)
        input_batch = [img_tensor]

        with torch.no_grad():
            outputs = model(input_batch)

        # Extract predictions
        boxes = outputs[0]["boxes"].cpu().numpy()
        scores = outputs[0]["scores"].cpu().numpy()
        labels = outputs[0]["labels"].cpu().numpy()

        mask = scores > confidence_threshold
        boxes = boxes[mask]
        scores = scores[mask]
        labels = labels[mask]

        pred_boxes = []
        for box, score, label in zip(boxes, scores, labels):
            pred_boxes.append([box[0], box[1], box[2], box[3], score])

        predictions[img_name] = pred_boxes

    metrics = calculate_metrics(predictions, ground_truths)

    return metrics


def main():
    val_data_types = ["validation-favorable"]

    experiment_path = Path("/workspaces/sip-deliverables") / \
        "10_data" / "models" / "ResNet_epochs"
    results_path = Path("/workspaces/sip-deliverables") / \
        "10_data" / "results" / "csv" / "resnet_epochs"

    if not experiment_path.exists():
        experiment_path.mkdir(parents=True, exist_ok=True)

    csv_data = {
        "model": None,
        "mAP50": 0,
        "mAP50-95": 0,
        "mAR50": 0,
        "mAR100": 0,
    }

    generate_validation_csv(results_path, val_data_types, csv_data, False)

    models = list(experiment_path.glob("*.pth"))

    for model_path in models:
        for val_data_type in val_data_types:
            metrics = validate_model(model_path, val_data_type)

            csv_data["model"] = model_path.name
            csv_data["mAP50"] = metrics["mAP50"]
            csv_data["mAP50-95"] = metrics["mAP50-95"]
            csv_data["mAR50"] = metrics["mAR50"]
            csv_data["mAR100"] = metrics["mAR100"]

            with open(results_path / f"{val_data_type}.csv", "a") as f:
                f.write(",".join([str(v) for v in csv_data.values()]) + "\n")


if __name__ == "__main__":
    main()
