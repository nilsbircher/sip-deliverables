from pathlib import Path
from ultralytics import YOLO
from PIL import Image, ImageFont, ImageDraw
import torch
from torchvision.transforms import functional as F
from torchvision.models.detection import fasterrcnn_resnet50_fpn

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


def extract_data_from_model_name(model_name: str, extractor: str) -> str | None:
    parts = model_name.split("__")
    for part in parts:
        if extractor in part:
            value = part.split("-")[-1]
            return value
    return None


def create_stacked_bbox_image(
    original_image_path: Path,
    results: dict,
    output_path: Path,
):
    if not output_path.parent.exists():
        output_path.parent.mkdir(parents=True, exist_ok=True)

    custom_color = [
        "#0F4E6A",
        "#A1AFBB",
        "#156082",
    ]

    for image_name, model_results in results.items():
        img = Image.open(original_image_path / f"{image_name}.jpg")
        width, height = img.size

        legend_height = 30 * len(model_results)
        new_img = Image.new(
            "RGB", (width, height + legend_height), color="white")
        new_img.paste(img, (0, 0))

        draw = ImageDraw.Draw(new_img)

        legend_y = height + 10

        for idx, (model_name, boxes) in enumerate(model_results.items()):
            draw.rectangle(
                [(10, legend_y), (30, legend_y + 15)],
                fill=custom_color[idx],
            )
            draw.text(
                (35, legend_y),
                model_name,
                fill="#000000",
                font=ImageFont.load_default(),
            )
            legend_y += 25
            for box in boxes:
                x1, y1, x2, y2, _ = box
                r, g, b = tuple(
                    int(custom_color[idx].lstrip("#")[i: i + 2], 16) for i in (0, 2, 4)
                )
                draw_rect = ImageDraw.Draw(new_img, "RGBA")
                draw_rect.rectangle(
                    [x1, y1, x2, y2],
                    outline=(r, g, b, 200),
                    width=4,
                )

        if not (output_path / "stacked").exists():
            (output_path / "stacked").mkdir(parents=True, exist_ok=True)

        new_img.save(output_path / "stacked" / f"{image_name}.jpg", quality=95)


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
        models.extend(experiment_path.glob("*.pth"))

        for val_data_type in val_data_types:
            results = {}

            for model_path in models:
                if model_path.suffix == ".pt":
                    model = YOLO(model_path)

                    img_folder = Path(
                        "/workspaces/sip-deliverables/10_data") / val_data_type

                    if not (results_path / experiment / val_data_type).exists():
                        (results_path / experiment / val_data_type).mkdir(
                            parents=True, exist_ok=True
                        )

                    for img_path in img_folder.glob("**/*.jpg"):
                        _key = extract_data_from_model_name(
                            model_name=model_path.name, extractor=extractor
                        )
                        if img_path.stem not in results.keys():
                            results[img_path.stem] = {}
                        results[img_path.stem][_key] = []

                        for box in model.predict(source=img_path, conf=0.25)[0].boxes:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            score = box.conf[0].item()
                            results[img_path.stem][_key].append(
                                [x1, y1, x2, y2, score])

                elif model_path.suffix == ".pth":
                    model = fasterrcnn_resnet50_fpn(num_classes=num_classes)
                    model.load_state_dict(torch.load(
                        model_path, map_location=device))
                    model.to(device)
                    model.eval()

                    img_folder = Path(
                        "/workspaces/sip-deliverables/10_data/") / val_data_type

                    for img_path in img_folder.glob("**/*.jpg"):
                        print(f"Processing {img_path}")
                        img = Image.open(img_path).convert("RGB")
                        img_tensor = F.to_tensor(img).to(device)

                        input_batch = [img_tensor]

                        with torch.no_grad():
                            predictions = model(input_batch)

                        _key = extract_data_from_model_name(
                            model_name=model_path.name, extractor=extractor
                        )
                        if img_path.stem not in results.keys():
                            results[img_path.stem] = {}
                        results[img_path.stem][_key] = []

                        for box, score in zip(
                            predictions[0]["boxes"],
                            predictions[0]["scores"],
                        ):
                            if score > 0.25:
                                results[img_path.stem][_key].append(
                                    [*box, score])

            output_path = results_path / experiment / val_data_type

            create_stacked_bbox_image(
                original_image_path=Path(
                    "/workspaces/sip-deliverables/10_data/")
                / val_data_type
                / "images",
                results=results,
                output_path=output_path,
            )


if __name__ == "__main__":
    main()
