from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

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


def create_bbox_image(
    original_image_path: Path,
    results: dict,
    output_path: Path,
    colors: dict,
):
    if not output_path.parent.exists():
        output_path.parent.mkdir(parents=True, exist_ok=True)

    img = Image.open(original_image_path)
    draw = ImageDraw.Draw(img)

    legend_y = 10
    for model_name in colors.keys():
        draw.rectangle([(10, legend_y), (30, legend_y + 15)],
                       fill=colors[model_name])
        draw.text(
            (35, legend_y),
            model_name,
            fill=colors[model_name],
            font=ImageFont.load_default(size=20),
        )
        legend_y += 25

    for model_name, boxes in results.items():
        color = colors[model_name]
        for box in boxes:
            x1, y1, x2, y2, score = box

            text = f"{score:.2f}"
            font = ImageFont.load_default(size=15)
            text_bbox = draw.textbbox((0, 0), text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]

            text_box = (
                x1,
                y1 - text_height - 4,
                x1 + text_width + 4,
                y1,
            )

            if (y1 - text_height - 4) < 0:
                text_box = (
                    x1,
                    y2,
                    x1 + text_width + 4,
                    y2 + text_height + 4,
                )

            draw.rectangle(
                text_box,
                fill=color,
                outline=color,
            )

            text_x_pos = text_box[0] + 2
            text_y_pos = text_box[1] - 2

            draw.text(
                (text_x_pos, text_y_pos),
                text,
                fill="white",
                font=font,
            )

            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

    img.save(output_path)


def main():
    val_data_types = ["validation-favorable", "validation-random"]
    experiments = [
        "model_architecture",
    ]

    experiments_path = Path("/workspaces/sip-deliverables") / \
        "10_data" / "models" / "experiments"
    results_path = Path("/workspaces/sip-deliverables") / \
        "10_data" / "results" / "images"

    for experiment in experiments:
        experiment_path = experiments_path / experiment
        models = list(experiment_path.glob("*.pth"))

        for model_path in models:
            for val_data_type in val_data_types:
                model = fasterrcnn_resnet50_fpn(num_classes=num_classes)
                model.load_state_dict(torch.load(
                    model_path, map_location=device))
                model.to(device)
                model.eval()

                img_folder = Path("/workspaces/sip-deliverables") / \
                    "10_data" / val_data_type

                for img_path in img_folder.glob("**/*.jpg"):
                    print(f"Processing {img_path}")
                    img = Image.open(img_path).convert("RGB")
                    img_tensor = F.to_tensor(img).to(device)

                    input_batch = [img_tensor]

                    with torch.no_grad():
                        predictions = model(input_batch)

                    for box, label, score in zip(
                        predictions[0]["boxes"],
                        predictions[0]["labels"],
                        predictions[0]["scores"],
                    ):
                        if score > 0.25:  # confidence threshold
                            print(
                                f"Detected {classes[label - 1]} at {box.cpu().numpy()} with confidence {score:.2f}"
                            )

                    model_name = [
                        v.split("-")[1]
                        for v in model_path.name.split("__")
                        if "model_name" in v
                    ][0]

                    output_path = (
                        results_path
                        / experiment
                        / val_data_type
                        / f"{img_path.stem}_{model_name}.jpg"
                    )

                    create_bbox_image(
                        original_image_path=img_path,
                        results={
                            "resnet": [
                                box.cpu().numpy().tolist() + [score.item()]
                                for box, score in zip(
                                    predictions[0]["boxes"],
                                    predictions[0]["scores"],
                                )
                                if score > 0.25
                            ],
                        },
                        output_path=output_path,
                        colors={"resnet": (0, 0, 255)},
                    )

                    print(
                        f"Saved bounding box image to {output_path} with {len(predictions[0]['boxes'])} detections."
                    )


if __name__ == "__main__":
    main()
