import torch
import os
from PIL import Image
from torchvision.transforms import functional as F


class YoloDataset(torch.utils.data.Dataset):
    def __init__(self, images_dir, labels_dir, classes, transforms=None):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.transforms = transforms
        self.classes = classes

        self.image_files = []

        # Only include image/label pairs with non-empty labels
        for fname in sorted(os.listdir(images_dir)):
            if not (fname.endswith('.jpg') or fname.endswith('.png')):
                continue

            label_path = os.path.join(
                labels_dir, os.path.splitext(fname)[0] + '.txt')

            if not os.path.exists(label_path):
                continue

            with open(label_path, 'r') as f:
                lines = f.readlines()

            if len(lines) == 0:
                continue  # skip files with no annotations

            self.image_files.append(fname)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, image_name)
        label_path = os.path.join(
            self.labels_dir, os.path.splitext(image_name)[0] + '.txt')

        # Load image
        img = Image.open(img_path).convert("RGB")
        width, height = img.size

        # Read YOLO label
        boxes = []
        labels = []
        with open(label_path, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) != 5:
                    continue  # skip malformed lines

                class_id = int(parts[0])
                x_center = float(parts[1]) * width
                y_center = float(parts[2]) * height
                w = float(parts[3]) * width
                h = float(parts[4]) * height

                # Convert to [x1, y1, x2, y2]
                x1 = x_center - w / 2
                y1 = y_center - h / 2
                x2 = x_center + w / 2
                y2 = y_center + h / 2

                boxes.append([x1, y1, x2, y2])
                # +1 because 0 is background in Faster R-CNN
                labels.append(class_id + 1)

        # Final safety check
        if len(boxes) == 0:
            raise ValueError(f"No valid annotations found for {img_path}")

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
        }

        if self.transforms:
            img = self.transforms(img)

        img = F.to_tensor(img)
        return img, target
