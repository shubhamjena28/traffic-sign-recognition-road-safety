#!/usr/bin/env python3
"""
Fine-tune Faster R-CNN (torchvision ResNet-50 FPN) on Traffic Sign Detection.

Target Classes:
- 0: Background
- 1: Stop sign
- 2: Speed limit sign
- 3: Danger/Hazard warning sign
"""

import argparse
import glob
import os
from pathlib import Path
import time
from typing import Dict, List, Tuple

from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


class RoadSignDetectionDataset(Dataset):
    """Loads full road images and bounding box annotations from labels txt."""

    def __init__(self, img_dir: str, lbl_dir: str, transform=None):
        self.img_paths = sorted(glob.glob(os.path.join(img_dir, "*.jpg")) + glob.glob(os.path.join(img_dir, "*.png")))
        self.lbl_dir = lbl_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = Image.open(img_path).convert("RGB")
        w, h = img.size

        lbl_filename = Path(img_path).stem + ".txt"
        lbl_path = os.path.join(self.lbl_dir, lbl_filename)

        boxes = []
        labels = []

        if os.path.exists(lbl_path):
            with open(lbl_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    cls_id, xc, yc, bw, bh = map(float, parts)
                    # Convert YOLO norm (xc, yc, bw, bh) to absolute (x1, y1, x2, y2)
                    x1 = max(0.0, (xc - bw / 2.0) * w)
                    y1 = max(0.0, (yc - bh / 2.0) * h)
                    x2 = min(float(w), (xc + bw / 2.0) * w)
                    y2 = min(float(h), (yc + bh / 2.0) * h)

                    if x2 > x1 and y2 > y1:
                        boxes.append([x1, y1, x2, y2])
                        # Shift by +1: 0 is background in torchvision Faster R-CNN
                        labels.append(int(cls_id) + 1)

        if not boxes:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx]),
        }

        # ToTensor transform
        img_tensor = transforms.functional.to_tensor(img)
        return img_tensor, target


def collate_fn(batch):
    return tuple(zip(*batch))


def build_faster_rcnn(num_classes: int = 4, pretrained: bool = True) -> nn.Module:
    """Builds Faster R-CNN ResNet-50 FPN with custom replacement predictor head."""
    weights = models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None
    model = models.detection.fasterrcnn_resnet50_fpn(weights=weights)

    # Replace box predictor
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def train_faster_rcnn(
    data_dir: str = "data",
    epochs: int = 5,
    batch_size: int = 4,
    lr: float = 0.0005,
    device_str: str = "auto",
    checkpoint_path: str = "checkpoints/faster_rcnn_resnet50.pt",
):
    device = torch.device("cuda" if (device_str == "auto" and torch.cuda.is_available()) or device_str == "cuda" else "cpu")

    print(f"\n=======================================================")
    print(f" Fine-Tuning Faster R-CNN (ResNet-50 FPN)")
    print(f" Device: {device} | Epochs: {epochs} | Batch Size: {batch_size}")
    print(f" Target Classes: 0: Background, 1: Stop, 2: Speed, 3: Danger")
    print(f"=======================================================\n")

    train_img_dir = os.path.join(data_dir, "images", "train")
    train_lbl_dir = os.path.join(data_dir, "labels", "train")
    val_img_dir = os.path.join(data_dir, "images", "val")
    val_lbl_dir = os.path.join(data_dir, "labels", "val")

    train_dataset = RoadSignDetectionDataset(train_img_dir, train_lbl_dir)
    val_dataset = RoadSignDetectionDataset(val_img_dir, val_lbl_dir) if os.path.exists(val_img_dir) else None

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn) if val_dataset else None

    model = build_faster_rcnn(num_classes=4, pretrained=True).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(params, lr=lr, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.33)

    Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")

    t_start = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        batch_count = 0

        for images, targets in train_loader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            epoch_loss += losses.item()
            batch_count += 1

        lr_scheduler.step()
        avg_loss = epoch_loss / max(batch_count, 1)

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "loss": avg_loss,
            }, checkpoint_path)
            flag = "[SAVED CHECKPOINT]"
        else:
            flag = ""

        print(f"Epoch [{epoch:02d}/{epochs:02d}] Total Detection Loss: {avg_loss:.4f} {flag}")

    print(f"\nFaster R-CNN training finished in {time.time() - t_start:.2f}s. Best Loss: {best_loss:.4f}")

    # Benchmark test inference
    model.eval()
    sample_images = glob.glob(os.path.join(data_dir, "images", "val", "*.jpg"))
    if not sample_images:
        sample_images = glob.glob(os.path.join(data_dir, "images", "train", "*.jpg"))

    if sample_images:
        with torch.no_grad():
            t0 = time.perf_counter()
            test_img = transforms.functional.to_tensor(Image.open(sample_images[0]).convert("RGB")).to(device)
            preds = model([test_img])[0]
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            n_boxes = len(preds["boxes"])
            print(f"Sample Inference: Detected {n_boxes} candidate boxes in {elapsed_ms:.2f} ms")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Faster R-CNN on Road Sign Detection")
    parser.add_argument("--data-dir", type=str, default="data", help="Root data folder")
    parser.add_argument("--epochs", type=int, default=5, help="Epochs to train")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.0005, help="Learning rate")
    parser.add_argument("--device", type=str, default="auto", help="Compute device")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/faster_rcnn_resnet50.pt", help="Checkpoint output")
    args = parser.parse_args()

    train_faster_rcnn(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device_str=args.device,
        checkpoint_path=args.checkpoint,
    )


if __name__ == "__main__":
    main()
