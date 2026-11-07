#!/usr/bin/env python3
"""
Semantic Segmentation on BDD100K Road Scene / Drivable Area Subset.

Models:
- DeepLabV3 / FCN on Drivable Area & Road Signs
- Reports Mean IoU (mIoU) and Pixel Accuracy
- Visualizes predictions saved to checkpoints/segmentation_prediction.png
"""

import argparse
import os
from pathlib import Path
import time
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


class SyntheticOrBDD100KSegDataset(Dataset):
    """BDD100K Semantic Segmentation Dataset loader with graceful synthetic fallback."""

    def __init__(self, data_dir: str = "data/bdd100k", num_samples: int = 50, img_size: int = 128):
        self.img_size = img_size
        self.samples = []

        images_dir = Path(data_dir) / "images"
        masks_dir = Path(data_dir) / "labels"

        if images_dir.exists() and masks_dir.exists():
            img_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
            for p in img_files[:num_samples]:
                m_path = masks_dir / (p.stem + ".png")
                if m_path.exists():
                    self.samples.append((str(p), str(m_path)))

        self.use_synthetic = len(self.samples) == 0
        self.num_samples = num_samples if self.use_synthetic else len(self.samples)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if not self.use_synthetic:
            img_path, mask_path = self.samples[idx]
            img = Image.open(img_path).convert("RGB").resize((self.img_size, self.img_size))
            mask = Image.open(mask_path).convert("L").resize((self.img_size, self.img_size), Image.NEAREST)
            img_tensor = transforms.functional.to_tensor(img)
            mask_arr = np.array(mask, dtype=np.int64)
            mask_arr = np.clip(mask_arr, 0, 2)
            return img_tensor, torch.from_numpy(mask_arr)

        # Generate realistic road scene synthetic segmentation
        # Classes: 0: Background/Sky/Surroundings, 1: Drivable Road, 2: Road Signs/Markings
        img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        mask = np.zeros((self.img_size, self.img_size), dtype=np.int64)

        # Sky/Surroundings
        img[: self.img_size // 2, :] = [180, 150, 120]  # Sky/Horizon

        # Drivable road triangle/trapezoid
        road_pts = np.array([
            [self.img_size // 4, self.img_size // 2],
            [3 * self.img_size // 4, self.img_size // 2],
            [self.img_size, self.img_size],
            [0, self.img_size]
        ], dtype=np.int32)
        import cv2
        cv2.fillPoly(img, [road_pts], (80, 80, 80))
        cv2.fillPoly(mask, [road_pts], 1)

        # Road Sign / Lane Marker
        cv2.circle(img, (self.img_size // 2, 2 * self.img_size // 3), 8, (0, 0, 255), -1)
        cv2.circle(mask, (self.img_size // 2, 2 * self.img_size // 3), 8, 2)

        img_tensor = transforms.functional.to_tensor(img)
        return img_tensor, torch.from_numpy(mask)


def compute_miou_and_pixel_acc(preds: torch.Tensor, targets: torch.Tensor, num_classes: int = 3) -> Tuple[float, float]:
    """Computes Mean Intersection over Union (mIoU) and Pixel Accuracy."""
    preds = preds.view(-1)
    targets = targets.view(-1)

    correct_pixels = (preds == targets).sum().item()
    total_pixels = targets.numel()
    pixel_acc = correct_pixels / max(total_pixels, 1)

    ious = []
    for cls_id in range(num_classes):
        pred_inds = preds == cls_id
        target_inds = targets == cls_id
        intersection = (pred_inds & target_inds).sum().item()
        union = (pred_inds | target_inds).sum().item()
        if union > 0:
            ious.append(intersection / union)

    miou = np.mean(ious) if ious else 0.0
    return float(miou), float(pixel_acc)


def build_segmentation_model(num_classes: int = 3) -> nn.Module:
    """Builds a lightweight DeepLabV3-MobileNetV3 segmentation model."""
    try:
        model = models.segmentation.deeplabv3_mobilenet_v3_large(weights=None, num_classes=num_classes)
    except Exception:
        # Fallback to FCN
        model = models.segmentation.fcn_resnet50(weights=None, num_classes=num_classes)
    return model


def train_segmentation(
    epochs: int = 5,
    batch_size: int = 8,
    lr: float = 0.001,
    device_str: str = "auto",
    checkpoint_path: str = "checkpoints/segmentation_bdd100k.pt",
):
    device = torch.device("cuda" if (device_str == "auto" and torch.cuda.is_available()) or device_str == "cuda" else "cpu")

    print(f"\n=======================================================")
    print(f" Training Road Scene Semantic Segmentation (BDD100K)")
    print(f" Device: {device} | Epochs: {epochs} | Batch Size: {batch_size}")
    print(f" Semantic Classes: 0: Background, 1: Drivable Road, 2: Road Signs")
    print(f"=======================================================\n")

    dataset = SyntheticOrBDD100KSegDataset(num_samples=40, img_size=128)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = build_segmentation_model(num_classes=3).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        all_preds = []
        all_targets = []

        for imgs, masks in loader:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            out = model(imgs)["out"]
            loss = criterion(out, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            preds = torch.argmax(out, dim=1)
            all_preds.append(preds.cpu())
            all_targets.append(masks.cpu())

        avg_loss = running_loss / len(dataset)
        preds_all = torch.cat(all_preds, dim=0)
        targets_all = torch.cat(all_targets, dim=0)
        miou, pixel_acc = compute_miou_and_pixel_acc(preds_all, targets_all, num_classes=3)

        print(f"Epoch [{epoch:02d}/{epochs:02d}] Loss: {avg_loss:.4f} | Mean IoU: {miou*100:5.2f}% | Pixel Acc: {pixel_acc*100:5.2f}%")

    print(f"\nSegmentation training complete in {time.time() - t_start:.2f}s.")
    print(f"Final Performance: Mean IoU = {miou*100:.2f}%, Pixel Accuracy = {pixel_acc*100:.2f}%")

    # Save model
    torch.save({
        "model_state_dict": model.state_dict(),
        "miou": miou,
        "pixel_acc": pixel_acc,
    }, checkpoint_path)

    # Visualization
    model.eval()
    with torch.no_grad():
        sample_img, sample_mask = dataset[0]
        out = model(sample_img.unsqueeze(0).to(device))["out"]
        pred_mask = torch.argmax(out, dim=1)[0].cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(10, 4))
    axes[0].imshow(np.transpose(sample_img.numpy(), (1, 2, 0)))
    axes[0].set_title("Input Road Scene")
    axes[0].axis("off")

    axes[1].imshow(sample_mask.numpy(), cmap="viridis")
    axes[1].set_title("Ground-Truth Mask")
    axes[1].axis("off")

    axes[2].imshow(pred_mask, cmap="viridis")
    axes[2].set_title(f"Predicted (mIoU: {miou*100:.1f}%)")
    axes[2].axis("off")

    vis_path = Path(checkpoint_path).parent / "segmentation_prediction.png"
    plt.tight_layout()
    plt.savefig(vis_path)
    plt.close()
    print(f"Segmentation visualization saved to: {vis_path}")


def main():
    parser = argparse.ArgumentParser(description="Train Semantic Segmentation Model on BDD100K subset")
    parser.add_argument("--epochs", type=int, default=5, help="Epochs to train")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--device", type=str, default="auto", help="Compute device")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/segmentation_bdd100k.pt", help="Checkpoint path")
    args = parser.parse_args()

    train_segmentation(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device_str=args.device,
        checkpoint_path=args.checkpoint,
    )


if __name__ == "__main__":
    main()
