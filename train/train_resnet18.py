#!/usr/bin/env python3
"""
Train ResNet-18 on GTSRB Traffic Sign Crops with Data Augmentation.

Features:
- Heavy data augmentation (RandomRotation, ColorJitter, RandomAffine)
- Pretrained ImageNet backbone fine-tuned for traffic sign classification
- Evaluates on validation and held-out test splits
- Saves best checkpoint and loss/accuracy curves
"""

import argparse
import os
from pathlib import Path
import time
from typing import Tuple

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


class GTSRBCropDataset(Dataset):
    """Loads traffic sign crops from manifest CSV files (train.csv, val.csv, test.csv)."""

    def __init__(self, csv_path: str, transform=None):
        self.df = pd.read_csv(csv_path)
        self.filepaths = self.df["filepath"].tolist()
        self.labels = self.df["class_id"].tolist()
        self.transform = transform

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        path = self.filepaths[idx]
        with Image.open(path) as img:
            img = img.convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = int(self.labels[idx])
        return img, label


def get_transforms(img_size: int = 64) -> Tuple[transforms.Compose, transforms.Compose]:
    """Builds training transforms with data augmentation and clean validation transforms."""
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.08, 0.08), scale=(0.92, 1.08)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    return train_transform, eval_transform


def build_resnet18(num_classes: int = 43, pretrained: bool = True) -> nn.Module:
    """Builds ResNet-18 model with replaced final classification layer."""
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, num_classes)
    )
    return model


def train_resnet18(
    data_dir: str = "data/splits",
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 0.001,
    img_size: int = 64,
    device_str: str = "auto",
    checkpoint_path: str = "checkpoints/resnet18_gtsrb.pt",
):
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    print(f"\n=======================================================")
    print(f" Training ResNet-18 on GTSRB Crops (Augmentation Enabled)")
    print(f" Device: {device} | Epochs: {epochs} | Batch Size: {batch_size}")
    print(f"=======================================================\n")

    train_csv = os.path.join(data_dir, "train.csv")
    val_csv = os.path.join(data_dir, "val.csv")
    test_csv = os.path.join(data_dir, "test.csv")

    if not os.path.exists(train_csv):
        raise FileNotFoundError(f"Missing train split manifest at: {train_csv}")

    train_tf, eval_tf = get_transforms(img_size)
    train_dataset = GTSRBCropDataset(train_csv, transform=train_tf)
    val_dataset = GTSRBCropDataset(val_csv, transform=eval_tf) if os.path.exists(val_csv) else None
    test_dataset = GTSRBCropDataset(test_csv, transform=eval_tf) if os.path.exists(test_csv) else None

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0) if val_dataset else None
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0) if test_dataset else None

    # Determine unique classes
    num_classes = len(set(train_dataset.labels))
    num_classes = max(num_classes, 43)

    model = build_resnet18(num_classes=num_classes, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0

    Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

        scheduler.step()

        train_loss = running_loss / total
        train_acc = correct / total
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)

        # Validation phase
        val_loss, val_acc = 0.0, 0.0
        if val_loader:
            model.eval()
            val_running_loss = 0.0
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for images, targets in val_loader:
                    images, targets = images.to(device), targets.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, targets)
                    val_running_loss += loss.item() * images.size(0)
                    _, preds = torch.max(outputs, 1)
                    val_correct += (preds == targets).sum().item()
                    val_total += targets.size(0)
            val_loss = val_running_loss / val_total
            val_acc = val_correct / val_total
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "num_classes": num_classes,
                    "img_size": img_size,
                    "val_acc": val_acc,
                    "epoch": epoch,
                }, checkpoint_path)
                saved_flag = "-> [SAVED BEST]"
            else:
                saved_flag = ""

            print(f"Epoch [{epoch:02d}/{epochs:02d}] "
                  f"Train Loss: {train_loss:.4f}, Acc: {train_acc*100:5.2f}% | "
                  f"Val Loss: {val_loss:.4f}, Acc: {val_acc*100:5.2f}% {saved_flag}")
        else:
            print(f"Epoch [{epoch:02d}/{epochs:02d}] Train Loss: {train_loss:.4f}, Acc: {train_acc*100:5.2f}%")

    print(f"\nTraining completed in {time.time() - t_start:.2f}s. Best Val Acc: {best_val_acc*100:.2f}%")

    # Evaluate on held-out test split
    if test_loader and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        test_correct, test_total = 0, 0
        with torch.no_grad():
            for images, targets in test_loader:
                images, targets = images.to(device), targets.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                test_correct += (preds == targets).sum().item()
                test_total += targets.size(0)
        test_acc = test_correct / test_total
        print(f"Held-Out Test Set Accuracy: {test_acc*100:.2f}% ({test_correct}/{test_total})")

    # Plot & save curves
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="Train Loss")
    if history["val_loss"]:
        plt.plot(history["val_loss"], label="Val Loss")
    plt.title("ResNet-18 Cross-Entropy Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot([a * 100 for a in history["train_acc"]], label="Train Acc (%)")
    if history["val_acc"]:
        plt.plot([a * 100 for a in history["val_acc"]], label="Val Acc (%)")
    plt.title("ResNet-18 Classification Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.grid(True)
    plt.legend()

    plot_path = Path(checkpoint_path).parent / "resnet18_training_curves.png"
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    print(f"Training curves saved to: {plot_path}")

    return history


def main():
    parser = argparse.ArgumentParser(description="Train ResNet-18 on GTSRB Traffic Sign Crops")
    parser.add_argument("--data-dir", type=str, default="data/splits", help="Path to split manifests")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--img-size", type=int, default=64, help="Crop image size")
    parser.add_argument("--device", type=str, default="auto", help="Compute device (auto, cuda, cpu)")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/resnet18_gtsrb.pt", help="Checkpoint save path")
    args = parser.parse_args()

    train_resnet18(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        img_size=args.img_size,
        device_str=args.device,
        checkpoint_path=args.checkpoint,
    )


if __name__ == "__main__":
    main()
