#!/usr/bin/env python3
"""
Train GoogLeNet (Inception v1) on GTSRB Traffic Sign Crops.

Records:
- Top-1 Classification Accuracy
- Normalized Confusion Matrix plot
- Per-class recall with special focus on Stop (14), Speed Limit (0-8), and Danger (11, 18-31) classes
"""

import argparse
import os
from pathlib import Path
import time
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import confusion_matrix, classification_report
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


class GTSRBCropDataset(Dataset):
    def __init__(self, csv_path: str, transform=None):
        self.df = pd.read_csv(csv_path)
        self.filepaths = self.df["filepath"].tolist()
        self.labels = self.df["class_id"].tolist()
        self.transform = transform

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        with Image.open(self.filepaths[idx]) as img:
            img = img.convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, int(self.labels[idx])


def build_googlenet(num_classes: int = 43, pretrained: bool = True) -> nn.Module:
    weights = models.GoogLeNet_Weights.DEFAULT if pretrained else None
    model = models.googlenet(weights=weights, aux_logits=False)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(in_features, num_classes)
    )
    return model


def train_googlenet(
    data_dir: str = "data/splits",
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 0.001,
    img_size: int = 64,
    device_str: str = "auto",
    checkpoint_path: str = "checkpoints/googlenet_gtsrb.pt",
):
    device = torch.device("cuda" if (device_str == "auto" and torch.cuda.is_available()) or device_str == "cuda" else "cpu")

    print(f"\n=======================================================")
    print(f" Training GoogLeNet on GTSRB Traffic Sign Crops")
    print(f" Device: {device} | Epochs: {epochs} | Batch Size: {batch_size}")
    print(f"=======================================================\n")

    train_csv = os.path.join(data_dir, "train.csv")
    val_csv = os.path.join(data_dir, "val.csv")
    test_csv = os.path.join(data_dir, "test.csv")

    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomRotation(degrees=12),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    eval_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = GTSRBCropDataset(train_csv, transform=train_tf)
    val_dataset = GTSRBCropDataset(val_csv, transform=eval_tf) if os.path.exists(val_csv) else None
    test_dataset = GTSRBCropDataset(test_csv, transform=eval_tf) if os.path.exists(test_csv) else None

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0) if val_dataset else None
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0) if test_dataset else None

    num_classes = max(len(set(train_dataset.labels)), 43)
    model = build_googlenet(num_classes=num_classes, pretrained=True).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
    best_acc = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for imgs, targets in train_loader:
            imgs, targets = imgs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

        train_loss = running_loss / total
        train_acc = correct / total

        val_acc = 0.0
        if val_loader:
            model.eval()
            v_correct, v_total = 0, 0
            with torch.no_grad():
                for imgs, targets in val_loader:
                    imgs, targets = imgs.to(device), targets.to(device)
                    outputs = model(imgs)
                    _, preds = torch.max(outputs, 1)
                    v_correct += (preds == targets).sum().item()
                    v_total += targets.size(0)
            val_acc = v_correct / v_total
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "num_classes": num_classes,
                    "val_acc": val_acc,
                    "epoch": epoch,
                }, checkpoint_path)
                flag = "[SAVED BEST]"
            else:
                flag = ""
            print(f"Epoch [{epoch:02d}/{epochs:02d}] Train Acc: {train_acc*100:5.2f}% | Val Acc: {val_acc*100:5.2f}% {flag}")
        else:
            print(f"Epoch [{epoch:02d}/{epochs:02d}] Train Acc: {train_acc*100:5.2f}%")

    print(f"\nTraining finished. Best Val Acc: {best_acc*100:.2f}%")

    # Evaluation on Test Set & Per-Class Recall + Confusion Matrix
    eval_target_loader = test_loader or val_loader
    if eval_target_loader and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        all_preds = []
        all_targets = []
        with torch.no_grad():
            for imgs, targets in eval_target_loader:
                imgs = imgs.to(device)
                outputs = model(imgs)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy().tolist())
                all_targets.extend(targets.numpy().tolist())

        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)

        overall_acc = (all_preds == all_targets).mean()
        print(f"\n=======================================================")
        print(f" GoogLeNet Evaluation Results on Test Split")
        print(f" Overall Accuracy: {overall_acc*100:.2f}% ({len(all_targets)} samples)")
        print(f"=======================================================")

        # Per-class recall
        cm = confusion_matrix(all_targets, all_preds, labels=list(range(num_classes)))
        per_class_recall = {}
        print("\nPer-Class Recall Summary:")
        for c in range(min(num_classes, 43)):
            total_c = cm[c, :].sum()
            correct_c = cm[c, c]
            recall_c = (correct_c / total_c * 100.0) if total_c > 0 else 0.0
            per_class_recall[c] = recall_c
            if total_c > 0:
                print(f"  Class {c:02d}: Recall = {recall_c:6.2f}% ({correct_c}/{total_c})")

        # Plot & Save Confusion Matrix
        plt.figure(figsize=(10, 8))
        cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-8)
        plt.imshow(cm_norm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title("GoogLeNet Normalized Confusion Matrix")
        plt.colorbar()
        plt.xlabel("Predicted Class")
        plt.ylabel("True Class")
        plt.tight_layout()

        cm_path = Path(checkpoint_path).parent / "googlenet_confusion_matrix.png"
        plt.savefig(cm_path)
        plt.close()
        print(f"\nConfusion matrix saved to: {cm_path}")

        # Save metrics JSON
        summary_path = Path(checkpoint_path).parent / "googlenet_summary.json"
        with open(summary_path, "w") as f:
            import json
            json.dump({
                "model": "GoogLeNet",
                "accuracy": float(overall_acc),
                "per_class_recall": {int(k): float(v) for k, v in per_class_recall.items()}
            }, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Train GoogLeNet on GTSRB Traffic Sign Crops")
    parser.add_argument("--data-dir", type=str, default="data/splits", help="Path to split manifests")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--img-size", type=int, default=64, help="Crop image size")
    parser.add_argument("--device", type=str, default="auto", help="Compute device (auto, cuda, cpu)")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/googlenet_gtsrb.pt", help="Checkpoint save path")
    args = parser.parse_args()

    train_googlenet(
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
