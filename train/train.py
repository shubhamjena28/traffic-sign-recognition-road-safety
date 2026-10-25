"""Traffic Sign Recognition & Detection Model Training Script.

Supports both:
1. Object Detection using Ultralytics YOLO (detecting bounding boxes and classes)
2. Custom Deep CNN for multi-class traffic sign image classification
"""

import argparse
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


class TrafficSignCNN(nn.Module):
    """Custom Convolutional Neural Network for traffic sign image classification."""

    def __init__(self, num_classes: int = 43):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.25),
            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.25),
            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Dropout2d(0.3),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


def get_data_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((48, 48)),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    return train_transform, val_transform


def train_yolo(data_yaml: str, model_name: str = "yolov8n.pt", epochs: int = 30, imgsz: int = 640):
    """Trains a YOLO object detection model on traffic sign datasets."""
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] ultralytics is not installed. Please run: pip install -r requirements.txt")
        return

    print(f"Loading YOLO model '{model_name}' for traffic sign detection...")
    model = YOLO(model_name)
    results = model.train(data=data_yaml, epochs=epochs, imgsz=imgsz)
    print("YOLO Training complete. Best weights saved in runs/detect/train/weights/best.pt")
    return results


def train_cnn(data_dir: str, output_dir: str, epochs: int, batch_size: int, lr: float, num_classes: int):
    """Trains custom CNN on class-folder dataset."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training CNN on device: {device}")

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    data_path = Path(data_dir)
    if not data_path.exists() or not any(data_path.iterdir()):
        print(f"[WARNING] Dataset directory '{data_path}' is empty or does not exist.")
        print("Please place the GTSRB dataset into data/processed/ or specify --data-dir.")
        return

    train_tf, val_tf = get_data_transforms()
    full_dataset = datasets.ImageFolder(root=str(data_path), transform=train_tf)
    val_size = int(0.2 * len(full_dataset))
    train_size = len(full_dataset) - val_size

    train_set, val_set = random_split(full_dataset, [train_size, val_size])
    val_set.dataset.transform = val_tf

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2)

    model = TrafficSignCNN(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0.0
    print(f"Starting CNN training for {epochs} epochs...")

    for epoch in range(1, epochs + 1):
        start_time = time.time()
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        train_loss = running_loss / max(total, 1)
        train_acc = 100.0 * correct / max(total, 1)

        # Validation
        model.eval()
        v_loss, v_correct, v_total = 0.0, 0, 0
        with torch.no_grad():
            for images, targets in val_loader:
                images, targets = images.to(device), targets.to(device)
                outputs = model(images)
                loss = criterion(outputs, targets)
                v_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                v_total += targets.size(0)
                v_correct += predicted.eq(targets).sum().item()

        val_loss = v_loss / max(v_total, 1)
        val_acc = 100.0 * v_correct / max(v_total, 1)
        scheduler.step()

        elapsed = time.time() - start_time
        print(f"Epoch [{epoch:02d}/{epochs:02d}] ({elapsed:.1f}s) | "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_path = out_path / "best_model.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_acc": val_acc,
                "num_classes": num_classes,
            }, best_model_path)
            print(f"  --> Saved new best checkpoint: {best_model_path} ({val_acc:.2f}%)")

    print(f"\nTraining completed. Best Validation Accuracy: {best_val_acc:.2f}%")


def main():
    parser = argparse.ArgumentParser(description="Train Traffic Sign Recognition & Detection")
    parser.add_argument("--mode", choices=["cnn", "yolo"], default="cnn", help="Model type: 'cnn' (classifier) or 'yolo' (detector)")
    parser.add_argument("--data-dir", type=str, default="../data/processed", help="Path to processed image folders (for CNN)")
    parser.add_argument("--data-yaml", type=str, default="data.yaml", help="Path to dataset YAML (for YOLO)")
    parser.add_argument("--output-dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--num-classes", type=int, default=43, help="Number of classes")
    args = parser.parse_args()

    if args.mode == "yolo":
        train_yolo(args.data_yaml, epochs=args.epochs)
    else:
        train_cnn(args.data_dir, args.output_dir, args.epochs, args.batch_size, args.lr, args.num_classes)


if __name__ == "__main__":
    main()
