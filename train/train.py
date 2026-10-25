"""Traffic Sign Recognition Model Training Script.

Trains a Convolutional Neural Network (CNN) on traffic sign datasets
such as the German Traffic Sign Recognition Benchmark (GTSRB).
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
    """Custom Convolutional Neural Network for 32x32 to 64x64 traffic sign images."""

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


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, targets in loader:
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

    epoch_loss = running_loss / max(total, 1)
    epoch_acc = 100.0 * correct / max(total, 1)
    return epoch_loss, epoch_acc


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    epoch_loss = running_loss / max(total, 1)
    epoch_acc = 100.0 * correct / max(total, 1)
    return epoch_loss, epoch_acc


def main():
    parser = argparse.ArgumentParser(description="Train Traffic Sign Recognition CNN")
    parser.add_argument("--data-dir", type=str, default="../data/processed", help="Path to processed dataset")
    parser.add_argument("--output-dir", type=str, default="checkpoints", help="Directory to save model checkpoints")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument("--num-classes", type=int, default=43, help="Number of traffic sign classes")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_tf, val_tf = get_data_transforms()
    data_path = Path(args.data_dir)

    if not data_path.exists() or not any(data_path.iterdir()):
        print(f"[WARNING] Dataset directory '{data_path}' is empty or does not exist.")
        print("Please place the GTSRB dataset into data/processed/ or specify --data-dir.")
        print("Example: python train.py --data-dir ../data/processed --epochs 20")
        sys.exit(0)

    full_dataset = datasets.ImageFolder(root=str(data_path), transform=train_tf)
    val_size = int(0.2 * len(full_dataset))
    train_size = len(full_dataset) - val_size

    train_set, val_set = random_split(full_dataset, [train_size, val_size])
    val_set.dataset.transform = val_tf

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = TrafficSignCNN(num_classes=args.num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_acc = 0.0
    print(f"Starting training for {args.epochs} epochs...")

    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()
        elapsed = time.time() - start_time

        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] ({elapsed:.1f}s) | "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_path = output_dir / "best_model.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_acc": val_acc,
                "num_classes": args.num_classes,
            }, best_model_path)
            print(f"  --> Saved new best model checkpoint to {best_model_path} (Acc: {val_acc:.2f}%)")

    print(f"\nTraining completed. Best Validation Accuracy: {best_val_acc:.2f}%")


if __name__ == "__main__":
    main()
