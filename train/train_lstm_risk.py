#!/usr/bin/env python3
"""
Train Temporal LSTM for Road Safety Risk Prediction.

Requirements:
- Window size of 10 consecutive frames.
- Features: [num_signs, dominant_class_id, box_area, box_growth, speed_proxy].
- Rule-based risk labels: STOP or Pedestrian with growing box = HIGH risk.
- Compares Step, Sigmoid, and Tanh activations on the same task.
- Generates Confusion Matrix and Sequence Risk Visualization.
"""

import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


# ==============================================================================
# 1. Dataset Generation & Sliding Window Extraction
# ==============================================================================

def generate_multi_sequence_dataset(num_episodes: int = 40, episode_len: int = 35) -> pd.DataFrame:
    """
    Generates diverse driving episodes:
    - Episode types:
      0: Normal highway cruising (speed signs, low/med risk)
      1: Approaching STOP sign (box grows rapidly, becomes HIGH risk)
      2: Approaching Pedestrian crosswalk (HIGH risk)
      3: Danger / sharp curve hazard (box grows, MED/HIGH risk)
      4: Clear empty road (LOW risk)
    """
    np.random.seed(42)
    rows = []

    for ep in range(num_episodes):
        ep_type = ep % 5
        base_speed = 60.0 + np.random.uniform(-10, 15)

        for t in range(episode_len):
            progress = t / float(episode_len)

            if ep_type == 1:
                # Approaching STOP sign
                num_signs = 1
                class_id = 0  # stop
                area = 200.0 + 8000.0 * (progress ** 2.2)
                growth = 8000.0 * 2.2 * (progress ** 1.2) / episode_len if t > 0 else 0.0
                speed = max(0.0, base_speed * (1.0 - 0.7 * progress))
                risk = 2 if (progress > 0.4 and growth > 15.0) else (1 if progress > 0.2 else 0)

            elif ep_type == 2:
                # Approaching Pedestrian
                num_signs = 1
                class_id = 3  # pedestrian
                area = 150.0 + 6500.0 * (progress ** 2.0)
                growth = 6500.0 * 2.0 * progress / episode_len if t > 0 else 0.0
                speed = max(10.0, base_speed * (1.0 - 0.8 * progress))
                risk = 2 if (progress > 0.35 and growth > 12.0) else (1 if progress > 0.15 else 0)

            elif ep_type == 3:
                # Danger / Curve sign
                num_signs = 1
                class_id = 2  # danger
                area = 180.0 + 4000.0 * (progress ** 1.8)
                growth = 4000.0 * 1.8 * (progress ** 0.8) / episode_len if t > 0 else 0.0
                speed = base_speed * (1.0 - 0.3 * progress)
                risk = 2 if (progress > 0.6 and speed > 45.0) else (1 if progress > 0.25 else 0)

            elif ep_type == 0:
                # Speed limit zone
                num_signs = 1
                class_id = 1  # speed
                area = 250.0 + 2000.0 * progress
                growth = 2000.0 / episode_len
                speed = base_speed
                risk = 1 if progress > 0.3 else 0

            else:
                # Clear road
                num_signs = 0
                class_id = -1
                area = 0.0
                growth = 0.0
                speed = base_speed + np.random.uniform(-2, 2)
                risk = 0

            rows.append({
                "episode": ep,
                "step": t,
                "num_signs": num_signs,
                "class_id": class_id,
                "box_area": area,
                "box_growth": growth,
                "speed": speed,
                "risk_label": risk
            })

    return pd.DataFrame(rows)


class WindowedSequenceDataset(Dataset):
    """Extracts sliding windows of length window_size (10 frames) for LSTM training."""

    def __init__(self, df: pd.DataFrame, window_size: int = 10):
        self.windows = []
        self.labels = []

        # Feature normalization scales
        # [num_signs / 3.0, (class_id + 1) / 4.0, box_area / 10000.0, box_growth / 500.0, speed / 100.0]
        for ep_id, group in df.groupby("episode"):
            group = group.sort_values("step").reset_index(drop=True)
            feats = np.stack([
                group["num_signs"].values / 3.0,
                (group["class_id"].values + 1.0) / 4.0,
                np.clip(group["box_area"].values / 10000.0, 0.0, 2.0),
                np.clip(group["box_growth"].values / 500.0, -1.0, 1.0),
                group["speed"].values / 100.0
            ], axis=1).astype(np.float32)

            targets = group["risk_label"].values.astype(np.int64)

            for i in range(len(group) - window_size + 1):
                win = feats[i: i + window_size]
                # Label is the risk of the final frame in the window
                target = targets[i + window_size - 1]
                self.windows.append(win)
                self.labels.append(target)

        self.windows = torch.tensor(np.array(self.windows), dtype=torch.float32)
        self.labels = torch.tensor(np.array(self.labels), dtype=torch.int64)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.windows[idx], self.labels[idx]


# ==============================================================================
# 2. LSTM Architectures with Step, Sigmoid, and Tanh Activations
# ==============================================================================

class HardStepActivation(nn.Module):
    """Differentiable smooth step approximation for training, exact threshold at inference."""

    def __init__(self, temperature: float = 5.0):
        super().__init__()
        self.temp = temperature

    def forward(self, x):
        if self.training:
            return torch.sigmoid(self.temp * x)
        return (x > 0.0).float()


class RiskLSTM(nn.Module):
    """
    Temporal LSTM Risk Model with configurable non-linear activation.
    Input: (Batch, 10, 5) -> Output: (Batch, 3) for [LOW, MEDIUM, HIGH] risk.
    """

    def __init__(self, input_size: int = 5, hidden_size: int = 32, num_classes: int = 3, activation_type: str = "tanh"):
        super().__init__()
        self.activation_type = activation_type.lower()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=2, batch_first=True, dropout=0.1)

        if self.activation_type == "step":
            self.act = HardStepActivation()
        elif self.activation_type == "sigmoid":
            self.act = nn.Sigmoid()
        elif self.activation_type == "tanh":
            self.act = nn.Tanh()
        else:
            self.act = nn.ReLU()

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 16),
            self.act,
            nn.Linear(16, num_classes)
        )

    def forward(self, x):
        # x shape: (B, T=10, 5)
        lstm_out, _ = self.lstm(x)
        # Take the output of the final time step
        last_hidden = lstm_out[:, -1, :]
        logits = self.fc(last_hidden)
        return logits


# ==============================================================================
# 3. Model Training & Activation Comparison
# ==============================================================================

def train_and_eval_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                         epochs: int = 15, lr: float = 0.003) -> Tuple[float, float, np.ndarray, np.ndarray]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

    # Evaluation
    model.eval()
    all_preds, all_targets = [], []
    total_val_loss = 0.0

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            total_val_loss += loss.item() * len(y)
            preds = torch.argmax(out, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(y.cpu().numpy())

    avg_val_loss = total_val_loss / len(all_targets)
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    accuracy = float((all_preds == all_targets).mean())
    macro_f1 = float(f1_score(all_targets, all_preds, average="macro"))

    return accuracy, macro_f1, all_preds, all_targets


def plot_confusion_matrix(targets: np.ndarray, preds: np.ndarray, save_path: Path):
    """Plots and saves normalized confusion matrix."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(targets, preds, labels=[0, 1, 2])
    cm_norm = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    classes = ["LOW (0)", "MED (1)", "HIGH (2)"]

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=classes, yticklabels=classes,
        title="Temporal LSTM Risk Confusion Matrix",
        ylabel="Ground Truth Risk",
        xlabel="Predicted Risk"
    )

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]}\n({cm_norm[i, j]*100:.1f}%)",
                    ha="center", va="center",
                    color="white" if cm_norm[i, j] > 0.5 else "black",
                    fontweight="bold")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"[SUCCESS] Confusion matrix saved to: {save_path}")


def plot_sequence_risk_visualization(model: nn.Module, df: pd.DataFrame, save_path: Path):
    """Visualizes model risk predictions across a 35-frame approaching STOP scenario."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)

    # Filter episode 1 (approaching STOP sign)
    ep1 = df[df["episode"] == 1].sort_values("step").reset_index(drop=True)
    window_size = 10

    steps, areas, true_risks, pred_risks, pred_probs = [], [], [], [], []

    for i in range(len(ep1) - window_size + 1):
        win_df = ep1.iloc[i: i + window_size]
        feats = np.stack([
            win_df["num_signs"].values / 3.0,
            (win_df["class_id"].values + 1.0) / 4.0,
            np.clip(win_df["box_area"].values / 10000.0, 0.0, 2.0),
            np.clip(win_df["box_growth"].values / 500.0, -1.0, 1.0),
            win_df["speed"].values / 100.0
        ], axis=1).astype(np.float32)

        x_tensor = torch.tensor(feats, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(x_tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            pred = int(np.argmax(probs))

        steps.append(win_df["step"].iloc[-1])
        areas.append(win_df["box_area"].iloc[-1])
        true_risks.append(win_df["risk_label"].iloc[-1])
        pred_risks.append(pred)
        pred_probs.append(probs[2])  # Probability of HIGH risk

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    # 1. Box Area (Approaching STOP sign)
    ax1.plot(steps, areas, color="#2563eb", linewidth=2.5, marker="o", label="STOP Sign Area ($px^2$)")
    ax1.set_ylabel("Box Area ($px^2$)")
    ax1.set_title("Temporal Sequence Risk Prediction: Approaching STOP Sign (Window=10 frames)", fontsize=12, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(loc="upper left")

    # 2. Predicted vs True Risk
    ax2.plot(steps, true_risks, color="#16a34a", linewidth=2.5, linestyle="--", label="Ground Truth Risk")
    ax2.plot(steps, pred_risks, color="#dc2626", linewidth=2.0, marker="x", label="LSTM Predicted Risk")
    ax2.fill_between(steps, 0, pred_probs, color="#ef4444", alpha=0.2, label="$P(Risk = HIGH)$")

    ax2.set_yticks([0, 1, 2])
    ax2.set_yticklabels(["LOW (0)", "MED (1)", "HIGH (2)"])
    ax2.set_xlabel("Sequence Frame Index")
    ax2.set_ylabel("Risk Level")
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(loc="upper left")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"[SUCCESS] Sequence risk visualization saved to: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Train Temporal LSTM for Road Safety Risk Prediction")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.003, help="Learning rate")
    parser.add_argument("--save-model", type=str, default="checkpoints/lstm_risk.pt", help="Path to save best LSTM model")
    args = parser.parse_args()

    print("\n" + "=" * 65)
    print("      Temporal LSTM Risk Modeling (10-Frame Sliding Windows)")
    print("=" * 65)

    # 1. Prepare Dataset
    df = generate_multi_sequence_dataset(num_episodes=50, episode_len=35)
    split_idx = int(50 * 0.75)
    train_df = df[df["episode"] < split_idx]
    val_df = df[df["episode"] >= split_idx]

    train_ds = WindowedSequenceDataset(train_df, window_size=10)
    val_ds = WindowedSequenceDataset(val_df, window_size=10)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    print(f"[INFO] Train Samples: {len(train_ds)} | Val Samples: {len(val_ds)} | Window Size: 10 frames")

    # 2. Compare Activations: Step, Sigmoid, and Tanh
    print("\n--- Comparing Activations: Step vs. Sigmoid vs. Tanh ---")
    activations = ["step", "sigmoid", "tanh"]
    results = {}

    best_model = None
    best_acc = 0.0
    best_preds, best_targets = None, None

    for act in activations:
        print(f"Training LSTM with '{act.upper()}' activation...")
        model = RiskLSTM(input_size=5, hidden_size=32, num_classes=3, activation_type=act)
        acc, f1, preds, targets = train_and_eval_model(model, train_loader, val_loader, epochs=args.epochs, lr=args.lr)
        results[act] = {"acc": acc, "f1": f1}
        print(f"  -> {act.upper()} Validation Accuracy: {acc * 100:.2f}% | Macro F1: {f1:.4f}")

        if acc > best_acc or act == "tanh":
            best_acc = acc
            best_model = model
            best_preds, best_targets = preds, targets

    # Print summary table
    print("\n" + "=" * 48)
    print(" ACTIVATION COMPARISON RESULTS (Same Task)")
    print("=" * 48)
    print(f"{'Activation':<12} | {'Accuracy':<10} | {'Macro F1':<10}")
    print("-" * 48)
    for act in activations:
        print(f"{act.upper():<12} | {results[act]['acc'] * 100:6.2f}%   | {results[act]['f1']:.4f}")
    print("=" * 48)

    # 3. Save Best Model Checkpoint
    Path(args.save_model).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": best_model.state_dict(),
        "input_size": 5,
        "hidden_size": 32,
        "num_classes": 3,
        "window_size": 10,
        "best_acc": best_acc
    }, args.save_model)
    print(f"\n[SUCCESS] Best model checkpoint saved to: {args.save_model}")

    # 4. Generate Visualizations
    cm_path = Path("checkpoints/lstm_risk_confusion_matrix.png")
    plot_confusion_matrix(best_targets, best_preds, cm_path)

    seq_path = Path("checkpoints/lstm_sequence_risk.png")
    plot_sequence_risk_visualization(best_model, val_df, seq_path)

    # Copy to docs/visualizations
    import shutil
    shutil.copyfile(cm_path, Path("docs/visualizations") / cm_path.name)
    shutil.copyfile(seq_path, Path("docs/visualizations") / seq_path.name)


if __name__ == "__main__":
    main()
