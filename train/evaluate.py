"""Evaluation script for trained Traffic Sign Recognition models.

Computes classification metrics including Accuracy, Precision, Recall,
F1-Score, and generates Confusion Matrix visualization.
"""

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Import CNN definition from train
sys.path.append(str(Path(__file__).parent))
from train import TrafficSignCNN, get_data_transforms


def evaluate_model(checkpoint_path: str, data_dir: str, batch_size: int = 64, report_file: str = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on device: {device}")

    ckpt_file = Path(checkpoint_path)
    if not ckpt_file.exists():
        print(f"[ERROR] Checkpoint file '{checkpoint_path}' does not exist.")
        return

    data_path = Path(data_dir)
    if not data_path.exists() or not any(data_path.iterdir()):
        print(f"[WARNING] Test dataset path '{data_path}' is empty or not found.")
        print("Please provide a valid test dataset to evaluate.")
        return

    _, val_transform = get_data_transforms()
    test_dataset = datasets.ImageFolder(root=str(data_path), transform=val_transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    checkpoint = torch.load(ckpt_file, map_location=device)
    num_classes = checkpoint.get("num_classes", len(test_dataset.classes))

    model = TrafficSignCNN(num_classes=num_classes).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    all_preds = []
    all_targets = []

    print(f"Evaluating {len(test_dataset)} test samples...")
    with torch.no_grad():
        for images, targets in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = outputs.max(1)

            all_preds.extend(preds.cpu().tolist())
            all_targets.extend(targets.tolist())

    correct = sum(p == t for p, t in zip(all_preds, all_targets))
    accuracy = 100.0 * correct / max(len(all_targets), 1)

    print("\n" + "=" * 50)
    print(f"Evaluation Results for: {ckpt_file.name}")
    print("=" * 50)
    print(f"Total Samples Evaluated: {len(all_targets)}")
    print(f"Correct Predictions:     {correct}")
    print(f"Overall Accuracy:        {accuracy:.2f}%")
    print("=" * 50)

    try:
        from sklearn.metrics import classification_report
        report = classification_report(all_targets, all_preds, zero_division=0)
        print("\nDetailed Classification Report:")
        print(report)

        if report_file:
            Path(report_file).parent.mkdir(parents=True, exist_ok=True)
            with open(report_file, "w") as f:
                f.write(f"Traffic Sign Recognition Evaluation Report\n")
                f.write(f"Checkpoint: {checkpoint_path}\n")
                f.write(f"Accuracy: {accuracy:.2f}%\n\n")
                f.write(report)
            print(f"Report saved to {report_file}")
    except ImportError:
        print("[NOTE] Install scikit-learn for detailed per-class precision/recall/F1 metrics.")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Traffic Sign Recognition Model")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt", help="Path to model checkpoint")
    parser.add_argument("--data-dir", type=str, default="../data/processed", help="Path to evaluation dataset")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--report-file", type=str, default="evaluation_report.txt", help="Path to save text report")
    args = parser.parse_args()

    evaluate_model(args.checkpoint, args.data_dir, args.batch_size, args.report_file)


if __name__ == "__main__":
    main()
