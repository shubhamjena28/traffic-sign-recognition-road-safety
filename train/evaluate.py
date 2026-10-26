"""Evaluation script for trained Traffic Sign Recognition models.

Computes classification metrics including Accuracy, Precision, Recall,
F1-Score, CPU inference latency per frame, missed-sign rates for safety classes,
and generates Confusion Matrix visualization.
"""

import argparse
import json
from pathlib import Path
import sys
import time

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Import definitions from train
sys.path.append(str(Path(__file__).parent))
from train import TrafficSignCNN, CSVTrafficSignDataset, get_data_transforms


# Project target mappings for road safety evaluation
STOP_CLASSES = {14}
SPEED_CLASSES = {0, 1, 2, 3, 4, 5, 7, 8}
DANGER_CLASSES = {11, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31}


def evaluate_model(
    checkpoint_path: str,
    test_csv: str = None,
    data_dir: str = None,
    batch_size: int = 64,
    report_file: str = None,
    confusion_matrix_path: str = None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Evaluating on device: {device}")

    ckpt_file = Path(checkpoint_path)
    if not ckpt_file.exists():
        print(f"[ERROR] Checkpoint file '{checkpoint_path}' does not exist.")
        return

    _, val_transform = get_data_transforms()

    if test_csv and Path(test_csv).exists():
        print(f"[INFO] Loading test dataset from manifest: {test_csv}")
        test_dataset = CSVTrafficSignDataset(test_csv, transform=val_transform)
    elif data_dir and Path(data_dir).exists() and any(Path(data_dir).iterdir()):
        print(f"[INFO] Loading test dataset from directory: {data_dir}")
        test_dataset = datasets.ImageFolder(root=str(data_dir), transform=val_transform)
    else:
        print(f"[ERROR] Neither valid test CSV ({test_csv}) nor data directory ({data_dir}) exists.")
        return

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    checkpoint = torch.load(ckpt_file, map_location=device)
    num_classes = checkpoint.get("num_classes", 43)

    model = TrafficSignCNN(num_classes=num_classes).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    all_preds = []
    all_targets = []
    latencies = []

    print(f"Evaluating {len(test_dataset)} test samples...")
    with torch.no_grad():
        for images, targets in test_loader:
            images = images.to(device)

            t0 = time.perf_counter()
            outputs = model(images)
            t1 = time.perf_counter()

            # Latency per sample in ms
            batch_latency = ((t1 - t0) * 1000.0) / images.size(0)
            latencies.extend([batch_latency] * images.size(0))

            _, preds = outputs.max(1)
            all_preds.extend(preds.cpu().tolist())
            all_targets.extend(targets.tolist())

    correct = sum(p == t for p, t in zip(all_preds, all_targets))
    accuracy = 100.0 * correct / max(len(all_targets), 1)
    avg_latency = sum(latencies) / max(len(latencies), 1)

    # Missed-sign rate calculations for stop, speed, danger
    def compute_missed_rate(target_set, name):
        indices = [i for i, t in enumerate(all_targets) if t in target_set]
        if not indices:
            return 0.0, 0, 0
        total = len(indices)
        missed = sum(1 for i in indices if all_preds[i] != all_targets[i])
        miss_rate = (missed / total) * 100.0
        return miss_rate, missed, total

    stop_miss, stop_cnt_miss, stop_tot = compute_missed_rate(STOP_CLASSES, "stop")
    speed_miss, speed_cnt_miss, speed_tot = compute_missed_rate(SPEED_CLASSES, "speed")
    danger_miss, danger_cnt_miss, danger_tot = compute_missed_rate(DANGER_CLASSES, "danger")

    print("\n" + "=" * 60)
    print(f"Evaluation Results for: {ckpt_file.name}")
    print("=" * 60)
    print(f"Total Samples Evaluated:     {len(all_targets)}")
    print(f"Overall Accuracy:            {accuracy:.2f}%")
    print(f"Avg CPU Inference Latency:   {avg_latency:.2f} ms/frame (Target: <= 150 ms)")
    print(f"Stop Sign Missed Rate:       {stop_miss:.2f}% ({stop_cnt_miss}/{stop_tot}) (Target: <= 5%)")
    print(f"Speed Limit Missed Rate:     {speed_miss:.2f}% ({speed_cnt_miss}/{speed_tot}) (Target: <= 5%)")
    print(f"Danger Warning Missed Rate:  {danger_miss:.2f}% ({danger_cnt_miss}/{danger_tot}) (Target: <= 5%)")
    print("=" * 60)

    try:
        from sklearn.metrics import classification_report, confusion_matrix
        import matplotlib.pyplot as plt

        report = classification_report(all_targets, all_preds, zero_division=0)
        print("\nDetailed Classification Report:")
        print(report)

        if confusion_matrix_path and len(set(all_targets)) > 1:
            cm = confusion_matrix(all_targets, all_preds)
            fig, ax = plt.subplots(figsize=(10, 8))
            cax = ax.matshow(cm, cmap=plt.cm.Blues)
            fig.colorbar(cax)
            ax.set_title("GTSRB Confusion Matrix", pad=20)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            Path(confusion_matrix_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(confusion_matrix_path, bbox_inches="tight", dpi=150)
            plt.close()
            print(f"[SUCCESS] Saved confusion matrix to {confusion_matrix_path}")

        if report_file:
            Path(report_file).parent.mkdir(parents=True, exist_ok=True)
            with open(report_file, "w") as f:
                f.write(f"Traffic Sign Recognition Evaluation Report\n")
                f.write(f"Checkpoint: {checkpoint_path}\n")
                f.write(f"Overall Accuracy: {accuracy:.2f}%\n")
                f.write(f"Avg Latency: {avg_latency:.2f} ms/frame\n")
                f.write(f"Stop Missed Rate: {stop_miss:.2f}%\n")
                f.write(f"Speed Missed Rate: {speed_miss:.2f}%\n")
                f.write(f"Danger Missed Rate: {danger_miss:.2f}%\n\n")
                f.write(report)
            print(f"[SUCCESS] Report saved to {report_file}")
    except ImportError:
        pass


def main():
    parser = argparse.ArgumentParser(description="Evaluate Traffic Sign Recognition Model")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt", help="Path to model checkpoint")
    parser.add_argument("--test-csv", type=str, default="data/splits/test.csv", help="Path to test.csv manifest")
    parser.add_argument("--data-dir", type=str, default="data/processed", help="Path to evaluation dataset directory (fallback)")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--report-file", type=str, default="data/splits/evaluation_report.txt", help="Path to save text report")
    parser.add_argument("--confusion-matrix", type=str, default="data/splits/confusion_matrix.png", help="Path to save confusion matrix image")
    args = parser.parse_args()

    evaluate_model(
        checkpoint_path=args.checkpoint,
        test_csv=args.test_csv,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        report_file=args.report_file,
        confusion_matrix_path=args.confusion_matrix,
    )


if __name__ == "__main__":
    main()
