#!/usr/bin/env python3
"""
Comprehensive YOLO Results Verifier, Safety Benchmark & Latency Profiler.

Executes all 5 verification phases:
1. Output Inspection (results.csv, curves, confusion matrix, val predictions, best.pt)
2. Test Set Evaluation (mAP@0.5, mAP@0.5:0.95, Precision, Recall, Per-class AP50)
3. Safety Metric: Missed-Sign Rate calculation (IoU >= 0.5 for Stop, Speed, Danger, All)
4. CPU Inference Latency Benchmark & Model Size verification
5. Benchmark Table generation & decision recommendations
"""

import argparse
import glob
import json
import os
from pathlib import Path
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
from ultralytics import YOLO


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Computes pairwise Intersection over Union (IoU) between boxes1 and boxes2 [x1, y1, x2, y2]."""
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N, M, 2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N, M, 2]

    wh = (rb - lt).clamp(min=0)  # [N, M, 2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N, M]

    union = area1[:, None] + area2 - inter
    return inter / union.clamp(min=1e-8)


def check_output_artifacts(run_dir: Path) -> Dict[str, bool]:
    """Phase 1: Checks for the existence of required diagnostic and visual artifacts."""
    print("\n" + "=" * 70)
    print("PHASE 1: INSPECT OUTPUT ARTIFACTS & DIAGNOSTICS")
    print("=" * 70)
    print(f"Run directory: {run_dir}")

    artifacts = {
        "weights/best.pt": run_dir / "weights" / "best.pt",
        "results.csv": run_dir / "results.csv",
        "results.png": run_dir / "results.png",
        "confusion_matrix.png": run_dir / "confusion_matrix.png",
        "val_batch0_pred.jpg": run_dir / "val_batch0_pred.jpg",
    }

    # Ultralytics PR/F1 curve variations
    pr_candidates = [run_dir / "BoxPR_curve.png", run_dir / "PR_curve.png"]
    f1_candidates = [run_dir / "BoxF1_curve.png", run_dir / "F1_curve.png"]
    artifacts["PR_curve.png"] = next((p for p in pr_candidates if p.exists()), pr_candidates[0])
    artifacts["F1_curve.png"] = next((p for p in f1_candidates if p.exists()), f1_candidates[0])

    status = {}
    for name, path in artifacts.items():
        exists = path.exists()
        status[name] = exists
        size_str = f"({path.stat().st_size / 1024:.1f} KB)" if exists else "[NOT FOUND]"
        print(f"  [{'✓' if exists else '✗'}] {name:<22} : {size_str}")

    # Inspect results.csv if present for overfitting red flags
    csv_path = artifacts["results.csv"]
    if csv_path.exists():
        try:
            import pandas as pd
            df = pd.read_csv(csv_path)
            df.columns = [c.strip() for c in df.columns]
            print(f"\n  Inspected results.csv: {len(df)} epochs logged.")
            if len(df) > 1:
                initial_loss = df.iloc[0].get("train/box_loss", 0.0)
                final_loss = df.iloc[-1].get("train/box_loss", 0.0)
                print(f"    Train Box Loss: {initial_loss:.4f} -> {final_loss:.4f}")
        except Exception as e:
            print(f"    [NOTE] Could not parse CSV details: {e}")

    return status


def evaluate_test_set(model: YOLO, data_yaml: str, imgsz: int = 640) -> Dict[str, float]:
    """Phase 2: Evaluates model on held-out test split."""
    print("\n" + "=" * 70)
    print("PHASE 2: REAL NUMBERS ON THE TEST SPLIT")
    print("=" * 70)

    # Run validation on test split
    metrics = model.val(data=data_yaml, split="test", imgsz=imgsz, verbose=False)

    map50 = float(metrics.box.map50)
    map50_95 = float(metrics.box.map)
    mp = float(metrics.box.mp)
    mr = float(metrics.box.mr)

    print(f"  mAP@0.5      : {map50:.4f}   (Target: >= 0.80) -> {'PASS' if map50 >= 0.80 else 'FAIL/NEEDS_MORE_TRAINING'}")
    print(f"  mAP@0.5:0.95 : {map50_95:.4f}")
    print(f"  mean precision: {mp:.4f}")
    print(f"  mean recall   : {mr:.4f}")

    per_class_ap50 = {}
    print("\n  Per-Class AP50:")
    for i, name in metrics.names.items():
        try:
            cls_idx = list(metrics.box.ap_class_index).index(i)
            ap50 = float(metrics.box.ap50[cls_idx])
            per_class_ap50[name] = ap50
            print(f"    - {name:<12} : AP50 = {ap50:.4f}")
        except (ValueError, IndexError):
            per_class_ap50[name] = 0.0
            print(f"    - {name:<12} : AP50 = N/A (no instances in test split)")

    return {
        "map50": map50,
        "map50_95": map50_95,
        "precision": mp,
        "recall": mr,
        "per_class_ap50": per_class_ap50,
    }


def compute_missed_sign_rate(
    model: YOLO, test_images_pattern: str, conf_thresh: float = 0.25, iou_thresh: float = 0.50
) -> Dict[str, Dict[str, float]]:
    """Phase 3: Computes exact safety metric (missed-sign rate) overall and per safety class."""
    print("\n" + "=" * 70)
    print(f"PHASE 3: MEASURE MISSED-SIGN RATE (Conf >= {conf_thresh}, IoU >= {iou_thresh})")
    print("=" * 70)

    class_names = {0: "stop", 1: "speed", 2: "danger"}
    stats = {
        "all": {"total": 0, "missed": 0},
        "stop": {"total": 0, "missed": 0},
        "speed": {"total": 0, "missed": 0},
        "danger": {"total": 0, "missed": 0},
    }

    test_images = glob.glob(test_images_pattern)
    if not test_images:
        print(f"[WARNING] No test images matched pattern: {test_images_pattern}")
        return {}

    for img_path in test_images:
        lbl_path = img_path.replace("images", "labels").rsplit(".", 1)[0] + ".txt"
        if not os.path.exists(lbl_path):
            continue

        res = model(img_path, conf=conf_thresh, verbose=False)[0]
        h, w = res.orig_shape
        pred_boxes = res.boxes.xyxy.cpu()
        pred_cls = res.boxes.cls.cpu()

        with open(lbl_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                c, xc, yc, bw, bh = map(float, parts)
                c = int(c)
                c_name = class_names.get(c, f"class_{c}")

                # Convert norm xywh to absolute xyxy
                gt_box = torch.tensor([[(xc - bw / 2.0) * w, (yc - bh / 2.0) * h, (xc + bw / 2.0) * w, (yc + bh / 2.0) * h]])

                stats["all"]["total"] += 1
                if c_name in stats:
                    stats[c_name]["total"] += 1

                hit = False
                if len(pred_boxes) > 0:
                    ious = box_iou(gt_box, pred_boxes)[0]
                    hit = bool(any(ious[j] >= iou_thresh and int(pred_cls[j]) == c for j in range(len(ious))))

                if not hit:
                    stats["all"]["missed"] += 1
                    if c_name in stats:
                        stats[c_name]["missed"] += 1

    rates = {}
    print(f"  Safety Class Results (Total test images: {len(test_images)}):")
    for group in ["stop", "speed", "danger", "all"]:
        tot = stats[group]["total"]
        mis = stats[group]["missed"]
        rate = (mis / tot * 100.0) if tot > 0 else 0.0
        status_str = "PASS (<= 5%)" if rate <= 5.0 else "EXCEEDS TARGET"
        rates[group] = {"total": tot, "missed": mis, "rate_percent": rate}
        print(f"    - {group.upper():<10} : {rate:6.2f}% missed ({mis}/{tot} signs) -> {status_str}")

    return rates


def benchmark_cpu_latency_and_size(model_path: Path, test_images: List[str], imgsz: int = 640) -> Tuple[float, float]:
    """Phase 4: Measures CPU latency (ms/frame) and deployed model file size (MB)."""
    print("\n" + "=" * 70)
    print("PHASE 4: MEASURE CPU LATENCY & MODEL FILE SIZE")
    print("=" * 70)

    # Size on disk
    size_mb = model_path.stat().st_size / (1024 * 1024)
    print(f"  Deployed Model Size: {size_mb:.2f} MB  (Target: <= 50 MB) -> {'PASS' if size_mb <= 50.0 else 'FAIL'}")

    # Latency test on CPU
    cpu_model = YOLO(str(model_path))

    # Warmup
    sample_img = test_images[0] if test_images else np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
    _ = cpu_model(sample_img, device="cpu", imgsz=imgsz, verbose=False)

    sample_pool = test_images[:min(50, len(test_images))] if test_images else [sample_img] * 30
    t0 = time.perf_counter()
    for p in sample_pool:
        _ = cpu_model(p, device="cpu", imgsz=imgsz, verbose=False)
    elapsed = time.perf_counter() - t0

    ms_per_frame = (elapsed / len(sample_pool)) * 1000.0
    print(f"  CPU Latency        : {ms_per_frame:.2f} ms/frame  (Target: <= 150 ms) -> {'PASS' if ms_per_frame <= 150.0 else 'FAIL'}")

    return ms_per_frame, size_mb


def print_and_save_benchmark_table(
    model_name: str,
    val_metrics: Dict[str, float],
    missed_rates: Dict[str, Dict[str, float]],
    latency_ms: float,
    size_mb: float,
    output_json: Path,
) -> None:
    """Phase 5: Builds the final benchmark table and documents action recommendations."""
    print("\n" + "=" * 78)
    print("                     PHASE 5: FINAL BENCHMARK TABLE")
    print("=" * 78)

    map50 = val_metrics.get("map50", 0.0)
    map50_95 = val_metrics.get("map50_95", 0.0)
    overall_missed = missed_rates.get("all", {}).get("rate_percent", 0.0)
    stop_missed = missed_rates.get("stop", {}).get("rate_percent", 0.0)
    speed_missed = missed_rates.get("speed", {}).get("rate_percent", 0.0)
    danger_missed = missed_rates.get("danger", {}).get("rate_percent", 0.0)

    header = f"{'Model':<12} | {'mAP@0.5':<8} | {'mAP@50:95':<9} | {'Missed (Stop/Speed/Danger/All)':<30} | {'ms/frame':<9} | {'Size':<8}"
    print(header)
    print("-" * 78)

    missed_combined = f"{stop_missed:.1f}% / {speed_missed:.1f}% / {danger_missed:.1f}% / {overall_missed:.1f}%"
    row = f"{model_name:<12} | {map50:<8.4f} | {map50_95:<9.4f} | {missed_combined:<30} | {latency_ms:<6.1f} ms | {size_mb:<5.1f} MB"
    print(row)
    print("=" * 78)

    # Next step recommendation logic
    print("\nDECISION & NEXT ACTIONS:")
    if map50 >= 0.80 and overall_missed <= 5.0:
        print("  ✓ Result: Target criteria satisfied (mAP@0.5 >= 0.80 and missed rate <= 5%).")
        print("    -> Recommendation: Keep this as your baseline and consider training YOLOv8s for higher precision.")
    elif overall_missed > 5.0 and map50 >= 0.70:
        print("  ! Result: Missed-sign rate exceeds 5% safety threshold.")
        print("    -> Recommendation: Lower confidence threshold from 0.25 to 0.15 in inference to minimize false negatives.")
    else:
        print("  ! Result: Model requires further training convergence.")
        print("    -> Recommendation: Train for 100+ epochs on Colab using the full GTSDB/GTSRB dataset.")

    # Save to JSON
    summary = {
        "model": model_name,
        "map50": map50,
        "map50_95": map50_95,
        "precision": val_metrics.get("precision", 0.0),
        "recall": val_metrics.get("recall", 0.0),
        "per_class_ap50": val_metrics.get("per_class_ap50", {}),
        "missed_sign_rates": missed_rates,
        "cpu_latency_ms": latency_ms,
        "model_size_mb": size_mb,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[SUCCESS] Benchmark summary saved to: {output_json}")


def main():
    parser = argparse.ArgumentParser(description="Check YOLO Model Results & Road Safety Metrics")
    parser.add_argument("--weights", type=str, default="runs/detect/train-3/weights/best.pt", help="Path to best.pt")
    parser.add_argument("--data", type=str, default="data/traffic_signs.yaml", help="Path to data YAML")
    parser.add_argument("--test-pattern", type=str, default="data/images/test/*.jpg", help="Pattern for test images")
    parser.add_argument("--imgsz", type=int, default=640, help="Evaluation image resolution")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold for missed-sign calculation")
    parser.add_argument("--output-json", type=str, default="runs/detect/benchmark_summary.json", help="Summary JSON output")
    args = parser.parse_args()

    weights_path = Path(args.weights)
    if not weights_path.exists():
        # Fall back to alternative runs
        candidates = list(Path("runs/detect").glob("*/weights/best.pt"))
        if candidates:
            weights_path = candidates[0]
            print(f"[INFO] Using detected weights: {weights_path}")
        else:
            raise FileNotFoundError(f"Could not find model weights at: {args.weights}")

    run_dir = weights_path.parent.parent

    # Phase 1: Output artifacts
    check_output_artifacts(run_dir)

    # Load model
    print(f"\nLoading YOLO model: {weights_path} ...")
    model = YOLO(str(weights_path))

    # Phase 2: Test set evaluation
    val_metrics = evaluate_test_set(model, args.data, imgsz=args.imgsz)

    # Phase 3: Missed-sign rate
    missed_rates = compute_missed_sign_rate(model, args.test_pattern, conf_thresh=args.conf, iou_thresh=0.50)

    # Phase 4: CPU latency & model size
    test_imgs = glob.glob(args.test_pattern)
    latency_ms, size_mb = benchmark_cpu_latency_and_size(weights_path, test_imgs, imgsz=args.imgsz)

    # Phase 5: Final summary & benchmark table
    print_and_save_benchmark_table(
        weights_path.stem, val_metrics, missed_rates, latency_ms, size_mb, Path(args.output_json)
    )


if __name__ == "__main__":
    main()
