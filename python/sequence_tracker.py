#!/usr/bin/env python3
"""
Sequence Tracker: YOLO + ResNet-18 Multi-Frame Perception & Approaching Sign Tracking.

Requirements:
- Runs YOLO detector and ResNet-18 classifier over consecutive frame sequences.
- Per frame, calculates:
  * Number of detected signs
  * Class groups (stop, speed, danger, pedestrian)
  * Largest box area (w * h)
  * Growth of box size (area_t - area_{t-1}), serving as a proxy for closing vehicle distance.
- Saves telemetry to CSV and visualizes approaching curves.
"""

import argparse
import csv
import math
import os
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import models, transforms
from PIL import Image


CLASS_NAMES = {0: "stop", 1: "speed", 2: "danger", 3: "pedestrian"}


def generate_synthetic_approach_sequence(output_dir: Path, num_frames: int = 30) -> List[Path]:
    """
    Generates a realistic 30-frame road approach dashcam sequence.
    Signs grow in size as the vehicle approaches (area increases quadratically).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_paths = []

    width, height = 640, 480

    for i in range(num_frames):
        # Progress from 0.1 (far away) to 1.0 (close up)
        progress = 0.15 + 0.85 * (i / max(1, num_frames - 1))
        # Simulated road background
        frame = np.full((height, width, 3), 70, dtype=np.uint8)
        # Horizon and sky
        frame[:200, :] = [180, 150, 100]
        # Road lane lines
        cv2.line(frame, (100, height), (300, 200), (255, 255, 255), 3)
        cv2.line(frame, (width - 100, height), (340, 200), (255, 255, 255), 3)

        # Approaching sign on right shoulder
        sign_radius = int(12 + progress * 58)  # radius grows from 12 to 70 px
        cx = int(380 + progress * 140)         # moves outward as camera approaches
        cy = int(210 + progress * 80)

        # Sign type switches across sequence phases or shows STOP approach
        if i < 15:
            # STOP Sign (Octagon)
            angles = np.linspace(0, 2 * np.pi, 9)[:-1] + (np.pi / 8)
            pts = np.array([[int(cx + sign_radius * np.cos(a)), int(cy + sign_radius * np.sin(a))] for a in angles], np.int32)
            cv2.fillPoly(frame, [pts.reshape((-1, 1, 2))], (20, 20, 220))
            cv2.polylines(frame, [pts.reshape((-1, 1, 2))], True, (255, 255, 255), max(1, sign_radius // 15))
            if sign_radius > 20:
                font_scale = sign_radius / 38.0
                cv2.putText(frame, "STOP", (cx - int(sign_radius * 0.7), cy + int(sign_radius * 0.25)),
                            cv2.FONT_HERSHEY_DUPLEX, font_scale, (255, 255, 255), max(1, int(font_scale * 2)))
        else:
            # Speed Limit 50 Sign (Circle)
            cv2.circle(frame, (cx, cy), sign_radius, (20, 20, 220), -1)
            cv2.circle(frame, (cx, cy), max(2, sign_radius - max(2, sign_radius // 6)), (255, 255, 255), -1)
            if sign_radius > 20:
                font_scale = sign_radius / 36.0
                cv2.putText(frame, "50", (cx - int(sign_radius * 0.45), cy + int(sign_radius * 0.3)),
                            cv2.FONT_HERSHEY_DUPLEX, font_scale, (0, 0, 0), max(1, int(font_scale * 2)))

        frame_path = output_dir / f"frame_{i:04d}.jpg"
        cv2.imwrite(str(frame_path), frame)
        frame_paths.append(frame_path)

    print(f"[INFO] Generated {len(frame_paths)} synthetic dashcam frames in: {output_dir}")
    return frame_paths


class SequenceTracker:
    """Combines YOLO detection with ResNet-18 classification to extract sequence telemetry."""

    def __init__(self, yolo_weights: str = "runs/detect/train-3/weights/best.pt",
                 resnet_weights: str = "checkpoints/resnet18_gtsrb.pt"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.yolo_model = None
        self.resnet_model = None
        self.eval_transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Load YOLO if available
        if os.path.exists(yolo_weights):
            try:
                from ultralytics import YOLO
                self.yolo_model = YOLO(yolo_weights)
                print(f"[✓] YOLO model loaded from {yolo_weights}")
            except Exception as e:
                print(f"[WARN] Failed to load YOLO: {e}")

        # Load ResNet-18
        try:
            r18 = models.resnet18(weights=None)
            in_feat = r18.fc.in_features
            r18.fc = torch.nn.Sequential(
                torch.nn.Dropout(0.3),
                torch.nn.Linear(in_feat, 43)
            )
            if os.path.exists(resnet_weights):
                ckpt = torch.load(resnet_weights, map_location=self.device, weights_only=False)
                state = ckpt.get("model_state_dict", ckpt)
                r18.load_state_dict(state)
                print(f"[✓] ResNet-18 classifier loaded from {resnet_weights}")
            r18.eval().to(self.device)
            self.resnet_model = r18
        except Exception as e:
            print(f"[NOTE] ResNet-18 initialized with fallback: {e}")

    def detect_signs_in_frame(self, frame_bgr: np.ndarray) -> List[Dict]:
        """Detects sign candidates in frame using YOLO or contour heuristic fallback."""
        detections = []
        h, w = frame_bgr.shape[:2]

        if self.yolo_model is not None:
            results = self.yolo_model(frame_bgr, conf=0.50, verbose=False)[0]
            boxes = results.boxes.xyxy.cpu().numpy()
            clss = results.boxes.cls.cpu().numpy()
            confs = results.boxes.conf.cpu().numpy()
            for b, c, score in zip(boxes, clss, confs):
                x1, y1, x2, y2 = b
                area = float((x2 - x1) * (y2 - y1))
                if score >= 0.50 and 200 < area < (w * h * 0.4):
                    detections.append({
                        "box": (float(x1), float(y1), float(x2), float(y2)),
                        "area": area,
                        "yolo_cls": int(c),
                        "conf": float(score)
                    })
            # Keep top 3 most confident detections
            detections = sorted(detections, key=lambda d: d["conf"], reverse=True)[:3]

        # Fallback if no detector weights or zero detections in synthetic sequence
        if not detections:
            # Detect prominent red regions (traffic sign markers)
            hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
            mask1 = cv2.inRange(hsv, np.array([0, 120, 120]), np.array([10, 255, 255]))
            mask2 = cv2.inRange(hsv, np.array([170, 120, 120]), np.array([179, 255, 255]))
            mask = mask1 | mask2
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if 100 < area < (w * h * 0.4):
                    x, y, bw, bh = cv2.boundingRect(cnt)
                    aspect = bw / float(bh)
                    if 0.5 < aspect < 2.0:
                        detections.append({
                            "box": (float(x), float(y), float(x + bw), float(y + bh)),
                            "area": float(bw * bh),
                            "yolo_cls": 0,
                            "conf": 0.88
                        })

        return detections

    def classify_crop(self, crop_bgr: np.ndarray) -> Tuple[int, str]:
        """Classifies cropped traffic sign using ResNet-18."""
        if crop_bgr.size == 0 or self.resnet_model is None:
            return 0, "stop"

        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(crop_rgb)
        inp = self.eval_transform(pil_img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.resnet_model(inp)
            pred_id = int(torch.argmax(logits, dim=1).item())

        # Map to safety categories: 0=stop, 1=speed, 2=danger, 3=pedestrian
        if pred_id in [14, 0]:
            cat_id, cat_name = 0, "stop"
        elif pred_id in [1, 2, 3, 4, 5, 7, 8]:
            cat_id, cat_name = 1, "speed"
        elif pred_id in [18, 19, 20, 21, 22, 23, 24, 25]:
            cat_id, cat_name = 2, "danger"
        elif pred_id in [27, 28]:
            cat_id, cat_name = 3, "pedestrian"
        else:
            cat_id, cat_name = (pred_id % 3), CLASS_NAMES[pred_id % 3]

        return cat_id, cat_name

    def process_sequence(self, frame_paths: List[Path], fps: float = 30.0) -> List[Dict]:
        """Processes all frames in sequence, computing box growth and temporal telemetry."""
        telemetry_rows = []
        prev_largest_area = 0.0

        for frame_idx, fpath in enumerate(frame_paths):
            timestamp = frame_idx / fps
            frame_bgr = cv2.imread(str(fpath))
            if frame_bgr is None:
                continue

            raw_detections = self.detect_signs_in_frame(frame_bgr)
            num_signs = len(raw_detections)

            class_groups = set()
            largest_area = 0.0
            dominant_class = "none"
            dominant_class_id = -1

            for det in raw_detections:
                x1, y1, x2, y2 = map(int, det["box"])
                crop = frame_bgr[max(0, y1):min(frame_bgr.shape[0], y2), max(0, x1):min(frame_bgr.shape[1], x2)]
                cid, cname = self.classify_crop(crop)
                class_groups.add(cname)

                if det["area"] > largest_area:
                    largest_area = det["area"]
                    dominant_class = cname
                    dominant_class_id = cid

            # Box growth = Area_t - Area_{t-1}
            box_growth = largest_area - prev_largest_area if frame_idx > 0 else 0.0
            prev_largest_area = largest_area

            # Estimated relative distance proxy: d proportional to 1 / sqrt(area)
            dist_proxy = round(100.0 / math.sqrt(largest_area), 2) if largest_area > 10 else 100.0

            # Deterministic Rule-Based Risk Label:
            # - HIGH (2): STOP or Danger sign with positive box growth (approaching) and significant box area
            # - MEDIUM (1): Speed limit sign or moderate approach
            # - LOW (0): Stationary, distant, or no sign
            if (dominant_class in ["stop", "danger", "pedestrian"] and box_growth > 10.0 and largest_area > 1200) or largest_area > 5000:
                risk_level = 2
                risk_label = "HIGH"
            elif dominant_class in ["speed", "danger"] or box_growth > 5.0 or (num_signs > 0 and largest_area > 400):
                risk_level = 1
                risk_label = "MEDIUM"
            else:
                risk_level = 0
                risk_label = "LOW"

            telemetry_rows.append({
                "frame_id": frame_idx,
                "timestamp_sec": round(timestamp, 3),
                "num_signs": num_signs,
                "class_groups": ";".join(sorted(list(class_groups))) if class_groups else "none",
                "dominant_class": dominant_class,
                "dominant_class_id": dominant_class_id,
                "largest_box_area": round(largest_area, 2),
                "box_growth": round(box_growth, 2),
                "distance_proxy_m": dist_proxy,
                "risk_level": risk_level,
                "risk_label": risk_label
            })

        return telemetry_rows


def save_telemetry_csv(rows: List[Dict], out_path: Path) -> None:
    """Exports frame sequence telemetry to CSV."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[✓] Sequence telemetry saved to: {out_path}")


def plot_sequence_curves(rows: List[Dict], out_plot_path: Path) -> None:
    """Plots largest box area, box growth rate, and risk level over frame sequence."""
    out_plot_path.parent.mkdir(parents=True, exist_ok=True)
    frames = [r["frame_id"] for r in rows]
    areas = [r["largest_box_area"] for r in rows]
    growths = [r["box_growth"] for r in rows]
    risks = [r["risk_level"] for r in rows]

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    # 1. Box Area
    ax1.plot(frames, areas, color="#2563eb", linewidth=2.5, marker="o", markersize=4, label="Largest Box Area ($px^2$)")
    ax1.set_ylabel("Box Area ($px^2$)")
    ax1.set_title("Approaching Traffic Sign Perception: Area & Growth Tracking Over Sequence", fontsize=13, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(loc="upper left")

    # 2. Box Growth (Proxy for vehicle closing speed)
    colors = ["#16a34a" if g >= 0 else "#dc2626" for g in growths]
    ax2.bar(frames, growths, color=colors, width=0.6, label=r"Box Growth $\Delta Area_t$ ($px^2/frame$)")
    ax2.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax2.set_ylabel("Growth Rate")
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(loc="upper left")

    # 3. Risk Level
    risk_colors = ["#16a34a" if r == 0 else ("#f59e0b" if r == 1 else "#dc2626") for r in risks]
    ax3.scatter(frames, risks, c=risk_colors, s=70, zorder=3)
    ax3.plot(frames, risks, color="#4b5563", linestyle=":", alpha=0.7)
    ax3.set_yticks([0, 1, 2])
    ax3.set_yticklabels(["LOW (0)", "MEDIUM (1)", "HIGH (2)"])
    ax3.set_xlabel("Frame Sequence Index")
    ax3.set_ylabel("Risk State")
    ax3.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(out_plot_path, dpi=200)
    plt.close()
    print(f"[✓] Sequence tracking growth plot saved to: {out_plot_path}")


def main():
    parser = argparse.ArgumentParser(description="Multi-frame Approaching Sign Tracker (YOLO + ResNet)")
    parser.add_argument("--frames-dir", type=str, default="data/sequences/frames", help="Path to input sequence frames")
    parser.add_argument("--out-csv", type=str, default="data/sequences/approach_sequences.csv", help="Output telemetry CSV")
    parser.add_argument("--out-plot", type=str, default="checkpoints/sequence_tracking_growth.png", help="Output tracking plot")
    parser.add_argument("--yolo-weights", type=str, default="runs/detect/train-3/weights/best.pt", help="YOLO model path")
    parser.add_argument("--resnet-weights", type=str, default="checkpoints/resnet18_gtsrb.pt", help="ResNet-18 weights path")
    parser.add_argument("--num-synthetic", type=int, default=30, help="Number of synthetic frames if directory empty")
    args = parser.parse_args()

    frames_dir = Path(args.frames_dir)
    frame_paths = sorted(list(frames_dir.glob("*.jpg")) + list(frames_dir.glob("*.png"))) if frames_dir.exists() else []

    if not frame_paths:
        frame_paths = generate_synthetic_approach_sequence(frames_dir, num_frames=args.num_synthetic)

    tracker = SequenceTracker(yolo_weights=args.yolo_weights, resnet_weights=args.resnet_weights)
    telemetry = tracker.process_sequence(frame_paths)

    save_telemetry_csv(telemetry, Path(args.out_csv))
    plot_sequence_curves(telemetry, Path(args.out_plot))

    # Also copy to docs/visualizations
    doc_plot = Path("docs/visualizations/sequence_tracking_growth.png")
    doc_plot.parent.mkdir(parents=True, exist_ok=True)
    import shutil
    shutil.copyfile(args.out_plot, doc_plot)


if __name__ == "__main__":
    main()
