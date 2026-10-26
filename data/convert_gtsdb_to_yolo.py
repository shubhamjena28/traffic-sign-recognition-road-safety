#!/usr/bin/env python3
"""
Convert GTSDB (German Traffic Sign Detection Benchmark) annotations to YOLO format.

Preserves the original annotations in data/raw/gtsdb/annotations/gt.txt while
generating normalized YOLO bounding box text files with strict mapping to the
project's target classes:
  0: stop
  1: speed
  2: danger
"""

import argparse
from collections import defaultdict
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# Strict GTSDB Class ID to Target Class ID Mapping
# Target 0: stop
# Target 1: speed
# Target 2: danger
GTSDB_TO_TARGET_MAP: Dict[int, int] = {
    # 0: stop
    14: 0,  # Stop sign

    # 1: speed
    0: 1,   # Speed limit 20 km/h
    1: 1,   # Speed limit 30 km/h
    2: 1,   # Speed limit 50 km/h
    3: 1,   # Speed limit 60 km/h
    4: 1,   # Speed limit 70 km/h
    5: 1,   # Speed limit 80 km/h
    7: 1,   # Speed limit 100 km/h
    8: 1,   # Speed limit 120 km/h

    # 2: danger
    11: 2,  # Right-of-way at intersection
    18: 2,  # General danger
    19: 2,  # Dangerous curve to the left
    20: 2,  # Dangerous curve to the right
    21: 2,  # Double curve
    22: 2,  # Bumpy road
    23: 2,  # Slippery road
    24: 2,  # Road narrows on the right
    25: 2,  # Road work
    26: 2,  # Traffic signals
    27: 2,  # Pedestrians
    28: 2,  # Children crossing
    29: 2,  # Bicycles crossing
    30: 2,  # Beware of ice/snow
    31: 2,  # Wild animals crossing
}

TARGET_NAMES = {0: "stop", 1: "speed", 2: "danger"}


def convert_bbox(
    x1: float, y1: float, x2: float, y2: float, img_w: int, img_h: int
) -> Tuple[float, float, float, float]:
    """Convert absolute [x1, y1, x2, y2] to YOLO normalized [xc, yc, w, h]."""
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h
    xc = (x1 + x2) / (2.0 * img_w)
    yc = (y1 + y2) / (2.0 * img_h)

    # Clamp to [0.0, 1.0]
    xc = max(0.0, min(1.0, xc))
    yc = max(0.0, min(1.0, yc))
    w = max(0.0, min(1.0, w))
    h = max(0.0, min(1.0, h))

    return xc, yc, w, h


def parse_gtsdb_annotations(
    gt_file: Path,
    img_width: int = 1360,
    img_height: int = 1024,
    ignore_unmapped: bool = True,
) -> Tuple[Dict[str, List[Tuple[int, float, float, float, float]]], Dict[str, int]]:
    """
    Parse GTSDB semicolon-delimited ground truth file.
    Returns:
      annotations_by_image: {base_stem: [(target_class_id, xc, yc, w, h)]}
      stats: count statistics
    """
    annotations_by_image = defaultdict(list)
    stats = {
        "total_signs": 0,
        "stop": 0,
        "speed": 0,
        "danger": 0,
        "unmapped_ignored": 0,
    }

    if not gt_file.exists():
        raise FileNotFoundError(f"Annotations file not found: {gt_file}")

    with open(gt_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split(";")
            if len(parts) < 6:
                continue

            img_name = parts[0].strip()
            x1 = float(parts[1].strip())
            y1 = float(parts[2].strip())
            x2 = float(parts[3].strip())
            y2 = float(parts[4].strip())
            raw_class_id = int(parts[5].strip())

            stats["total_signs"] += 1

            if raw_class_id not in GTSDB_TO_TARGET_MAP:
                stats["unmapped_ignored"] += 1
                if ignore_unmapped:
                    continue
                target_id = -1
            else:
                target_id = GTSDB_TO_TARGET_MAP[raw_class_id]
                stats[TARGET_NAMES[target_id]] += 1

            xc, yc, w, h = convert_bbox(x1, y1, x2, y2, img_width, img_height)
            stem = Path(img_name).stem
            annotations_by_image[stem].append((target_id, xc, yc, w, h))

    return annotations_by_image, stats


def export_yolo_labels(
    annotations: Dict[str, List[Tuple[int, float, float, float, float]]],
    output_dir: Path,
) -> int:
    """Write normalized YOLO txt label files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    count = 0

    for stem, boxes in annotations.items():
        label_file = output_dir / f"{stem}.txt"
        with open(label_file, "w", encoding="utf-8") as f:
            for cls_id, xc, yc, w, h in boxes:
                if cls_id >= 0:
                    f.write(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")
        count += 1

    return count


def main():
    parser = argparse.ArgumentParser(description="Convert GTSDB annotations to YOLO format")
    parser.add_argument(
        "--annotations",
        type=Path,
        default=Path("data/raw/gtsdb/annotations/gt.txt"),
        help="Path to GTSDB gt.txt",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/gtsdb/labels"),
        help="Target directory for converted YOLO txt labels",
    )
    parser.add_argument("--img-width", type=int, default=1360, help="GTSDB image width")
    parser.add_argument("--img-height", type=int, default=1024, help="GTSDB image height")
    parser.add_argument(
        "--keep-unmapped",
        action="store_true",
        help="Retain unmapped classes instead of ignoring them",
    )

    args = parser.parse_args()

    print(f"[INFO] Reading annotations from: {args.annotations}")
    if not args.annotations.exists():
        print(f"[WARN] File {args.annotations} does not exist yet. Running in demonstration mode.")
        return

    annotations, stats = parse_gtsdb_annotations(
        args.annotations,
        img_width=args.img_width,
        img_height=args.img_height,
        ignore_unmapped=not args.keep_unmapped,
    )

    exported = export_yolo_labels(annotations, args.output_dir)
    print(f"[SUCCESS] Converted labels for {exported} images to {args.output_dir}")
    print(f"  Total signs processed: {stats['total_signs']}")
    print(f"  Stop signs mapped:     {stats['stop']}")
    print(f"  Speed signs mapped:    {stats['speed']}")
    print(f"  Danger signs mapped:   {stats['danger']}")
    print(f"  Unmapped signs excluded: {stats['unmapped_ignored']}")


if __name__ == "__main__":
    main()
