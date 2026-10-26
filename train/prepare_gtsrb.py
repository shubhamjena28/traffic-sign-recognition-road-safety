#!/usr/bin/env python3
"""
GTSRB Track-Based Deduplication and Stratified Splitting Pipeline.

Follows the required workflow:
1. Identifies original GTSRB class folders and track structures.
2. Computes perceptual hashes (pHash) using imagehash.
3. Uses a fast BK-Tree (Burkhard-Keller Tree) metric index to efficiently find
   near-duplicate candidates (Hamming distance <= 4) without quadratic O(N^2) comparisons.
4. Distinguishes intra-track motion sequences (preserved) from cross-track duplicates (removed).
5. Safeguards conflicting labels before removal.
6. Performs reproducible track-based stratified splitting:
   70% Training, 15% Validation, 15% Testing with fixed seed (42).
7. Exports split manifests (train.csv, val.csv, test.csv, split_summary.json)
   and deduplication audit log (deduplication_log.csv).
"""

import argparse
import csv
import json
import os
from pathlib import Path
import random
import sys
import time
from typing import Dict, List, Optional, Set, Tuple

try:
    from PIL import Image, ImageDraw
    import imagehash
except ImportError as e:
    print(f"[ERROR] Required library missing: {e}")
    print("Please run: pip install imagehash Pillow pandas scikit-learn")
    sys.exit(1)


# =====================================================================
# BK-Tree (Burkhard-Keller Metric Tree) for Fast Hamming Distance Search
# =====================================================================

class BKTreeNode:
    """Node in the BK-Tree storing an image hash and its associated metadata."""

    def __init__(self, hash_val: imagehash.ImageHash, item: Dict):
        self.hash_val = hash_val
        self.item = item
        # Maps integer distance -> child BKTreeNode
        self.children: Dict[int, "BKTreeNode"] = {}


class BKTree:
    """
    Metric tree for efficient discrete distance search in Hamming space.
    Reduces candidate search from O(N) to O(log N) average complexity.
    """

    def __init__(self):
        self.root: Optional[BKTreeNode] = None
        self.size = 0

    def insert(self, hash_val: imagehash.ImageHash, item: Dict) -> None:
        """Insert a hash and its metadata into the BK-tree."""
        if self.root is None:
            self.root = BKTreeNode(hash_val, item)
            self.size = 1
            return

        current = self.root
        while True:
            dist = hash_val - current.hash_val
            if dist in current.children:
                current = current.children[dist]
            else:
                current.children[dist] = BKTreeNode(hash_val, item)
                self.size += 1
                break

    def find_within_distance(
        self, query_hash: imagehash.ImageHash, max_distance: int = 4
    ) -> List[Tuple[Dict, int]]:
        """
        Find all items in the tree within `max_distance` Hamming distance.
        Uses triangle inequality: |d(n, x) - d(n, q)| <= d(x, q) <= max_distance.
        """
        results: List[Tuple[Dict, int]] = []
        if self.root is None:
            return results

        candidates = [self.root]
        while candidates:
            node = candidates.pop()
            dist = query_hash - node.hash_val

            if dist <= max_distance:
                results.append((node.item, dist))

            # Triangle inequality pruning
            low = max(0, dist - max_distance)
            high = dist + max_distance

            for edge_dist, child in node.children.items():
                if low <= edge_dist <= high:
                    candidates.append(child)

        return results


# =====================================================================
# Track Identification & Dataset Discovery
# =====================================================================

def parse_gtsrb_filename(filepath: Path) -> Tuple[int, str, int]:
    """
    Parse GTSRB file naming convention:
    Format: .../<class_dir>/<track_id>_<frame_id>.ppm
    Example: .../00014/00002_00019.ppm -> class_id=14, track_id="00014_00002", frame_id=19
    """
    parent_name = filepath.parent.name
    try:
        class_id = int(parent_name)
    except ValueError:
        class_id = 0

    stem = filepath.stem
    if "_" in stem:
        parts = stem.split("_")
        track_part = parts[0]
        try:
            frame_id = int(parts[1])
        except ValueError:
            frame_id = 0
        track_id = f"{parent_name}_{track_part}"
    else:
        track_id = f"{parent_name}_{stem[:5]}"
        frame_id = 0

    return class_id, track_id, frame_id


def find_all_images(dataset_dir: Path) -> List[Path]:
    """Recursively discover all PPM, PNG, and JPG images in directory."""
    extensions = {".ppm", ".png", ".jpg", ".jpeg"}
    images = []
    for ext in extensions:
        images.extend(dataset_dir.rglob(f"*{ext}"))
        images.extend(dataset_dir.rglob(f"*{ext.upper()}"))
    # Sort for deterministic processing order
    return sorted(list(set(images)))


# =====================================================================
# Deduplication with BK-Tree & Perceptual Hashing
# =====================================================================

def deduplicate_images(
    image_paths: List[Path],
    threshold: int = 4,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Compute pHash for all images and detect duplicates using BK-Tree.
    Rules:
    - Retains legitimate intra-track consecutive frames.
    - Removes cross-track near-duplicates (distance <= threshold).
    - Protects pairs with conflicting labels from deletion and logs for review.
    """
    tree = BKTree()
    retained_items: List[Dict] = []
    audit_logs: List[Dict] = []

    print(f"[INFO] Computing perceptual hashes and deduplicating {len(image_paths)} images...")
    start_time = time.time()

    for idx, path in enumerate(image_paths):
        if (idx + 1) % 500 == 0 or idx == len(image_paths) - 1:
            elapsed = time.time() - start_time
            print(f"  Processed {idx + 1}/{len(image_paths)} images ({elapsed:.1f}s, tree size: {tree.size})...")

        try:
            with Image.open(path) as img:
                ph = imagehash.phash(img)
        except Exception as err:
            print(f"[WARN] Failed to open image {path}: {err}")
            continue

        class_id, track_id, frame_id = parse_gtsrb_filename(path)
        item = {
            "filepath": str(path.resolve()),
            "class_id": class_id,
            "track_id": track_id,
            "frame_id": frame_id,
        }

        # Query BK-Tree for candidate near-duplicates
        candidates = tree.find_within_distance(ph, max_distance=threshold)

        is_duplicate = False
        for matched_item, dist in candidates:
            # Check 1: Same track?
            if matched_item["track_id"] == track_id:
                if dist == 0:
                    # Identical twin frame within same track
                    is_duplicate = True
                    audit_logs.append({
                        "removed_image": str(path),
                        "matched_image": matched_item["filepath"],
                        "hamming_distance": dist,
                        "action": "REMOVED",
                        "reason": "Exact identical perceptual hash in same track",
                        "class_id_removed": class_id,
                        "class_id_matched": matched_item["class_id"],
                        "track_id_removed": track_id,
                        "track_id_matched": matched_item["track_id"],
                    })
                    break
                else:
                    # Consecutive frames showing legitimate approach: PRESERVE!
                    continue

            # Check 2: Different track (potential cross-track leakage)
            if matched_item["class_id"] != class_id:
                # Conflicting label warning! Review before deleting!
                audit_logs.append({
                    "removed_image": str(path),
                    "matched_image": matched_item["filepath"],
                    "hamming_distance": dist,
                    "action": "FLAGGED_RETAINED",
                    "reason": "Near-duplicate candidate has conflicting class label",
                    "class_id_removed": class_id,
                    "class_id_matched": matched_item["class_id"],
                    "track_id_removed": track_id,
                    "track_id_matched": matched_item["track_id"],
                })
                # Do not delete due to conflicting label
                continue
            else:
                # Cross-track duplicate leakage with matching label: REMOVE
                is_duplicate = True
                audit_logs.append({
                    "removed_image": str(path),
                    "matched_image": matched_item["filepath"],
                    "hamming_distance": dist,
                    "action": "REMOVED",
                    "reason": f"Cross-track duplicate leakage (Hamming distance {dist} <= {threshold})",
                    "class_id_removed": class_id,
                    "class_id_matched": matched_item["class_id"],
                    "track_id_removed": track_id,
                    "track_id_matched": matched_item["track_id"],
                })
                break

        if not is_duplicate:
            tree.insert(ph, item)
            retained_items.append(item)

    print(f"[SUCCESS] Deduplication completed in {time.time() - start_time:.2f}s.")
    print(f"  Retained: {len(retained_items)} images")
    removed_count = sum(1 for log in audit_logs if log["action"] == "REMOVED")
    flagged_count = sum(1 for log in audit_logs if log["action"] == "FLAGGED_RETAINED")
    print(f"  Duplicates removed: {removed_count}")
    print(f"  Label conflicts flagged: {flagged_count}")

    return retained_items, audit_logs


# =====================================================================
# Track-Based Stratified 70/15/15 Splitting
# =====================================================================

def split_tracks_stratified(
    retained_items: List[Dict],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict], List[Dict], Dict]:
    """
    Split remaining images strictly by track into 70% Train, 15% Val, 15% Test.
    Stratifies tracks across each of the 43 sign classes using a fixed seed.
    """
    rng = random.Random(seed)

    # Group items by class and track: class_id -> {track_id: [items]}
    class_tracks: Dict[int, Dict[str, List[Dict]]] = {}
    for item in retained_items:
        cid = item["class_id"]
        tid = item["track_id"]
        if cid not in class_tracks:
            class_tracks[cid] = {}
        if tid not in class_tracks[cid]:
            class_tracks[cid][tid] = []
        class_tracks[cid][tid].append(item)

    train_items: List[Dict] = []
    val_items: List[Dict] = []
    test_items: List[Dict] = []

    per_class_summary = {}

    for cid in sorted(class_tracks.keys()):
        tracks = sorted(list(class_tracks[cid].keys()))
        rng.shuffle(tracks)
        n_tracks = len(tracks)

        if n_tracks == 1:
            # Distribute single-track classes across train/val/test to maintain ratio
            if cid % 10 in (0, 1):
                val_t = tracks
                train_t, test_t = [], []
            elif cid % 10 in (2, 3):
                test_t = tracks
                train_t, val_t = [], []
            else:
                train_t = tracks
                val_t, test_t = [], []
        elif n_tracks == 2:
            train_t = [tracks[0]]
            val_t = [tracks[1]] if cid % 2 == 0 else []
            test_t = [tracks[1]] if cid % 2 != 0 else []
        else:
            n_train = max(1, int(round(n_tracks * train_ratio)))
            n_val = max(1, int(round(n_tracks * val_ratio)))
            n_test = n_tracks - n_train - n_val

            if n_test <= 0:
                n_test = 1
                if n_train > 1:
                    n_train -= 1
                elif n_val > 1:
                    n_val -= 1

            train_t = tracks[:n_train]
            val_t = tracks[n_train : n_train + n_val]
            test_t = tracks[n_train + n_val :]

        train_c_imgs = 0
        val_c_imgs = 0
        test_c_imgs = 0

        for tid in train_t:
            for itm in class_tracks[cid][tid]:
                itm_copy = dict(itm)
                itm_copy["split"] = "train"
                train_items.append(itm_copy)
                train_c_imgs += 1

        for tid in val_t:
            for itm in class_tracks[cid][tid]:
                itm_copy = dict(itm)
                itm_copy["split"] = "val"
                val_items.append(itm_copy)
                val_c_imgs += 1

        for tid in test_t:
            for itm in class_tracks[cid][tid]:
                itm_copy = dict(itm)
                itm_copy["split"] = "test"
                test_items.append(itm_copy)
                test_c_imgs += 1

        per_class_summary[str(cid)] = {
            "total_tracks": n_tracks,
            "train_tracks": len(train_t),
            "val_tracks": len(val_t),
            "test_tracks": len(test_t),
            "train_images": train_c_imgs,
            "val_images": val_c_imgs,
            "test_images": test_c_imgs,
        }

    total_images = len(train_items) + len(val_items) + len(test_items)
    summary = {
        "seed": seed,
        "total_images": total_images,
        "train_images": len(train_items),
        "val_images": len(val_items),
        "test_images": len(test_items),
        "train_percentage": round(len(train_items) / total_images * 100, 2) if total_images else 0,
        "val_percentage": round(len(val_items) / total_images * 100, 2) if total_images else 0,
        "test_percentage": round(len(test_items) / total_images * 100, 2) if total_images else 0,
        "per_class": per_class_summary,
    }

    return train_items, val_items, test_items, summary


# =====================================================================
# Starter Sample Dataset Generator (for Verification & Demo)
# =====================================================================

def create_sample_gtsrb(
    target_dir: Path,
    num_classes: int = 43,
    tracks_per_class: int = 10,
    frames_per_track: int = 8,
) -> None:
    """Generate synthetic starter GTSRB dataset with distinct track visual signatures."""
    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Generating sample GTSRB dataset in {target_dir} ({num_classes} classes, {tracks_per_class} tracks/class)...")

    for cid in range(num_classes):
        class_folder = target_dir / f"{cid:05d}"
        class_folder.mkdir(parents=True, exist_ok=True)

        for tid in range(tracks_per_class):
            track_prefix = f"{tid:05d}"
            # Unique visual seed per track to prevent cross-track false duplicate hashing
            base_r = (cid * 29 + tid * 47) % 220 + 30
            base_g = (cid * 53 + tid * 31) % 220 + 30
            base_b = (cid * 17 + tid * 79) % 220 + 30

            for fid in range(frames_per_track):
                img_name = f"{track_prefix}_{fid:05d}.ppm"
                img_path = class_folder / img_name

                # Simulate consecutive video approach: sign size changes across frames
                size = 48 + (fid * 2)
                img = Image.new("RGB", (size, size), color=(base_r, base_g, base_b))
                draw = ImageDraw.Draw(img)

                # Distinct geometric pattern per class
                pad = 6
                if cid == 14:  # Stop sign: red octagon
                    draw.polygon([
                        (pad + 8, pad), (size - pad - 8, pad),
                        (size - pad, pad + 8), (size - pad, size - pad - 8),
                        (size - pad - 8, size - pad), (pad + 8, size - pad),
                        (pad, size - pad - 8), (pad, pad + 8),
                    ], fill=(220, 20, 20), outline=(255, 255, 255))
                elif cid in {0, 1, 2, 3, 4, 5, 7, 8}:  # Speed: circular ring
                    draw.ellipse([pad, pad, size - pad, size - pad], fill=(255, 255, 255), outline=(220, 20, 20), width=4)
                    # Unique track inner glyph
                    draw.rectangle([size // 2 - 4, size // 2 - 8, size // 2 + 4, size // 2 + 8], fill=(0, 0, 0))
                else:  # Warning/other signs
                    draw.polygon([(size // 2, pad), (size - pad, size - pad), (pad, size - pad)], fill=(255, 220, 0), outline=(0, 0, 0))

                # Track-specific identifier marker
                marker_x = pad + (tid % 5) * 4
                marker_y = pad + (tid // 5) * 4
                draw.rectangle([marker_x, marker_y, marker_x + 5, marker_y + 5], fill=(255 - base_r, 255 - base_g, 255 - base_b))

                img.save(img_path)

        # Write sample CSV for class
        csv_file = class_folder / f"GT-{cid:05d}.csv"
        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            f.write("Filename;Width;Height;Roi.X1;Roi.Y1;Roi.X2;Roi.Y2;ClassId\n")
            for tid in range(tracks_per_class):
                for fid in range(frames_per_track):
                    f.write(f"{tid:05d}_{fid:05d}.ppm;48;48;4;4;44;44;{cid}\n")

    print("[SUCCESS] Starter GTSRB structure created.")


# =====================================================================
# Export Manifests
# =====================================================================

def write_split_csv(items: List[Dict], output_file: Path) -> None:
    """Write split manifest CSV."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["filepath", "class_id", "track_id", "frame_id", "split"]
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(items)


def write_audit_csv(audit_logs: List[Dict], output_file: Path) -> None:
    """Write deduplication audit log CSV."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "removed_image", "matched_image", "hamming_distance", "action", "reason",
        "class_id_removed", "class_id_matched", "track_id_removed", "track_id_matched",
    ]
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(audit_logs)


def main():
    parser = argparse.ArgumentParser(description="GTSRB Deduplication and Stratified Splitting")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/raw/gtsrb/train"),
        help="Path to GTSRB training directory containing class folders (00000 to 00042)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/splits"),
        help="Output directory for train.csv, val.csv, test.csv, split_summary.json",
    )
    parser.add_argument("--threshold", type=int, default=4, help="Hamming distance threshold for duplicates (default: 4)")
    parser.add_argument("--seed", type=int, default=42, help="Fixed random seed (default: 42)")
    parser.add_argument("--train-ratio", type=float, default=0.70, help="Train track ratio (default: 0.70)")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation track ratio (default: 0.15)")
    parser.add_argument("--test-ratio", type=float, default=0.15, help="Test track ratio (default: 0.15)")
    parser.add_argument(
        "--create-sample-if-empty",
        action="store_true",
        default=True,
        help="Create synthetic starter GTSRB dataset if data-dir is empty",
    )

    args = parser.parse_args()

    # Discover or generate images
    images = find_all_images(args.data_dir)
    if not images:
        if args.create_sample_if-empty if hasattr(args, "create_sample_if-empty") else args.create_sample_if_empty:
            print(f"[INFO] Directory {args.data_dir} is empty. Generating starter GTSRB dataset...")
            create_sample_gtsrb(args.data_dir)
            images = find_all_images(args.data_dir)
        else:
            print(f"[ERROR] No images found in {args.data_dir}")
            sys.exit(1)

    print(f"[INFO] Found {len(images)} raw images in {args.data_dir}.")

    # Step 1: Perceptual Hash Deduplication with BK-Tree
    retained_items, audit_logs = deduplicate_images(images, threshold=args.threshold)

    # Step 2: Track-Based Stratified Splitting (70/15/15)
    train_items, val_items, test_items, summary = split_tracks_stratified(
        retained_items,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    # Step 3: Write Manifests and Logs
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_split_csv(train_items, args.output_dir / "train.csv")
    write_split_csv(val_items, args.output_dir / "val.csv")
    write_split_csv(test_items, args.output_dir / "test.csv")
    write_audit_csv(audit_logs, args.output_dir / "deduplication_log.csv")

    summary_file = args.output_dir / "split_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[SUCCESS] Split manifests successfully generated in {args.output_dir}:")
    print(f"  Training split:   {len(train_items):5d} images ({summary['train_percentage']}%)")
    print(f"  Validation split: {len(val_items):5d} images ({summary['val_percentage']}%)")
    print(f"  Testing split:    {len(test_items):5d} images ({summary['test_percentage']}%)")
    print(f"  Summary saved:    {summary_file}")
    print(f"  Audit log saved:  {args.output_dir / 'deduplication_log.csv'}")


if __name__ == "__main__":
    main()
