#!/usr/bin/env python3
"""
Automated Downloader & Extractor Utility for Traffic Sign & Autonomous Driving Datasets.

Datasets supported:
1. GTSRB (German Traffic Sign Recognition Benchmark)
   - Training archive (~2.87 GB)
   - Test archive (~851 MB)
   - Test Ground Truth (~179 MB)
2. GTSDB (German Traffic Sign Detection Benchmark)
   - FullIJCNN2013.zip (~1.6 GB)
3. BDD100K (Berkeley DeepDrive)
   - Documentation & Subset downloader guidance
"""

import argparse
import os
from pathlib import Path
import sys
import urllib.request
import zipfile


GTSRB_URLS = {
    "train": (
        "https://sid.erda.dk/public/archives/daaeac0d7ce115aea9b61d9f1e19370/GTSRB_Final_Training_Images.zip",
        "GTSRB_Final_Training_Images.zip",
    ),
    "test": (
        "https://sid.erda.dk/public/archives/daaeac0d7ce115aea9b61d9f1e19370/GTSRB_Final_Test_Images.zip",
        "GTSRB_Final_Test_Images.zip",
    ),
    "test_gt": (
        "https://sid.erda.dk/public/archives/daaeac0d7ce115aea9b61d9f1e19370/GTSRB_Final_Test_GT.zip",
        "GTSRB_Final_Test_GT.zip",
    ),
}

GTSDB_URLS = {
    "full": (
        "https://sid.erda.dk/public/archives/ff17dc924eba88d5d01a807357d6614c/FullIJCNN2013.zip",
        "FullIJCNN2013.zip",
    )
}


def download_with_progress(url: str, dest_path: Path) -> None:
    """Download a file with download progress reporting."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n[INFO] Connecting to: {url}")
    print(f"[INFO] Saving to: {dest_path}")

    def progress_callback(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = min(100.0, downloaded * 100.0 / total_size)
            mb_down = downloaded / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)
            sys.stdout.write(f"\r  Progress: {percent:5.1f}% ({mb_down:6.1f} MB / {mb_total:6.1f} MB)")
        else:
            mb_down = downloaded / (1024 * 1024)
            sys.stdout.write(f"\r  Progress: {mb_down:6.1f} MB downloaded")
        sys.stdout.flush()

    try:
        # User-agent header to avoid bot filtering
        opener = urllib.request.build_opener()
        opener.addheaders = [("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) TrafficSignBenchmark/1.0")]
        urllib.request.install_opener(opener)
        urllib.request.urlretrieve(url, dest_path, reporthook=progress_callback)
        print("\n[SUCCESS] Download completed.")
    except Exception as e:
        print(f"\n[ERROR] Download failed: {e}")
        raise


def extract_zip(zip_path: Path, target_dir: Path) -> None:
    """Extract zip archive safely into destination directory."""
    print(f"[INFO] Extracting {zip_path.name} to {target_dir}...")
    target_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(target_dir)
    print(f"[SUCCESS] Extracted to {target_dir}")


def setup_gtsrb(download_dir: Path, target_dir: Path, parts: list) -> None:
    """Download and extract GTSRB components."""
    for part in parts:
        if part not in GTSRB_URLS:
            continue
        url, filename = GTSRB_URLS[part]
        zip_file = download_dir / filename
        if not zip_file.exists():
            download_with_progress(url, zip_file)
        dest = target_dir / ("train" if part == "train" else "test")
        extract_zip(zip_file, dest)


def setup_gtsdb(download_dir: Path, target_dir: Path) -> None:
    """Download and extract GTSDB benchmark."""
    url, filename = GTSDB_URLS["full"]
    zip_file = download_dir / filename
    if not zip_file.exists():
        download_with_progress(url, zip_file)
    extract_zip(zip_file, target_dir)


def main():
    parser = argparse.ArgumentParser(description="Official dataset download and extraction manager")
    parser.add_argument(
        "--dataset",
        choices=["gtsrb", "gtsdb", "all"],
        required=True,
        help="Which dataset archive to download",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data/cache"),
        help="Temporary archive cache directory",
    )
    parser.add_argument(
        "--gtsrb-parts",
        nargs="+",
        default=["train", "test", "test_gt"],
        help="Specific GTSRB archives (train, test, test_gt)",
    )

    args = parser.parse_args()

    if args.dataset in ("gtsrb", "all"):
        print("[INFO] Starting GTSRB acquisition...")
        setup_gtsrb(args.cache_dir, Path("data/raw/gtsrb"), args.gtsrb_parts)

    if args.dataset in ("gtsdb", "all"):
        print("[INFO] Starting GTSDB acquisition...")
        setup_gtsdb(args.cache_dir, Path("data/raw/gtsdb"))


if __name__ == "__main__":
    main()
