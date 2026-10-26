# Traffic Sign Recognition & Road Safety Dataset Guide

This document describes dataset sources, download instructions, licensing policies, annotation standards, directory structure, and splitting procedures for model training.

---

## ⚠️ Critical Licensing & Upload Policy

> [!WARNING]
> **Do not upload a dataset until you have checked its license.**
>
> Many public datasets are licensed strictly for non-commercial research or academic competition (e.g. GTSRB, GTSDB, BDD100K). Do not commit raw or processed datasets, proprietary driving videos, or large model weights to this Git repository.
>
> All image archives, raw images, and binary masks inside `data/raw/`, `data/processed/`, `data/images/`, and `data/labels/` are automatically excluded by [`.gitignore`](file:///c:/Users/HP/Desktop/Projects/traffic-sign-recognition-road-safety/.gitignore). Dataset documentation, manifests, and conversion scripts remain tracked.

---

## 1. Dataset Directory Organization

The repository structures raw and processed data into clean directories:

```text
data/
├── README.md               # Dataset documentation and instructions
├── traffic_signs.yaml      # Ultralytics YOLO dataset configuration file
├── create_sample_dataset.py# Synthetic starter dataset generator
├── convert_gtsdb_to_yolo.py# GTSDB to YOLO format converter with class mapping
├── sample_bdd100k.py       # BDD100K reproducible sampling & validation script
├── download_datasets.py    # Official archive download & extraction utility
│
├── raw/                    # Original uncompressed datasets
│   ├── gtsrb/              # German Traffic Sign Recognition Benchmark
│   │   ├── train/          # Preserved original class folders (00000 to 00042)
│   │   ├── test/           # Test images and GT-final_test.csv
│   │   └── README.md       # GTSRB metadata, URL, license, and resolution notes
│   ├── gtsdb/              # German Traffic Sign Detection Benchmark
│   │   ├── images/         # 900 high-res road images (00000.ppm to 00899.ppm)
│   │   ├── annotations/    # Original gt.txt and documentation
│   │   └── README.md       # GTSDB source, license, limitations, and class mappings
│   └── bdd100k/            # Berkeley DeepDrive Drivable Area Subset
│       ├── images/         # 1,000 dashcam frames (1280x720)
│       ├── drivable_masks/ # 1,000 8-bit PNG segmentation masks (1280x720)
│       ├── selected_images.csv # Reproducible sampling manifest (seed=42)
│       └── README.md       # BDD100K taxonomy and sampling protocol
│
├── processed/              # Formatted/converted data ready for training
│   └── gtsdb/
│       └── labels/         # Converted YOLO format labels
├── images/                 # Active YOLO training/validation image splits
│   ├── train/
│   └── val/
└── labels/                 # Active YOLO training/validation label splits
    ├── train/
    └── val/
```

---

## 2. Guidelines for Downloading and Organizing Datasets

### A. GTSRB (Traffic Sign Recognition Benchmark)
- **Official URL**: [http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset)
- **Archive Mirror**: [ERDA Public Mirror](https://sid.erda.dk/public/archives/daaeac0d7ce115aea9b61d9f1e19370/published-archive.html)
- **Archives**: `GTSRB_Final_Training_Images.zip`, `GTSRB_Final_Test_Images.zip`, `GTSRB_Final_Test_GT.zip`
- **Location**: `data/raw/gtsrb/`
- **Preservation**: Keep original class directories (`00000/` through `00042/`) and `GT-*.csv` files intact.
- **Resolution Considerations**: GTSRB contains 43 sign classes with image crops ranging from $15 \times 15$ to $250 \times 250$ px. Small signs suffer from motion blur; consider resizing and CLAHE preprocessing for classification tasks.

### B. GTSDB (Traffic Sign Detection Benchmark)
- **Official URL**: [http://benchmark.ini.rub.de/?section=gtsdb&subsection=dataset](http://benchmark.ini.rub.de/?section=gtsdb&subsection=dataset)
- **Archive Mirror**: [ERDA FullIJCNN2013 Archive](https://sid.erda.dk/public/archives/ff17dc924eba88d5d01a807357d6614c/FullIJCNN2013.zip)
- **Location**: `data/raw/gtsdb/` (`images/`, `annotations/gt.txt`)
- **Important Limitation & Class Mapping**:
  GTSDB contains 900 images and its categories do not 1:1 match our project's 3 classes. Signs must not be arbitrarily relabeled based on visual similarity:
  - **`0: stop`**: GTSDB Class 14 (Stop sign)
  - **`1: speed`**: GTSDB Classes 0, 1, 2, 3, 4, 5, 7, 8 (20, 30, 50, 60, 70, 80, 100, 120 km/h)
  - **`2: danger`**: GTSDB Classes 11, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31 (danger/warning signs)
  - *Non-target signs* (classes 6, 9, 10, 12, 13, 15, 16, 17, 32-42) are excluded during conversion.
- **Conversion to YOLO**:
  ```powershell
  python data/convert_gtsdb_to_yolo.py --annotations data/raw/gtsdb/annotations/gt.txt --output-dir data/processed/gtsdb/labels
  ```

### C. BDD100K (1,000 Images and Drivable Area Masks)
- **Official URL**: [https://bdd-data.berkeley.edu/](https://bdd-data.berkeley.edu/)
- **Mirror**: [Hugging Face BDD100K Drivable Area](https://huggingface.co/datasets/dronefreak/BDD100K-Drivable-Area-Segmentation)
- **Location**: `data/raw/bdd100k/` (`images/`, `drivable_masks/`, `selected_images.csv`)
- **Nature of Data**: Drivable-area masks are semantic segmentation masks ($1280 \times 720$ PNG), **not** traffic-sign bounding boxes.
- **Mask Labels**:
  - `0`: Direct drivable area (ego lane)
  - `1`: Alternative drivable area (adjacent lane)
  - `2`: Background / non-drivable
- **Reproducible Sampling**:
  Sampled using fixed random seed (`seed = 42`) across lighting (daytime, night, dawn/dusk), weather (clear, rainy, overcast), and scene types with 800 train / 200 val split and zero sequence leakage.
- **Generate Manifest & Verify**:
  ```powershell
  python data/sample_bdd100k.py --count 1000 --seed 42
  ```

---

## 3. Sequence-Based Data Splitting Guidelines

> [!IMPORTANT]
> **Always split continuous video and dashcam data by recording or source sequence, NOT by random individual frames.**
>
> 1. **Data Leakage Risk**: Video dashcams record at 30 FPS. Consecutive frames $(t \text{ and } t+1)$ are nearly identical. If you randomly shuffle frames, the model will "memorize" the scene, yielding artificially high test accuracy that collapses in real-world driving.
> 2. **Sequence Partitioning**: Group all frames from the same driving run or track sequence together. Place the entire sequence into *either* `train/` or `val/`, never across both.
