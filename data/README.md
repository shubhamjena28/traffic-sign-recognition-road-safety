# Traffic Sign Recognition Dataset Guide

This document describes dataset sources, download instructions, licensing policies, annotation standards, directory structure, and splitting procedures for model training.

---

## ⚠️ Critical Licensing & Upload Policy

> [!WARNING]
> **Do not upload a dataset until you have checked its license.**
>
> Many public datasets are licensed strictly for non-commercial research or academic competition (e.g. GTSRB). Do not commit raw or processed datasets, proprietary driving videos, or large model weights to this Git repository.
>
> All image and label files inside `data/images/`, `data/labels/`, `data/raw/`, `data/processed/`, and `data/annotations/` (except `.gitkeep`) are automatically ignored by [`.gitignore`](file:///c:/Users/HP/Desktop/Projects/traffic-sign-recognition-road-safety/.gitignore).

---

## 1. Where to Obtain the Dataset

### 1.1 German Traffic Sign Recognition Benchmark (GTSRB)
The industry standard dataset for multi-class traffic sign classification:
- **Official Benchmark**: [https://benchmark.ini.rub.de/gtsrb_dataset.html](https://benchmark.ini.rub.de/gtsrb_dataset.html)
- **Direct Mirror**: [Ruhr-Universität Bochum INI Archive](https://sid.erda.dk/public/archives/daaeac0d7ce1152cedf83789b6da7187/published-archive.html)
- **Kaggle Mirror**: [GTSRB German Traffic Sign Dataset](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign)

### 1.2 Roboflow Traffic Sign Detection Datasets
Pre-annotated YOLO detection datasets featuring `stop`, `speed`, and `danger` signs with bounding boxes:
- **Roboflow Universe**: [Traffic Sign Detection Datasets](https://universe.roboflow.com/search?q=traffic+sign)

---

## 2. Dataset License & Attribution

- **License**: GTSRB is provided free of charge for academic, research, and educational purposes.
- **Attribution Citation**:
  ```text
  J. Stallkamp, M. Schlipsing, J. Salmen, and C. Igel.
  "The German Traffic Sign Recognition Benchmark: A multi-class classification competition."
  IEEE International Joint Conference on Neural Networks (IJCNN), 2011.
  ```

---

## 3. Directory Layout

The repository organizes data into two standard structures:

```text
data/
├── README.md               # Dataset documentation and instructions
├── traffic_signs.yaml      # Ultralytics YOLO dataset configuration file
├── create_sample_dataset.py# Automated synthetic starter dataset generator
│
├── images/                 # YOLO Object Detection image splits
│   ├── train/              # Training images (.jpg, .png)
│   └── val/                # Validation images (.jpg, .png)
│
├── labels/                 # YOLO Object Detection bounding box annotations
│   ├── train/              # Training labels (.txt)
│   └── val/                # Validation labels (.txt)
│
├── raw/                    # Original uncompressed dataset downloads (.zip, .tar)
│   └── .gitkeep
├── processed/              # Classification folders (0 to 42)
│   └── .gitkeep
└── annotations/            # Original CSV/XML track and bounding box files
    └── .gitkeep
```

---

## 4. Annotation Format (YOLO Bounding Box)

Each image in `images/` has a matching `.txt` file in `labels/` with normalized coordinates $[0.0, 1.0]$:

```text
<class_id> <center_x> <center_y> <width> <height>
```

### Classes Supported:
- `0`: **`stop`** (Stop signs)
- `1`: **`speed`** (Speed limit signs)
- `2`: **`danger`** (Hazard, warning, and caution signs)

**Example (`sample_0_000.txt`):**
```text
0 0.5000 0.5000 0.4375 0.4375
```

---

## 5. Sequence-Based Data Splitting Guidelines

> [!IMPORTANT]
> **Always split continuous video and dashcam data by recording or source sequence, NOT by random individual frames.**
>
> 1. **Data Leakage Risk**: Video dashcams record at 30 FPS. Consecutive frames $(t \text{ and } t+1)$ are nearly identical. If you randomly shuffle frames, the model will "memorize" the scene, yielding artificially high test accuracy that collapses in real-world driving.
> 2. **Sequence Partitioning**: Group all frames from the same driving run or track sequence together. Place the entire sequence into *either* `train/` or `val/`, never across both.

---

## 6. How to Prepare the Dataset

### Step 1: Generate Starter Dataset (Immediate Verification)
Run the built-in dataset generator to verify training instantly:
```bash
python data/create_sample_dataset.py
```

### Step 2: Download & Structure Real-World Data
1. Download image archives into `data/raw/`.
2. Extract and organize images into `data/images/train` and `data/images/val`.
3. Place matching YOLO `.txt` annotations into `data/labels/train` and `data/labels/val`.

### Step 3: Launch YOLO Training
```bash
yolo detect train data=data/traffic_signs.yaml model=yolov8n.pt epochs=30 imgsz=640
```
