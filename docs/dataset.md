# Dataset Specifications & Data Management

This document details dataset sources, licensing requirements, annotation formats, and best practices for dataset splitting.

---

## 1. Dataset Sources & Acquisition

The model is trained on multi-class road signage datasets:

1. **German Traffic Sign Recognition Benchmark (GTSRB)**:
   - Source: [Institut für Neuroinformatik, Ruhr-Universität Bochum](https://benchmark.ini.rub.de/)
   - Format: Over 50,000 physical image tracks with bounding box CSVs.
2. **Kaggle Traffic Sign Detection Mirrors**:
   - Source: [GTSRB Kaggle Mirror](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign)
3. **Roboflow Traffic Sign Detection Sets**:
   - Ready-to-train datasets exported directly in YOLOv8 format with `stop`, `speed`, and `danger` classes.

---

## 2. ⚠️ Licensing & Upload Verification Policy

> [!WARNING]
> **Do not upload a dataset until you have checked its license.**
> 
> - **License Terms**: GTSRB and academic benchmark datasets are free for non-commercial research and educational use.
> - **Redistribution Restrictions**: Many datasets prohibit redistributing the raw images in public repositories without permission.
> - **Git Policy**: Raw and processed image data must remain in your local `data/` directory and are automatically ignored by [`.gitignore`](file:///c:/Users/HP/Desktop/Projects/traffic-sign-recognition-road-safety/.gitignore).

---

## 3. Road Safety Classes

The object detection model maps signs to the primary safety classes used by `SafetyMonitor`:

| Class ID | Class Name | Sign Types Included | Safety Action in Java |
| :--- | :--- | :--- | :--- |
| `0` | **`stop`** | Octagonal STOP signs | `CRITICAL` alert to bring vehicle to a halt. |
| `1` | **`speed`** | Speed limits (20, 30, 50, 70, 80, 100, 120 km/h) | Dynamic speed tracking; `WARNING` / `CRITICAL` overspeed alerts. |
| `2` | **`danger`** | General caution, curves, pedestrian crossings, construction, ice/snow | `WARNING` alert to exercise caution. |

---

## 4. Annotation Format (YOLO Normalized Coordinates)

Each image in `data/images/{split}/` has an identically named `.txt` file in `data/labels/{split}/`.

Each line represents one sign bounding box:
```text
<class_id> <center_x> <center_y> <width> <height>
```
- All coordinates are normalized floats in the range $[0.0, 1.0]$.
- Example (`stop` sign in center of $640 \times 640$ frame):
  ```text
  0 0.5000 0.5000 0.1500 0.1500
  ```

---

## 5. Sequence-Based Data Splitting Guidelines

> [!IMPORTANT]
> **Split data by recording or source sequence (track-based splitting).**
>
> When working with dashcam or continuous camera datasets:
> - **The Problem**: Consecutive video frames are separated by only fractions of a second and look nearly identical. If you do a random image-level train/test split, virtually identical frames will appear in both sets, causing severe data leakage and artificially inflated accuracy.
> - **The Solution**: Group all images from the same driving sequence, vehicle pass, or recording clip together, and place the entire sequence into *either* the training set *or* the test set—never split across both.
