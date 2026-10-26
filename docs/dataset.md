# Dataset Specifications & Data Management

This document details dataset sources, licensing requirements, annotation formats, class mapping taxonomies, and best practices for dataset splitting across **GTSRB**, **GTSDB**, and **BDD100K**.

---

## 1. Dataset Sources & Acquisition

The system incorporates three specialized benchmark datasets:

1. **German Traffic Sign Recognition Benchmark (GTSRB)**:
   - **Type**: Multi-class traffic sign crop classification.
   - **Source**: [Institut für Neuroinformatik, Ruhr-Universität Bochum](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset) / [ERDA Public Mirror](https://sid.erda.dk/public/archives/daaeac0d7ce115aea9b61d9f1e19370/published-archive.html)
   - **Scale**: 39,209 training crops across 43 classes + 12,630 test images.
   - **Structure**: Preserves original class subdirectories (`00000/` through `00042/`) and `GT-*.csv` ground truth files.
   - **Resolution Considerations**: Sign patches range from $15 \times 15$ to $250 \times 250$ pixels. Low-resolution images feature motion blur and varying illumination; preprocessing with resizing and CLAHE is advised.

2. **German Traffic Sign Detection Benchmark (GTSDB)**:
   - **Type**: Full-scene traffic sign detection benchmark.
   - **Source**: [Ruhr-Universität Bochum](http://benchmark.ini.rub.de/?section=gtsdb&subsection=dataset) / [ERDA FullIJCNN2013 Archive](https://sid.erda.dk/public/archives/ff17dc924eba88d5d01a807357d6614c/FullIJCNN2013.zip)
   - **Scale**: 900 high-resolution images ($1360 \times 1024$ pixels, 600 train / 300 test) with ~1,200 signs.
   - **Format**: Semicolon-delimited bounding boxes (`gt.txt`: `Filename;LeftCol;TopRow;RightCol;BottomRow;ClassID`).
   - **Converter**: Use [`data/convert_gtsdb_to_yolo.py`](file:///c:/Users/HP/Desktop/Projects/traffic-sign-recognition-road-safety/data/convert_gtsdb_to_yolo.py) to generate YOLO formatted labels while preserving original ground truth files.

3. **Berkeley DeepDrive (BDD100K) Drivable Area Subset**:
   - **Type**: Autonomous driving road-scene understanding & drivable-area semantic segmentation.
   - **Source**: [UC Berkeley BAIR](https://bdd-data.berkeley.edu/) / [Hugging Face Repository](https://huggingface.co/datasets/dronefreak/BDD100K-Drivable-Area-Segmentation)
   - **Scale**: Representative subset of 1,000 dashcam images ($1280 \times 720$ JPEG) and matching masks ($1280 \times 720$ 8-bit PNG).
   - **Important Domain Note**: Drivable-area masks are **semantic segmentation masks**, NOT traffic-sign bounding boxes.
   - **Mask Taxonomy**:
     - `0`: Direct drivable area (ego vehicle lane)
     - `1`: Alternative drivable area (adjacent lane)
     - `2`: Background / non-drivable
   - **Sampling Manifest**: Exported to [`data/raw/bdd100k/selected_images.csv`](file:///c:/Users/HP/Desktop/Projects/traffic-sign-recognition-road-safety/data/raw/bdd100k/selected_images.csv) using reproducible seed `42`.

---

## 2. ⚠️ Licensing & Upload Verification Policy

> [!WARNING]
> **Do not upload a dataset until you have checked its license.**
> 
> - **License Terms**: GTSRB, GTSDB, and BDD100K are licensed for non-commercial research and educational use.
> - **Redistribution Restrictions**: Many datasets prohibit redistributing raw images in public repositories without permission.
> - **Git Policy**: Raw and processed image binaries (`*.ppm`, `*.jpg`, `*.png`, `*.zip`) are ignored by [`.gitignore`](file:///c:/Users/HP/Desktop/Projects/traffic-sign-recognition-road-safety/.gitignore). Dataset documentation, manifests, and conversion scripts remain tracked.

---

## 3. Road Safety Classes & Explicit Mapping

The object detection model maps signs to the primary safety classes used by `SafetyMonitor`:

| Project Target Class | Class ID | GTSDB Class IDs Included | Sign Types Included | Java Safety Action |
| :--- | :---: | :--- | :--- | :--- |
| **`stop`** | `0` | `14` | Octagonal STOP signs | `CRITICAL` alert to bring vehicle to a halt. |
| **`speed`** | `1` | `0, 1, 2, 3, 4, 5, 7, 8` | Speed limits (20, 30, 50, 60, 70, 80, 100, 120 km/h) | Dynamic speed tracking; `WARNING` / `CRITICAL` overspeed alerts. |
| **`danger`** | `2` | `11, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31` | Caution, curves, pedestrian crossings, road work, ice/snow | `WARNING` alert to exercise caution. |
| **Excluded** | *None* | `6, 9, 10, 12, 13, 15, 16, 17, 32-42` | End of speed limits, no overtaking, mandatory arrows, parking | Ignored to maintain training data purity. |

---

## 4. Annotation Format (YOLO Normalized Coordinates)

Each image in `data/images/{split}/` has an identically named `.txt` file in `data/labels/{split}/`.

Each line represents one sign bounding box:
```text
<class_id> <center_x> <center_y> <width> <height>
```
- All coordinates are normalized floats in the range $[0.0, 1.0]$.
- Coordinate transformation formula from $[x_1, y_1, x_2, y_2]$:
  $$\text{center\_x} = \frac{x_1 + x_2}{2 \times W}, \quad \text{center\_y} = \frac{y_1 + y_2}{2 \times H}$$
  $$\text{width} = \frac{x_2 - x_1}{W}, \quad \text{height} = \frac{y_2 - y_1}{H}$$

---

## 5. Sequence-Based Data Splitting Guidelines

> [!IMPORTANT]
> **Split data by recording or source sequence (track-based splitting).**
>
> When working with dashcam or continuous camera datasets:
> - **The Problem**: Consecutive video frames are separated by only fractions of a second and look nearly identical. If you do a random image-level train/test split, virtually identical frames will appear in both sets, causing severe data leakage and artificially inflated accuracy.
> - **The Solution**: Group all images from the same driving sequence, vehicle pass, or recording clip together, and place the entire sequence into *either* the training set *or* the test set—never split across both.
