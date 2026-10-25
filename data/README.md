# Traffic Sign Recognition Dataset Guide

This directory holds dataset assets for training, validating, and testing traffic sign recognition models.

## Directory Structure

```text
data/
├── raw/            # Original uncompressed dataset downloads
├── processed/      # Normalized, split into class-labeled subfolders (0-42)
├── annotations/    # Ground-truth annotations (CSV/XML/JSON bounding boxes)
└── README.md       # Dataset documentation and instructions
```

## Recommended Dataset: GTSRB (German Traffic Sign Recognition Benchmark)

The German Traffic Sign Recognition Benchmark (GTSRB) is the industry-standard benchmark for multi-class traffic sign classification:
- **43 distinct traffic sign classes** (speed limits, prohibitions, warnings, mandatory signs).
- **Over 50,000 images** in total across diverse lighting, weather, and motion blur conditions.
- Images sizes vary from 15x15 to 250x250 pixels.

### Download Instructions

1. Download the GTSRB dataset from official or benchmark mirrors:
   - Official Benchmark: [https://benchmark.ini.rub.de/gtsrb_dataset.html](https://benchmark.ini.rub.de/gtsrb_dataset.html)
   - Kaggle Mirror: [GTSRB - German Traffic Sign Dataset](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign)

2. Extract training images to `data/raw/` or organize directly into `data/processed/`:
   ```text
   data/processed/
   ├── 0/
   │   ├── 00000.png
   │   └── ...
   ├── 1/
   │   └── ...
   └── 42/
   ```

3. Place label mapping and bounding box CSV annotations into `data/annotations/`.

## Data Preprocessing Pipeline

Run preprocessing scripts or use `train/train.py` directly with `torchvision.datasets.ImageFolder` on `data/processed/`. Images are automatically resized to 48x48 and normalized before feeding into neural networks.
