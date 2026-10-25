# Traffic Sign Recognition Dataset Guide

This document describes how to obtain, verify, and prepare the dataset for training and evaluating the traffic sign recognition models.

---

## ⚠️ Important Repository Policy

> [!WARNING]
> **Do not commit large datasets, private data, or model weights (`*.pt`, `*.pth`, `*.onnx`, `*.weights`) to this Git repository.**
> 
> All files inside `data/raw/`, `data/processed/`, and `data/annotations/` (except `.gitkeep`) are automatically ignored by `.gitignore`.
> 
> **For reproducibility**, trained model checkpoints should be published using **GitHub Releases**, **Hugging Face Hub**, or an external object store rather than committed into Git.

---

## 1. Where to Obtain the Dataset

The primary benchmark dataset supported by this project is the **German Traffic Sign Recognition Benchmark (GTSRB)**:

- **Official Website**: [https://benchmark.ini.rub.de/gtsrb_dataset.html](https://benchmark.ini.rub.de/gtsrb_dataset.html)
- **Direct Download Mirror**: [Institut für Neuroinformatik, Ruhr-Universität Bochum](https://sid.erda.dk/public/archives/daaeac0d7ce1152cedf83789b6da7187/published-archive.html)
- **Kaggle Dataset Mirror**: [GTSRB - German Traffic Sign Dataset](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign)

### Dataset Overview
- **Number of Classes**: 43 traffic sign classes (speed limits, danger warnings, mandatory directions, prohibitory signs).
- **Total Images**: Over 50,000 physical images captured in varying daylight, weather, and camera perspectives.
- **Image Resolutions**: Ranging from $15 \times 15$ to $250 \times 250$ pixels.

---

## 2. Dataset License & Attribution

- **License**: The GTSRB dataset is provided free of charge for non-commercial research, academic, and educational purposes.
- **Attribution**:
  ```text
  J. Stallkamp, M. Schlipsing, J. Salmen, and C. Igel.
  "The German Traffic Sign Recognition Benchmark: A multi-class classification competition."
  IEEE International Joint Conference on Neural Networks (IJCNN), 2011.
  ```

---

## 3. Directory Layout

The project expects dataset assets to be organized under `data/`:

```text
data/
├── README.md               # This documentation file
│
├── raw/                    # Downloaded unextracted zip/tar archives and raw extracts
│   └── .gitkeep
│
├── processed/              # Formatted into class folders ready for ImageFolder loader
│   ├── 0/                  # Class 0: Speed limit (20km/h)
│   ├── 1/                  # Class 1: Speed limit (30km/h)
│   ├── ...
│   ├── 42/                 # Class 42: End of no passing by vehicles over 3.5 metric tons
│   └── .gitkeep
│
└── annotations/            # CSV / JSON bounding box ground-truth and track files
    ├── Train.csv
    ├── Test.csv
    └── .gitkeep
```

---

## 4. How to Prepare the Dataset

### Step 1: Download the Archives
Download the GTSRB training and test archives:
```bash
# Example using curl / wget into data/raw/
cd data/raw
curl -LO https://sid.erda.dk/public/archives/daaeac0d7ce1152cedf83789b6da7187/GTSRB_Final_Training_Images.zip
curl -LO https://sid.erda.dk/public/archives/daaeac0d7ce1152cedf83789b6da7187/GTSRB_Final_Test_Images.zip
curl -LO https://sid.erda.dk/public/archives/daaeac0d7ce1152cedf83789b6da7187/GTSRB_Final_Test_GT.zip
```

### Step 2: Extract Images
Extract into `data/processed/` such that each class directory contains the PNG/PPM images for that class ID:
```bash
unzip GTSRB_Final_Training_Images.zip -d ../processed/
```

Verify directory layout:
```bash
# Each class folder (0 to 42) should contain its image files
ls -l data/processed/
```

### Step 3: Run Training
With images in `data/processed/`, launch training from the repository root:
```bash
cd train
python train.py --data-dir ../data/processed --epochs 25 --batch-size 64
```

---

## 5. Model Weights & Reproducibility

To ensure experiment reproducibility without bloating the repository:
1. Export model weights using:
   ```bash
   python train/export_model.py --checkpoint train/checkpoints/best_model.pt
   ```
2. Upload the exported `best_model.pt` and `traffic_sign_model.onnx` artifacts to a **GitHub Release tag** (e.g. `v1.0.0`) on your repository:
   - Go to `https://github.com/shubhamjena28/traffic-sign-recognition-road-safety/releases/new`
   - Attach your model weights and link them in project documentation.
