# German Traffic Sign Recognition Benchmark (GTSRB)

## 1. Dataset Overview & Source Metadata
- **Dataset Name**: German Traffic Sign Recognition Benchmark (GTSRB)
- **Primary Source URL**: [http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset)
- **Public Mirror Archive (ERDA)**: [https://sid.erda.dk/public/archives/daaeac0d7ce115aea9b61d9f1e19370/published-archive.html](https://sid.erda.dk/public/archives/daaeac0d7ce115aea9b61d9f1e19370/published-archive.html)
- **Download Date**: October 2026
- **Archive Filenames**:
  - Training Set: `GTSRB_Final_Training_Images.zip` (~2.87 GB, 39,209 images across 43 classes)
  - Test Set: `GTSRB_Final_Test_Images.zip` (~851 MB, 12,630 images)
  - Test Ground Truth: `GTSRB_Final_Test_GT.zip` (~179 MB, `GT-final_test.csv`)
- **License**: Free for academic, educational, and research use (Institut für Neuroinformatik, Ruhr-Universität Bochum / University of Copenhagen).

---

## 2. Directory Layout & Organization
The archives are extracted into the following structure, strictly preserving original class subdirectories and annotation CSV files:

```
data/raw/gtsrb/
├── train/
│   ├── 00000/                # Class 00000: Speed limit (20km/h)
│   │   ├── 00000_00000.ppm
│   │   ├── ...
│   │   └── GT-00000.csv      # Bounding box and label annotations
│   ├── 00001/                # Class 00001: Speed limit (30km/h)
│   ├── ...
│   └── 00042/                # Class 00042: End of no passing by vehicles over 3.5 metric tons
├── test/
│   ├── 00000.ppm
│   ├── ...
│   └── GT-final_test.csv     # Test set ground truth
└── README.md
```

---

## 3. Class Definitions & Resolution Inspection
GTSRB contains **43 discrete traffic sign classes**.

### Important Inspection & Image Resolution Notes
1. **Variable and Small Image Resolutions**:
   - Signs range from **$15 \times 15$ pixels** up to **$250 \times 250$ pixels**.
   - Median resolution is approximately $32 \times 32$ pixels.
   - Low-resolution crops (e.g., $15 \times 15$ px) contain heavy motion blur, low contrast, and sensor noise.
2. **Task Applicability**:
   - GTSRB consists of **cropped image patches** centered on traffic signs with a 10% surrounding border.
   - It is designed as a **classification benchmark** (crop $\to$ class label).
   - For **full-scene object detection** (predicting bounding boxes from dashcam frames), use **GTSDB** (`data/raw/gtsdb/`) rather than GTSRB directly.
3. **Preprocessing Recommendations**:
   - Normalize and resize crops to a uniform input resolution (e.g., $32 \times 32$ or $64 \times 64$ for lightweight classifiers).
   - Apply contrast-limited adaptive histogram equalization (CLAHE) to mitigate uneven daylight/shadow illumination.
