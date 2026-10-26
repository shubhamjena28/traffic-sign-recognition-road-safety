# German Traffic Sign Detection Benchmark (GTSDB)

## 1. Dataset Overview & Source Metadata
- **Dataset Name**: German Traffic Sign Detection Benchmark (GTSDB)
- **Primary Source URL**: [http://benchmark.ini.rub.de/?section=gtsdb&subsection=dataset](http://benchmark.ini.rub.de/?section=gtsdb&subsection=dataset)
- **Public Mirror Archive (ERDA)**: [https://sid.erda.dk/public/archives/ff17dc924eba88d5d01a807357d6614c/FullIJCNN2013.zip](https://sid.erda.dk/public/archives/ff17dc924eba88d5d01a807357d6614c/FullIJCNN2013.zip)
- **Download Date**: October 2026
- **Archive Filename**: `FullIJCNN2013.zip` (~1.6 GB)
- **Total Images**: 900 high-resolution road scenes ($1360 \times 1024$ pixels) in PPM format (600 training, 300 test) containing ~1,200 traffic sign instances.
- **License**: Free for academic, educational, and research use (Institut für Neuroinformatik, Ruhr-Universität Bochum).

---

## 2. Directory Layout & Organization
Extracted without altering original annotation files:

```
data/raw/gtsdb/
├── images/                   # Full-resolution road images (00000.ppm - 00899.ppm)
├── annotations/              # Preserved original ground-truth files
│   ├── gt.txt                # Semicolon-delimited bounding boxes and class IDs
│   └── Readme.txt            # Original competition documentation
└── README.md
```

---

## 3. Original Annotation Format
The ground truth file `gt.txt` stores semicolon-separated annotations:
```text
Filename;LeftCol;TopRow;RightCol;BottomRow;ClassID
00000.ppm;774;411;815;451;11
00001.ppm;983;388;1024;432;40
00002.ppm;386;494;442;552;38
```
- Coordinates are in absolute pixel space: `(LeftCol, TopRow)` = top-left $(x_1, y_1)$, `(RightCol, BottomRow)` = bottom-right $(x_2, y_2)$.
- `ClassID` corresponds to the 43 GTSRB classification IDs ($0 \dots 42$).

---

## 4. Important Dataset Limitations & Target Class Mapping

### Critical Limitations
1. **Small Dataset Scale**: GTSDB contains only 900 images total, with certain rare sign classes appearing fewer than 10 times.
2. **Category Mismatch**: GTSDB was designed around 4 competition super-classes (prohibitory, danger, mandatory, other), which do **not** directly correspond to our project's 3 target classes: **`stop`**, **`speed`**, and **`danger`**.
3. **Strict Mapping Rule**: Do **not** arbitrarily relabel signs based only on superficial visual similarity. Non-target signs (e.g., Class 6 "End of speed limit 80", Class 9 "No passing", Class 32 "End of all speed and passing limits", mandatory blue arrows, or parking signs) must **not** be relabeled as target signs. Instead, unmapped signs are safely discarded during conversion to maintain training data purity.

### Official Class Mapping to Project Targets

| Project Target Class | Target Class ID | GTSDB Class IDs Included | GTSDB Sign Descriptions |
| :--- | :--- | :--- | :--- |
| **`stop`** | `0` | `14` | Stop sign (Octagonal red sign) |
| **`speed`** | `1` | `0, 1, 2, 3, 4, 5, 7, 8` | Speed limits: 20 km/h (`0`), 30 km/h (`1`), 50 km/h (`2`), 60 km/h (`3`), 70 km/h (`4`), 80 km/h (`5`), 100 km/h (`7`), 120 km/h (`8`) |
| **`danger`** | `2` | `11, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31` | Warning/danger signs: Right-of-way (`11`), General danger (`18`), Dangerous curve left (`19`), Dangerous curve right (`20`), Double curve (`21`), Bumpy road (`22`), Slippery road (`23`), Road narrows right (`24`), Road work (`25`), Traffic signals (`26`), Pedestrians (`27`), Children crossing (`28`), Bicycles (`29`), Ice/snow (`30`), Wild animals (`31`) |
| **Excluded / Unmapped** | *Ignored* | `6, 9, 10, 12, 13, 15, 16, 17, 32-42` | End of speed limits, no overtaking, priority road diamond, mandatory direction arrows, end of restrictions |

---

## 5. Converting to YOLO Format
Use [`data/convert_gtsdb_to_yolo.py`](file:///c:/Users/HP/Desktop/Projects/traffic-sign-recognition-road-safety/data/convert_gtsdb_to_yolo.py) to convert `gt.txt` into YOLO format while preserving original files:

```powershell
python data/convert_gtsdb_to_yolo.py --annotations data/raw/gtsdb/annotations/gt.txt --images data/raw/gtsdb/images --output-dir data/processed/gtsdb
```

### YOLO Bounding Box Coordinate Formula
Given image width $W = 1360$ and height $H = 1024$:
$$\text{x\_center} = \frac{x_1 + x_2}{2 \cdot W}, \quad \text{y\_center} = \frac{y_1 + y_2}{2 \cdot H}$$
$$\text{width} = \frac{x_2 - x_1}{W}, \quad \text{height} = \frac{y_2 - y_1}{H}$$
All bounding box values are normalized to $[0.0, 1.0]$.
