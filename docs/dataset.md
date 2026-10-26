# Dataset Card: Traffic Sign Recognition and Road Safety Monitor

## 1. Project Information
- **Project**: Traffic Sign Recognition and Road Safety Monitor
- **Domain**: Intelligent Transport Systems (ITS) & Autonomous Driving Perception
- **Purpose**: Traffic-sign detection, classification, and road-scene analysis for real-time safety monitoring.

---

## 2. Dataset Sources

### 2.1 German Traffic Sign Recognition Benchmark (GTSRB)
- **Source**: [https://benchmark.ini.rub.de/gtsrb_dataset.html](https://benchmark.ini.rub.de/gtsrb_dataset.html)
- **Archive Mirror**: [ERDA Public Repository](https://sid.erda.dk/public/archives/daaeac0d7ce115aea9b61d9f1e19370/published-archive.html)
- **Content**: 51,839 cropped traffic-sign images across 43 classes.
  - **Training Split**: 39,209 images organized in 43 class folders (`00000` to `00042`).
  - **Test Split**: 12,630 test images.
- **Purpose**: Training and evaluating traffic-sign classification models.
- **Annotations**: Class IDs and bounding box coordinates within crops stored in `GT-*.csv` files.
- **Limitation**: Images primarily contain individual traffic-sign crops ($15 \times 15$ to $250 \times 250$ pixels) rather than full dashcam road scenes.

### 2.2 German Traffic Sign Detection Benchmark (GTSDB)
- **Source**: [https://benchmark.ini.rub.de/gtsdb_dataset.html](https://benchmark.ini.rub.de/gtsdb_dataset.html)
- **Archive Mirror**: [ERDA FullIJCNN2013 Archive](https://sid.erda.dk/public/archives/ff17dc924eba88d5d01a807357d6614c/FullIJCNN2013.zip)
- **Content**: 900 high-resolution full-scene road images ($1360 \times 1024$ pixels in PPM format) containing ~1,200 traffic-sign instances.
  - **Training Split**: 600 images (`00000.ppm` to `00599.ppm`).
  - **Evaluation Split**: 300 images (`00600.ppm` to `00899.ppm`).
- **Purpose**: Traffic-sign localization and object detection in real-world driving environments.
- **Annotations**: Bounding box coordinates and class IDs stored in `gt.txt` (`Filename;LeftCol;TopRow;RightCol;BottomRow;ClassID`).
- **Limitation**: The dataset is relatively small (900 scenes), and its competition categories do not completely match this project's target classes. Signs must not be arbitrarily relabeled based only on visual similarity; non-target signs are excluded from target annotations.

### 2.3 Berkeley DeepDrive (BDD100K)
- **Source**: [https://github.com/bdd100k/bdd100k](https://github.com/bdd100k/bdd100k) / [https://bdd-data.berkeley.edu/](https://bdd-data.berkeley.edu/)
- **Repository Mirror**: [Hugging Face BDD100K Drivable Area](https://huggingface.co/datasets/dronefreak/BDD100K-Drivable-Area-Segmentation)
- **Content**: Diverse real-world driving video frames and semantic segmentation masks.
- **Selected Subset**: Exactly **1,000 images** ($1280 \times 720$ JPEG) and matching drivable-area masks ($1280 \times 720$ 8-bit single-channel PNG).
  - **Training Split**: 800 images (80%).
  - **Validation Split**: 200 images (20%).
  - **Sequence Count**: 717 unique video sequences with zero cross-split sequence leakage.
- **Purpose**: Road-scene analysis, drivable-area segmentation, and spatial context awareness for ego-vehicle path planning.
- **Annotations**: Pixel-level drivable-area masks:
  - `0`: Direct drivable area (current travel lane)
  - `1`: Alternative drivable area (adjacent lane reachable via lane change)
  - `2`: Background / non-drivable
- **Limitation**: Drivable-area masks provide road surface navigability and do not contain traffic-sign bounding boxes.

---

## 3. Licensing and Usage
The original dataset providers retain all applicable rights and licensing terms:

- **GTSRB**:
  - **Provider**: Ruhr-Universität Bochum (Institut für Neuroinformatik).
  - **Terms**: Free for non-commercial academic research and educational evaluation. Commercial redistribution or incorporation into commercial products without prior authorization is prohibited. Attribution required (Stallkamp et al., 2011).
- **GTSDB**:
  - **Provider**: Ruhr-Universität Bochum.
  - **Terms**: Provided free for academic competition and research purposes. Commercial redistribution without permission is strictly prohibited. Attribution required (Houben et al., 2013).
- **BDD100K**:
  - **Provider**: University of California, Berkeley (Berkeley Artificial Intelligence Research - BAIR).
  - **Terms**: Governed by the BDD100K License Agreement. Permitted strictly for non-commercial academic research and educational purposes with appropriate copyright and attribution notices (Yu et al., 2020). Commercial use, sublicensing, and open redistribution of raw video frames are restricted.

> [!WARNING]
> Dataset availability does not automatically authorize unrestricted redistribution or commercial use. Verify the applicable terms before sharing data or derived datasets publicly. Raw image archives, binary images, and pixel masks are kept out of Git version control via [`.gitignore`](file:///c:/Users/HP/Desktop/Projects/traffic-sign-recognition-road-safety/.gitignore).

---

## 4. Data Preparation
1. **Inspection**:
   - Datasets are checked for corrupted image files, missing annotation entries, duplicate frames, and dimensional mismatches between images and masks.
2. **Annotation Conversion**:
   - Detection annotations in GTSDB (`gt.txt`) are converted to YOLO normalized format (`<class_id> <x_center> <y_center> <width> <height>`) using [`data/convert_gtsdb_to_yolo.py`](file:///c:/Users/HP/Desktop/Projects/traffic-sign-recognition-road-safety/data/convert_gtsdb_to_yolo.py).
   - Coordinates are normalized to $[0.0, 1.0]$ based on the $1360 \times 1024$ native frame dimensions.
   - GTSDB categories are mapped to project targets:
     - **`0: stop`** $\leftarrow$ GTSDB Class 14 (Stop sign)
     - **`1: speed`** $\leftarrow$ GTSDB Classes 0, 1, 2, 3, 4, 5, 7, 8 (20 to 120 km/h)
     - **`2: danger`** $\leftarrow$ GTSDB Classes 11, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31 (warning/hazard signs)
     - Non-target signs (classes 6, 9, 10, 12, 13, 15, 16, 17, 32–42) are excluded to prevent label corruption.
3. **BDD100K Pairing & Verification**:
   - 1,000 images are matched 1:1 with corresponding drivable-area mask PNG files using [`data/sample_bdd100k.py`](file:///c:/Users/HP/Desktop/Projects/traffic-sign-recognition-road-safety/data/sample_bdd100k.py).
   - Dimensions are validated ($1280 \times 720$), and mask pixel values are verified to be a subset of $\{0, 1, 2\}$.
4. **Preservation of Raw Data**:
   - Original archives and unmodified annotation files (`gt.txt`, `GT-*.csv`) are preserved separately in `data/raw/` directories and never overwritten.

---

## 5. Data Splitting and Leakage Prevention
- **Separate Partitions**: Training, validation, and test sets are strictly partitioned.
- **Sequence-Based Splitting**:
  - Consecutive dashcam frames from continuous driving video sequences exhibit high temporal redundancy.
  - To eliminate cross-partition data leakage, all frames originating from the same video sequence ID are assigned exclusively to either the training set or the validation/test set.
  - In the BDD100K subset, all 717 unique sequences are partitioned without cross-split overlap.
- **Benchmark Integrity**: The original benchmark test sets are reserved strictly for final testing and are never used for model training, weight optimization, or hyperparameter tuning.

---

## 6. Evaluation
The project evaluates trained models against the following operational criteria:
- **Detection Accuracy**: $\text{mAP@0.5} \ge 0.80$.
- **Missed-Sign Rate**: $\le 5\%$ for `stop`, `speed`, and `danger` categories (Recall $\ge 95\%$).
- **Inference Latency**: $\le 150\text{ ms/frame}$ on CPU.
- **Deployed Model Size**: $\le 50\text{ MB}$ (ONNX / TorchScript / PyTorch).
- **Drivable-Area Segmentation**: Evaluated via Mean Intersection over Union (mIoU) if drivable mask segmentation is active.

Evaluation results are recorded and reported in [`docs/evaluation.md`](file:///c:/Users/HP/Desktop/Projects/traffic-sign-recognition-road-safety/docs/evaluation.md) after experimental validation runs.

---

## 7. Citation
```bibtex
@inproceedings{stallkamp2011german,
  title={The German Traffic Sign Recognition Benchmark: A multi-class classification competition},
  author={Stallkamp, Johannes and Schlipsing, Marc and Salmen, Jan and Igel, Christian},
  booktitle={Proceedings of the IEEE International Joint Conference on Neural Networks},
  pages={1453--1460},
  year={2011}
}

@inproceedings{houben2013detection,
  title={Detection of Traffic Signs in Real-World Images: The German Traffic Sign Detection Benchmark},
  author={Houben, Sebastian and Stallkamp, Johannes and Salmen, Jan and Schlipsing, Marc and Igel, Christian},
  booktitle={Proceedings of the International Joint Conference on Neural Networks},
  year={2013}
}

@inproceedings{yu2020bdd100k,
  title={BDD100K: A Diverse Driving Dataset for Heterogeneous Multitask Learning},
  author={Yu, Fisher and Chen, Haofeng and Wang, Xin and Xian, Wenqi and Chen, Yingying and Liu, Fangchen and Madhavan, Vashisht and Darrell, Trevor},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={2636--2645},
  year={2020}
}
```

---

## 8. Maintenance & Acquisition Log
This dataset card is updated as data processing evolves:

| Dataset | Acquisition / Manifest Date | Archive Filename / Version | Image Count | License | Manifest / Conversion Status |
| :--- | :---: | :--- | :--- | :--- | :--- |
| **GTSRB** | October 26, 2026 | `GTSRB_Final_Training_Images.zip`, `GTSRB_Final_Test_Images.zip`, `GTSRB_Final_Test_GT.zip` | 39,209 train / 12,630 test (51,839 total) | Non-commercial academic research | Structure preserved in `data/raw/gtsrb/` |
| **GTSDB** | October 26, 2026 | `FullIJCNN2013.zip` | 600 train / 300 test (900 total) | Academic competition & research | Converted to YOLO via `convert_gtsdb_to_yolo.py` with 3-class mapping |
| **BDD100K** | October 26, 2026 | Drivable Area Subset (Seed 42) | 800 train / 200 val (1,000 total) | BDD100K License (Non-commercial research) | Manifest verified in `data/raw/bdd100k/selected_images.csv` |
