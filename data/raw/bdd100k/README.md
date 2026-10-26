# Berkeley DeepDrive (BDD100K) Drivable Area Subset

## 1. Dataset Overview & Source Metadata
- **Dataset Name**: BDD100K Drivable Area Segmentation (1,000-Image Representative Subset)
- **Primary Source URL**: [https://bdd-data.berkeley.edu/](https://bdd-data.berkeley.edu/)
- **Mirror / Distribution Repository**: [https://huggingface.co/datasets/dronefreak/BDD100K-Drivable-Area-Segmentation](https://huggingface.co/datasets/dronefreak/BDD100K-Drivable-Area-Segmentation)
- **Download Date**: October 2026
- **License**: BDD100K Academic and Non-Commercial Research License (Fisher Yu, Trevor Darrell, UC Berkeley BAIR).
- **Domain Note**: BDD100K drivable-area masks provide **road-surface semantic segmentation**, NOT traffic-sign bounding boxes. They serve complementary road-scene understanding (distinguishing ego lane from adjacent lanes and non-drivable obstacles).

---

## 2. Directory Layout & Organization

```
data/raw/bdd100k/
├── images/                   # 1,000 RGB dashcam frames (1280 x 720 JPEG)
├── drivable_masks/           # 1,000 8-bit single-channel PNG segmentation masks (1280 x 720)
├── selected_images.csv       # Manifest with selection metadata, splits, and validation checks
└── README.md
```

---

## 3. Drivable-Area Mask Taxonomy
Each pixel in the drivable-area mask represents one of 3 classes:

| Class ID | Label Name | Definition | Visual Color (Palette) |
| :---: | :--- | :--- | :--- |
| **`0`** | **Direct Drivable Area** | The current travel lane occupied by the ego-vehicle. Directionally valid and immediately navigable. | Red `[255, 0, 0]` |
| **`1`** | **Alternative Drivable Area** | Adjacent travel lanes heading in the permitted direction, reachable via lane change. | Blue `[0, 0, 255]` |
| **`2`** | **Background / Non-Drivable** | Sidewalks, opposite traffic lanes, road shoulders, terrain, obstacles, and off-road structures. | Black `[0, 0, 0]` |

---

## 4. Selection & Sampling Methodology
The 1,000 images were selected using a **reproducible, stratified sampling protocol**:

1. **Fixed Random Seed**:
   - Seed: `42`
   - Guarantees exact reproducibility across different machines and runs.
2. **Split Stratification**:
   - Preserves original dataset ratio: **800 training images** (80%) and **200 validation images** (20%).
   - Sequence-based integrity: Frames originating from identical video sequence IDs are confined to the same partition to eliminate cross-split information leakage.
3. **Diverse Environmental Representation**:
   - **Lighting Conditions**: Daytime (~60%), Night (~25%), Dawn/Dusk (~15%).
   - **Weather Conditions**: Clear (~55%), Rainy (~18%), Overcast (~20%), Snowy/Foggy (~7%).
   - **Road Environments**: Highway, City Street, Residential.
4. **Validation Checklist (Per-Sample)**:
   - **1:1 Pair Correspondence**: Every image `[id].jpg` has an identically named mask `[id].png`.
   - **Dimensional Consistency**: Both image and mask must be exactly $1280 \times 720$ pixels.
   - **Label Set Validation**: Mask pixel values must strictly be a subset of $\{0, 1, 2\}$.

---

## 5. Manifest Specification (`selected_images.csv`)
The generated manifest documents each selected sample:

| Column | Description |
| :--- | :--- |
| `filename` | Image filename (e.g., `0004944c-ab2e4262.jpg`) |
| `mask_filename` | Drivable mask filename (e.g., `0004944c-ab2e4262.png`) |
| `split` | Original BDD100K partition (`train` or `val`) |
| `image_width` | Image pixel width (1280) |
| `image_height` | Image pixel height (720) |
| `mask_width` | Mask pixel width (1280) |
| `mask_height` | Mask pixel height (720) |
| `dimensions_match` | Boolean (`TRUE` if width and height match) |
| `valid_mask_labels` | Unique pixel labels observed in mask (must be $\subseteq \{0, 1, 2\}$) |
| `time_of_day` | Lighting category (`daytime`, `night`, `dawn/dusk`) |
| `weather` | Weather category (`clear`, `rainy`, `overcast`, `snowy`) |
| `scene` | Environment category (`highway`, `city street`, `residential`) |
| `sampling_seed` | Generator seed (`42`) |
