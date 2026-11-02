# Model Evaluation & Safety Metrics

This document defines the evaluation framework and records benchmark results across detection accuracy, safety missed-sign rates, CPU latency, and model size.

---

## 1. Project Benchmark Targets vs. Verified Test Results

The table below summarizes performance across the required metrics evaluated on the held-out test split:

| Model | mAP@0.5 | mAP@0.5:0.95 | Missed-Sign Rate (Stop / Speed / Danger / Overall) | CPU Latency (ms/frame) | Deployed Model Size | Verification Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Project Target** | **$\ge 0.80$** | **-** | **$\le 5.0\%$ (all & per-class)** | **$\le 150$ ms** | **$\le 50$ MB** | - |
| **YOLOv8n (Current)** | `0.6791` | `0.1950` | `0.0% / 0.0% / 0.0% / 0.0%` | `29.5 ms` | `6.2 MB` | Baseline Verified |
| **YOLOv8s (Next Run)**| *Pending* | *Pending* | *Pending* | *Target $\le 75$ ms* | *Target $\sim 22$ MB* | Next iteration |

---

## 2. 5-Phase YOLO Results Verification Framework

The automated verification tool is available at [`train/check_yolo_results.py`](file:///c:/Users/HP/Desktop/Projects/traffic-sign-recognition-road-safety/train/check_yolo_results.py).

### Phase 1: Output Artifacts & Diagnostics Inspection
After training finishes in `runs/detect/train/` (or `train2`, `train3`...), verify the following key files:

| File | Diagnostic Purpose | Expected Healthy Pattern | Red Flag Alert |
| :--- | :--- | :--- | :--- |
| `results.png` / `results.csv` | Loss and mAP curves per epoch | Training & validation loss decrease; mAP rises then plateaus | Val loss rising while train loss drops indicates **overfitting** |
| `confusion_matrix.png` | Inter-class confusion & background false negatives | High diagonal values | Heavy background row indicates **missed signs** |
| `PR_curve.png` / `F1_curve.png` | Precision-recall tradeoff per class | Smooth convex curves toward top-right corner | Near-zero curve indicates insufficient samples or label errors |
| `val_batch0_pred.jpg` | Ground-truth visual box alignment | Tight bounding boxes centered over signs | Misaligned or missing boxes |
| `weights/best.pt` | Checkpoint with best validation metric | Deployed model size $\le 50$ MB | Do not deploy `last.pt` |

---

### Phase 2: Real Numbers on Held-Out Test Set
Validation numbers in training logs reflect the tuning set. Test numbers must be measured on the independent `test` split:

```python
from ultralytics import YOLO

model = YOLO('runs/detect/train-3/weights/best.pt')
m = model.val(data='data/traffic_signs.yaml', split='test', imgsz=640)

print(f"mAP@0.5      : {m.box.map50:.4f}")
print(f"mAP@0.5:0.95 : {m.box.map:.4f}")
print(f"mean precision: {m.box.mp:.4f}, mean recall: {m.box.mr:.4f}")
for i, name in m.names.items():
    cls_idx = list(m.box.ap_class_index).index(i)
    print(f"  {name:<10} AP50 = {m.box.ap50[cls_idx]:.4f}")
```

#### Measured Per-Class AP50:
- **Speed Limit (`speed`)**: `0.9950` (Pass)
- **Stop Signs (`stop`)**: `0.7107`
- **Hazard Warning (`danger`)**: `0.3317`

---

### Phase 3: Safety Metric: Missed-Sign Rate
The missed-sign rate calculates the exact percentage of real traffic signs missed by the model at $\text{IoU} \ge 0.50$:

$$\text{Missed Rate} = \frac{\text{Missed Signs}}{\text{Total Ground Truth Signs}} \times 100\%$$

```python
import glob, os, torch
from torchvision.ops import box_iou
from ultralytics import YOLO

model = YOLO('runs/detect/train-3/weights/best.pt')
CONF, IOU_T = 0.25, 0.50
missed = total = 0

for img in glob.glob('data/images/test/*.jpg'):
    lbl = img.replace('images', 'labels').rsplit('.', 1)[0] + '.txt'
    if not os.path.exists(lbl):
        continue
    r = model(img, conf=CONF, verbose=False)[0]
    h, w = r.orig_shape
    pred_boxes, pred_cls = r.boxes.xyxy.cpu(), r.boxes.cls.cpu()
    for line in open(lbl):
        c, xc, yc, bw, bh = map(float, line.split())
        gt = torch.tensor([[(xc - bw/2)*w, (yc - bh/2)*h, (xc + bw/2)*w, (yc + bh/2)*h]])
        total += 1
        hit = False
        if len(pred_boxes):
            ious = box_iou(gt, pred_boxes)[0]
            hit = any(ious[j] >= IOU_T and pred_cls[j] == int(c) for j in range(len(ious)))
        missed += (not hit)

print(f"Missed-sign rate: {missed/total:.2%} ({missed}/{total})")
```

#### Measured Safety Results:
- **Stop Sign Missed Rate**: **`0.00%`** (0/5 signs missed) ✅ **Target $\le 5\%$ Satisfied**
- **Speed Limit Missed Rate**: **`0.00%`** (0/5 signs missed) ✅ **Target $\le 5\%$ Satisfied**
- **Danger Warning Missed Rate**: **`0.00%`** (0/5 signs missed) ✅ **Target $\le 5\%$ Satisfied**
- **Overall Missed Rate**: **`0.00%`** (0/15 signs missed) ✅ **Target $\le 5\%$ Satisfied**

> *Tip: If the missed rate rises above 5% on real highway footage, lower `CONF` to 0.15 to aggressively suppress false negatives.*

---

### Phase 4: CPU Latency & Deployed Footprint
Measures forward inference + post-processing on standard CPU architecture without requiring an external GPU:

- **Measured CPU Latency**: **`29.46 ms/frame`** on 13th Gen Intel Core i5 (Target: $\le 150$ ms).
  - Equivalent processing rate: **$\sim 34\text{ FPS}$** (real-time stream capability).
- **Deployed Checkpoint Size**: **`6.16 MB`** for `best.pt` (Target: $\le 50$ MB).

---

### Phase 5: Decision Matrix & Next Actions

| Evaluation Finding | Diagnostic Cause | Recommended Action |
| :--- | :--- | :--- |
| **mAP@0.5 $\ge 0.80$ & Missed $\le 5\%$** | Optimal training convergence | Keep as baseline and train `yolov8s` for precision enhancement. |
| **mAP is fine, but one class is weak** | Class imbalance in training samples | Apply class-targeted mosaic/affine augmentation or add 50+ samples of weak class. |
| **mAP is low overall ($< 0.70$)** | Training terminated prematurely | Train for 100+ epochs on Google Colab with `imgsz=640` using GPU runtime. |
| **Training mAP high, Test mAP low** | Overfitting or split leakage | Re-verify track-based splitting in `prepare_gtsrb.py` to eliminate cross-split duplicates. |

---

## 3. How to Run the Comprehensive Verifier

Run the automated Python verification script locally or in Colab:

```powershell
cd C:\Users\HP\Desktop\Projects\traffic-sign-recognition-road-safety
.venv\Scripts\python.exe train/check_yolo_results.py --weights runs/detect/train-3/weights/best.pt --data data/traffic_signs.yaml
```
Output results are exported directly to `runs/detect/benchmark_summary.json`.
