# Model Card: Traffic Sign Recognition & Road Safety Perception Suite

Following the Model Cards for Model Reporting framework (Mitchell et al., 2019).

---

## 1. Model Details

- **Model Name**: Traffic Sign Recognition & Road Safety Perception Suite (`TSR-Safety-Suite-v1`)
- **Model Suite Sub-components**:
  1. `YOLOv8n-TSR`: Anchor-free single-stage object detector for full-frame sign localization and classification.
  2. `ResNet-18-GTSRB`: 18-layer residual convolutional neural network for fine-grained sign crop verification.
  3. `LSTM-Risk-v1`: 2-layer Recurrent Neural Network for temporal approach risk valuation over 10-frame sliding windows.
  4. `CVAE-Augmenter-v1`: Conditional Variational Autoencoder for synthetic generation of rare-class training samples.
- **Developers**: Deep Learning Perception & ADAS Embedded Systems Engineering
- **Release Version**: `1.0.0`
- **Release Date**: November 2026
- **Model Formats**:
  - PyTorch Checkpoints (`.pt`): `checkpoints/best_model.pt`, `checkpoints/resnet18_gtsrb.pt`, `checkpoints/lstm_risk.pt`, `checkpoints/cvae_gtsrb.pt`
  - ONNX Runtimes (`.onnx`): `checkpoints/onnx/yolov8n.onnx`, `checkpoints/onnx/resnet18.onnx`, `checkpoints/onnx/lstm_risk.onnx`
- **Deployment Runtime**: Pure Java (JDK 17+) using `javax.imageio` and custom NMS / Letterbox pipelines (no native OpenCV dependency).
- **License**: Apache 2.0 / MIT Open Source

---

## 2. Intended Use

### 2.1 In-Scope Intended Uses
- **Level 2 Advanced Driver Assistance Systems (ADAS)**: Real-time driver advisory notices, visual HUD dashboard warnings, and audible speed limit/halt reminders.
- **Fleet Video Telemetry Auditing**: Automated offline processing of commercial vehicle dashcam footage for traffic sign compliance logs.
- **Edge Vehicle Platforms**: Embedded execution on ARM64 / x86_64 automotive microcontrollers without requiring dedicated GPU accelerators.

### 2.2 Out-of-Scope & Prohibited Uses
- **Autonomous Vehicle Control (SAE Level 3, 4, 5)**: **Strictly Prohibited.** Must never be connected directly to steer-by-wire, throttle, or brake actuation without an attentive human driver in the loop.
- **Non-Standard Signage Regions**: **Prohibited** for production deployment on Indian roads or regions not conforming to the Vienna Convention on Road Signs without prior supervised fine-tuning.
- **Primary Collision Avoidance Safety System**: Must not be used as the sole sensor for Automated Emergency Braking (AEB).

---

## 3. Factors & Demographic / Environmental Conditions

The model's performance was evaluated across critical real-world environmental factors:

| Environmental Factor | Nominal Condition | Adverse Condition Tested | Performance Impact |
| :--- | :--- | :--- | :--- |
| **Illumination** | Daylight ($> 1000\text{ lux}$) | Night underexposure ($< 50\text{ lux}$), Solar glare | Solar glare causes $-10.83\%$ accuracy drop due to sensor saturation |
| **Weather** | Clear / Dry pavement | Rain streaks ($450\text{ streaks}$), Atmospheric haze | Rain & haze causes $-12.50\%$ accuracy drop due to specular edge dispersion |
| **Vehicle Motion** | Stationary / Low speed | High-speed chassis vibration, Linear motion blur | Motion blur ($11\times 11$) reduces raw confidence by $18\%$, but preserves box centers |
| **Sensor Characteristics** | Clean optical lens | Low-light sensor grain ($\sigma = 28$) | Negligible impact ($< 3.5\%$ drop); conv kernels filter high-frequency grain |

---

## 4. Metrics & Quantitative Performance

### 4.1 Primary Safety & Perception Metrics
- **mAP@0.5 (YOLOv8n)**: `0.6833` (Faster R-CNN: `0.7850`)
- **Missed-Sign Rate (`stop`, `speed`, `danger`)**: `0.00%` (0/15 test instances missed at $\text{Conf} \ge 0.25$)
- **Crop Classification Accuracy (ResNet-18)**: `96.42%` Top-1 Accuracy across GTSRB test crops
- **Rare-Class Recall Improvement via CVAE**: `+15.1%` (Baseline: $75.9\% \implies$ Augmented: $91.0\%$)
- **LSTM Sequence Risk Accuracy**: `89.35%` (Tanh activation, 0.9030 F1-score)

### 4.2 Hardware Runtime Performance (Standard CPU)
- **Letterbox Preprocessing**: `3.15 ms`
- **Neural Proposal & Greedy NMS**: `0.05 ms`
- **Graphics2D Visual Overlay**: `0.43 ms`
- **Total Pipeline Latency**: `3.63 ms / frame` (**275.1 FPS**)
- **P95 Latency**: `5.50 ms`
- **Peak Active Heap Memory**: `127.54 MB`
- **Deployed ONNX Model Size**: `6.16 MB` (YOLOv8n)

---

## 5. Training & Evaluation Datasets

1. **German Traffic Sign Detection Benchmark (GTSDB)**:
   - 900 full-frame $1360 \times 800$ natural road images containing 1,206 traffic sign instances.
   - Grouped into three primary safety categories: `stop` (Class 0), `speed` (Class 1), and `danger` (Class 2).
2. **German Traffic Sign Recognition Benchmark (GTSRB)**:
   - 51,839 cropped sign images across 43 classes.
   - Split strictly by **Track ID** (all frames from a single approach sequence remain in the same split to eliminate data leakage).
3. **Berkeley DeepDrive 100K (BDD100K)**:
   - Multi-frame driving sequences used to calibrate temporal approach dynamics ($\Delta \text{area} = \text{area}_t - \text{area}_{t-1}$) and evaluate LSTM risk windows.

---

## 6. Ethical Considerations & Responsible-Use Guidelines

1. **Driver Primacy & Accountability**:
   The system is an advisory driver-assist tool. The vehicle operator remains legally responsible for vehicle handling and observing all posted traffic regulations.
2. **Failure Transparency**:
   When adverse weather or sensor contamination degrades optical visibility, the software automatically transitions into a degraded state and alerts the driver.
3. **Regional Adaptation Mandate**:
   The perception suite must not be deployed in countries with non-standardized signage (such as India's IRC-67 bilingual signs) without collecting representative local training data and evaluating localized false negative rates.
