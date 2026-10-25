# Model Evaluation & Safety Metrics

This document defines the evaluation framework used to assess perception accuracy and safety monitoring effectiveness.

---

## 1. Project Benchmark Targets

To guarantee reliable real-time operation on road safety hardware, the perception model must satisfy the following criteria:

| Metric | Target | Rationale |
| :--- | :--- | :--- |
| **mAP@0.5** | **$\ge 0.80$** | Ensures solid precision and recall across all traffic sign classes. |
| **Missed-sign rate for stop, speed, and danger classes** | **$\le 5\%$** (Recall $\ge 95\%$) | Critical safety requirement: safety-critical signs cannot be missed. |
| **CPU inference latency** | **$\le 150\text{ ms/frame}$** | Enables smooth video/dashcam processing ($\ge 7-10\text{ FPS}$) on standard vehicle CPUs. |
| **Deployed model size** | **$\le 50\text{ MB}$** | Allows deployment on resource-constrained embedded edge devices (e.g. Raspberry Pi, Jetson Nano, in-vehicle ECUs). |

---

## 2. Recommended Model Architecture: Small YOLO

A lightweight YOLO object-detection model (**YOLOv8n** or **YOLOv11n**) is the ideal starting point:
- **Parameter & Size Efficiency**: `yolov8n.pt` has $\sim 3.2\text{M}$ parameters and is **$\sim 6.2\text{ MB}$**, well below the $\le 50\text{ MB}$ ceiling.
- **CPU Speed**: Benchmark latency on modern desktop/laptop CPUs is typically **$30 - 60\text{ ms/frame}$**, easily meeting the $\le 150\text{ ms/frame}$ limit.
- **Unified Prediction**: Single forward pass predicts:
  1. Traffic-sign classes (`stop`, `speed`, `danger`)
  2. Bounding-box coordinates `(x, y, width, height)`
  3. Calibrated confidence scores (0.0 to 1.0)

> [!IMPORTANT]
> **Domain Fine-Tuning**: Off-the-shelf COCO pretrained models only recognize standard general objects (cars, pedestrians, traffic lights). You must fine-tune the model on an annotated traffic sign dataset with labeled `stop`, `speed`, and `danger` annotations.

---

## 3. Evaluation Metrics

### 3.1 Mean Average Precision (mAP@0.5)
$$\text{mAP@0.5} = \frac{1}{C} \sum_{c=1}^{C} \text{AP}_c@0.5$$
Evaluates precision-recall trade-off at Intersection-over-Union ($\text{IoU}$) threshold of 0.5.

### 3.2 Missed-Sign Rate (False Negative Rate)
$$\text{Missed Rate} = 1 - \text{Recall} = \frac{\text{FN}}{\text{TP} + \text{FN}}$$
- Must remain $\le 5\%$ for high-risk signs (`stop`, `speed`, `danger`).

---

## 4. Safety Decision Evaluation (Java Core)

The Java Safety Engine is evaluated on deterministic reaction correctness and latency:

| Scenario | Input Detection | Vehicle Speed | Expected Alert Level | Expected Reaction Time |
| :--- | :--- | :--- | :--- | :--- |
| **Normal Compliance** | 50 km/h Limit | 48 km/h | `INFO` | $< 1\text{ ms}$ |
| **Minor Speeding** | 50 km/h Limit | 55 km/h | `WARNING` | $< 1\text{ ms}$ |
| **Gross Speeding** | 50 km/h Limit | 72 km/h | `CRITICAL` | $< 1\text{ ms}$ |
| **Stop Sign Approach** | Stop Sign | 35 km/h | `CRITICAL` | $< 1\text{ ms}$ |
| **Noisy / Ghost Sign** | Any (Conf: 0.35) | Any | Ignored | $< 1\text{ ms}$ |
