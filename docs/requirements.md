# System Requirements & Specifications

## 1. Project Overview

The **Traffic Sign Recognition & Road Safety Monitoring System** is an intelligent Advanced Driver Assistance System (ADAS). It combines deep learning-based visual perception in Python with deterministic, real-time safety evaluation logic in Java.

---

## 2. Key Target Benchmark Metrics

To guarantee real-time road safety operation on in-vehicle hardware, the deployed system must satisfy:

| Metric | Target | Description / Rationale |
| :--- | :--- | :--- |
| **mAP@0.5** | **$\ge 0.80$** | High mean Average Precision across all detected traffic signs. |
| **Missed-Sign Rate (`stop`, `speed`, `danger`)** | **$\le 5\%$** (Recall $\ge 95\%$) | Critical road safety signs must not be missed under varying lighting/weather. |
| **CPU Inference Latency** | **$\le 150\text{ ms/frame}$** | Enables real-time video processing ($\ge 7-10\text{ FPS}$) on standard vehicle CPUs without requiring dedicated GPUs. |
| **Deployed Model Size** | **$\le 50\text{ MB}$** | Allows rapid deployment and cold-start on resource-constrained embedded edge devices (e.g., Raspberry Pi, Jetson Nano, vehicle ECUs). |

---

## 3. Functional Requirements

### 3.1 Python Perception Layer
- **Multi-Class Sign Recognition**: Detect and classify designated road safety signs:
  - `stop` (Class 0): Mandatory halt signs.
  - `speed` (Class 1): Speed limit regulations and speed zone indicators.
  - `danger` (Class 2): Caution, hazard, curve, and pedestrian warnings.
- **Bounding-Box Localization**: Predict accurate bounding box coordinates `(x, y, width, height)` for each detected sign.
- **Confidence Scoring**: Assign calibrated probabilities ($0.0 \le \text{confidence} \le 1.0$) to each candidate detection.
- **Integration Service**: Provide a lightweight HTTP inference service accepting image frames and returning JSON payloads compatible with Java domain objects.

### 3.2 Java Decision & Safety Layer (`SafetyMonitor`)
- **Threshold Filtering**: Discard detections with confidence below threshold (default $\ge 0.60$) to eliminate false alarms.
- **Speed Violation Monitoring**: Track vehicle telemetry against detected speed signs and issue `WARNING` or `CRITICAL` alerts.
- **Mandatory Action Enforcement**: Issue urgent `CRITICAL` alerts for approaching `stop` signs.
- **Hazard Notification**: Alert the driver with `WARNING` notices upon detecting `danger` signs.
- **Audit Logging**: Maintain an in-memory chronological record of all issued alerts.

---

## 4. Technology Stack & Prerequisites

- **Java JDK**: 17+ LTS (Tested on OpenJDK / Oracle JDK 25 LTS).
- **Build Tool**: Apache Maven 3.8+ (Tested on Maven 3.9.6).
- **Python Runtime**: Python 3.10+ (Tested on Python 3.14.6).
- **ML Framework**: PyTorch 2.0+, Ultralytics (YOLOv8 / YOLO11), OpenCV, ONNX Runtime.
