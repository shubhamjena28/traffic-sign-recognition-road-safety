# System Requirements & Specifications

## 1. Project Overview

The **Traffic Sign Recognition & Road Safety Monitoring System** combines deep learning-based visual perception with an intelligent rule-based driver assistance and safety monitoring engine.

## 2. Functional Requirements

### 2.1 Visual Perception (Machine Learning)
- **Multi-Class Sign Recognition**: Accurately recognize 43 distinct classes of European traffic signs (GTSRB standard) under various lighting, weather, and motion conditions.
- **Confidence Scoring**: Assign calibrated confidence probabilities (0.0 to 1.0) to all detected sign candidates.
- **Model Export**: Support export to open, cross-platform formats including ONNX and TorchScript for runtime interoperability.

### 2.2 Safety Monitoring Engine (Java Core)
- **Threshold Filtering**: Filter out low-confidence perceptions (default threshold $\ge 0.60$) to minimize false positives.
- **Speed Limit Tracking & Over-speed Warnings**:
  - Dynamically maintain the current active speed limit.
  - Issue **WARNING** alerts if vehicle speed exceeds current speed limit.
  - Issue **CRITICAL** alerts if vehicle speed exceeds current limit by $> 10 \text{ km/h}$.
- **Mandatory Action Signs**:
  - Issue immediate safety notices when approaching STOP and YIELD signs.
  - Escalate STOP warnings to **CRITICAL** if vehicle approach speed $> 20 \text{ km/h}$.
- **Hazard & Precaution Warnings**: Alert drivers of upcoming pedestrian zones, construction/road work, curves, and hazardous road conditions.
- **Audit Logging**: Maintain an in-memory chronological history of all triggered safety alerts for driver behavior auditing.

## 3. Non-Functional Requirements

- **Inference Latency**: Single frame prediction latency should be $\le 50\text{ ms}$ on standard edge/CPU hardware and $\le 10\text{ ms}$ on CUDA-capable accelerators.
- **Safety Decision Latency**: Java evaluation of incoming detections must be $\le 5\text{ ms}$.
- **Modularity**: Decoupled architecture allowing independent updating of perception models without refactoring Java safety logic.
- **Portability**: Code runs seamlessly across Windows, Linux, and macOS platforms.

## 4. Environment & Dependencies

| Component | Minimum Version | Recommended Version |
| :--- | :--- | :--- |
| **Java JDK** | 17 LTS | 17, 21, or 25 LTS |
| **Apache Maven** | 3.8+ | 3.9+ |
| **Python** | 3.10+ | 3.11 / 3.12 / 3.14 |
| **PyTorch** | 2.0.0+ | Latest stable |
| **OpenCV** | 4.8.0+ | Latest stable |
