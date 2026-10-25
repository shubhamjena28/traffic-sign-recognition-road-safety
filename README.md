# Traffic Sign Recognition & Road Safety Monitoring

[![Java](https://img.shields.io/badge/Java-17%2B-orange.svg)](https://www.oracle.com/java/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

An intelligent Advanced Driver Assistance System (ADAS) integrating deep learning visual perception with a deterministic real-time road safety monitoring engine.

---

## Project Structure

```text
traffic-sign-recognition-road-safety/
│
├── java-core/               # Core domain models, safety rules, and unit tests
│   ├── pom.xml
│   └── src/
│       ├── main/java/com/traffic/core/
│       │   ├── Detection.java
│       │   └── SafetyMonitor.java
│       └── test/java/com/traffic/core/
│           └── SafetyMonitorTest.java
│
├── java-app/                # Executable application orchestrating safety alerts
│   ├── pom.xml
│   └── src/main/java/com/traffic/app/
│       └── Main.java
│
├── train/                   # PyTorch CNN training, evaluation, and ONNX export
│   ├── train.py
│   ├── evaluate.py
│   ├── export_model.py
│   └── requirements.txt
│
├── python/                  # Real-time inference scripts and preprocessing utilities
│   ├── inference.py
│   └── utils.py
│
├── data/                    # Dataset directory conventions and documentation
│   ├── README.md
│   ├── raw/
│   ├── processed/
│   └── annotations/
│
├── docs/                    # Technical architecture, requirements, and evaluation specs
│   ├── requirements.md
│   ├── architecture.md
│   ├── dataset.md
│   └── evaluation.md
│
├── .gitignore
└── README.md
```

---

## Key Features

1. **Vision Perception Pipeline (Python)**
   - Convolutional Neural Network (CNN) architecture trained on the German Traffic Sign Recognition Benchmark (GTSRB 43 classes).
   - Export pipeline supporting both **ONNX** and **TorchScript** for edge and runtime deployment.
   - Image and video inference utility with bounding box visualization.

2. **Road Safety Decision Engine (Java Core)**
   - Threshold-filtered detection validation to eliminate noise and false positives.
   - Real-time vehicle speed limit tracking with automatic speed violation detection.
   - Escalated alerts (`INFO`, `WARNING`, `CRITICAL`) for STOP signs, YIELD conditions, and roadway hazards.
   - Full JUnit 5 test suite validating safety evaluation rules.

---

## Getting Started

### Prerequisites

- **Java JDK 17+** (JDK 17, 21, or 25)
- **Apache Maven 3.8+**
- **Python 3.10+**

---

### 1. Python Perception & Training

#### Install Dependencies
```bash
cd train
pip install -r requirements.txt
```

#### Train Model
```bash
python train.py --data-dir ../data/processed --epochs 25 --batch-size 64
```

#### Evaluate Model
```bash
python evaluate.py --checkpoint checkpoints/best_model.pt --data-dir ../data/processed
```

#### Export to ONNX & TorchScript
```bash
python export_model.py --checkpoint checkpoints/best_model.pt
```

#### Run Inference
```bash
cd ../python
python inference.py --image path/to/traffic_sign.jpg
```

---

### 2. Java Safety Monitoring Application

#### Run Unit Tests (Maven)
```bash
cd java-core
mvn test
```

#### Build and Run Application (Maven)
```bash
cd ../java-app
mvn compile exec:java -Dexec.mainClass="com.traffic.app.Main"
```

#### Run Directly via `javac` (Without Maven installed)
```bash
# From repository root:
javac -d bin/core java-core/src/main/java/com/traffic/core/*.java
javac -cp bin/core -d bin/app java-app/src/main/java/com/traffic/app/Main.java
java -cp "bin/core;bin/app" com.traffic.app.Main
```

---

## Documentation

- [System Requirements](docs/requirements.md)
- [Architecture & Data Flow](docs/architecture.md)
- [Dataset Specifications](docs/dataset.md)
- [Evaluation & Benchmark Criteria](docs/evaluation.md)

---

## License

This project is licensed under the Apache 2.0 / MIT License.
