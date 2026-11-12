# Traffic Sign Recognition & Road Safety Monitoring System: User Guide

A step-by-step developer and operator manual for setting up, building, training, and running the real-time Traffic Sign Perception and Road Safety ADAS suite.

---

## 1. System Requirements & Prerequisites

### 1.1 Hardware Specifications
- **CPU**: x86_64 or ARM64 multi-core processor (Intel Core i5/i7/i9, AMD Ryzen, Apple Silicon, or Raspberry Pi 4/5).
- **RAM**: Minimum 4 GB (8 GB recommended).
- **Camera**: No physical webcam or camera is required. The system includes a built-in sequential frame directory player (`VideoFolderPlayer.java`) that processes video feeds from image directories.

### 1.2 Software Dependencies
- **Java Development Kit (JDK)**: JDK 17 LTS or higher (Tested on Oracle JDK 25 and OpenJDK 17).
- **Apache Maven**: Version 3.8+ (Tested on Apache Maven 3.9.6).
- **Python**: Version 3.10+ (Tested on Python 3.14).
- **Python Libraries**: `torch`, `torchvision`, `ultralytics`, `opencv-python`, `numpy`, `matplotlib`, `pillow`, `pandas`.

---

## 2. Quick-Start Setup (Windows 1-Click)

The repository provides automated build and execution scripts that automatically resolve Windows PowerShell path and wildcard constraints:

### Option A: Using `run.bat`
Double-click `run.bat` in File Explorer, or open a Command Prompt / PowerShell in the repository root:
```cmd
run.bat
```
This script will:
1. Compile `java-core` classes into `bin/core`.
2. Compile `java-app` classes into `bin/app`.
3. Execute the `detect`, `benchmark`, and `demo-neurons` CLI commands automatically.

### Option B: Using `build_and_run.ps1`
Open PowerShell in the repository root:
```powershell
.\build_and_run.ps1
```

---

## 3. Maven Build & Unit Testing (`mvn test`)

The repository includes a root aggregator POM and modular submodules:
- **`java-core`**: Core domain logic, McCulloch-Pitts neurons, Sigmoid activation, greedy Non-Maximum Suppression (NMS), and `SafetyMonitor`.
- **`java-app`**: Application CLI, aspect-preserving Letterbox preprocessor, `Graphics2D` box renderer, and frame feed player.

### 3.1 Running Unit Tests
To clean, compile, and run all **34 JUnit 5 unit tests**:
```bash
mvn clean test
```
Or run specifically within the `java-core` directory:
```bash
cd java-core
mvn clean test
```

### 3.2 Packaging JAR Files
To package executable JARs:
```bash
mvn clean package
```
Artifacts are generated at:
- `java-core/target/java-core-1.0.0.jar`
- `java-app/target/java-app-1.0.0.jar`

---

## 4. Java Application CLI Reference

The Java application provides three primary CLI operational commands:

### Command 1: `detect` (Traffic Sign Perception & Safety Alerts)
Processes road camera image frames or video sequences, generates candidate detections, applies Java NMS filtering, tracks vehicle speed against posted limits, and renders annotated output frames with alert banners:
```bash
java -cp "bin/app;bin/core" com.traffic.app.Main detect [image_or_folder_path] [vehicle_speed_kph]
```
**Examples**:
```bash
# Process test frames at 65.0 km/h (within speed limit)
java -cp "bin/app;bin/core" com.traffic.app.Main detect data/images/test 65.0

# Process test frames at 92.5 km/h (triggers SPEED VIOLATION WARNING alert)
java -cp "bin/app;bin/core" com.traffic.app.Main detect data/images/test 92.5
```
Annotated output frames are saved to `data/output_frames/`.

---

### Command 2: `benchmark` (Latency & JVM Memory Profiling)
Runs a 50-iteration benchmark profiling letterbox preprocessing, proposal generation, greedy NMS, and `Graphics2D` rendering latency, alongside JVM active heap memory:
```bash
java -cp "bin/app;bin/core" com.traffic.app.Main benchmark
```
**Expected Output**:
```
================================================================================
   JAVA PIPELINE LATENCY & MEMORY BENCHMARK (50 ITERATIONS @ 640x640)
================================================================================
- Letterbox Preprocessing : 3.15 ms / frame (86.7%)
- Neural Perception & NMS : 0.05 ms / frame (1.4%)
- Graphics2D Overlay      : 0.43 ms / frame (12.0%)
--------------------------------------------------------------------------------
Average Latency           : 3.63 ms / frame (275.1 FPS)
P50: 3.00 ms | P95: 5.50 ms | Max: 6.00 ms
Peak Active Heap Memory   : 127.54 MB
Committed JVM Memory      : 266.00 MB
================================================================================
```

---

### Command 3: `demo-neurons` (McCulloch-Pitts & Sigmoid Logic)
Demonstrates discrete logic gate synthesis using McCulloch-Pitts threshold neurons (AND, OR, NOT, NAND, XOR) and continuous Sigmoid neuron activation:
```bash
java -cp "bin/app;bin/core" com.traffic.app.Main demo-neurons
```
**Expected Output**:
```
=== McCulloch-Pitts Discrete Logic Gates ===
AND Gate  : [0,0]->0, [0,1]->0, [1,0]->0, [1,1]->1
OR Gate   : [0,0]->0, [0,1]->1, [1,0]->1, [1,1]->1
NOT Gate  : [0]->1, [1]->0
NAND Gate : [0,0]->1, [0,1]->1, [1,0]->1, [1,1]->0
XOR Gate  : [0,0]->0, [0,1]->1, [1,0]->1, [1,1]->0

=== Continuous Sigmoid Artificial Neuron ===
Input z = -2.000 -> sigma(z) = 0.1192, dSigma(z) = 0.1050
Input z =  0.000 -> sigma(z) = 0.5000, dSigma(z) = 0.2500
Input z =  2.000 -> sigma(z) = 0.8808, dSigma(z) = 0.1050
```

---

## 5. Python Deep Learning Pipeline

All Python training and evaluation scripts reside in the `train/` and `python/` directories:

### 5.1 Dataset Preparation & Manifest Generation
```bash
python train/prepare_gtsrb.py
```
Prepares track-aware splits for GTSRB crops, ensuring approach sequences are never split across train/val/test partitions.

### 5.2 YOLOv8n Training & Safety Verification
```bash
# Train YOLOv8n detector
python train/train.py

# Verify test split mAP, missed-sign rates, and CPU latency
python train/check_yolo_results.py
```

### 5.3 Fine-Grained Classifier Training
```bash
# Train ResNet-18 classifier (96.42% accuracy)
python train/train_resnet18.py

# Train GoogLeNet classifier (94.80% accuracy)
python train/train_googlenet.py
```

### 5.4 Sequence Tracking & Temporal LSTM Risk Model
```bash
# Extract approach sequence dynamics (box area growth delta)
python python/sequence_tracker.py

# Train 2-layer LSTM risk model comparing Step, Sigmoid, and Tanh
python train/train_lstm_risk.py
```

### 5.5 Conditional VAE Data Augmentation
```bash
python train/train_cvae.py
```
Generates synthetic sign crops for rare classes and plots the latent space using t-SNE.

### 5.6 ONNX Model Export
```bash
python train/export_all_onnx.py
```
Exports YOLOv8n, ResNet-18, and LSTM models to `checkpoints/onnx/` with verified PyTorch equivalence.

### 5.7 Environmental Robustness Testing
```bash
python train/evaluate_robustness.py
```
Evaluates model accuracy drops under low brightness, solar glare, motion blur, gaussian blur, sensor grain, and rain streaks.

### 5.8 Error Analysis & Failure-Case Gallery Generation
```bash
python train/error_analysis.py
```
Generates the 6-panel failure-case gallery and detection/classification confusion matrices.

### 5.9 Google Colab Turnkey Notebook
For full GPU cloud training:
- Open [`train/vision_pipeline_colab.ipynb`](file:///C:/Users/HP/Desktop/Projects/traffic-sign-recognition-road-safety/train/vision_pipeline_colab.ipynb) in Google Colab.
- Select GPU runtime (T4 / V100).
- Run All to execute all 15 stages from dataset preparation to ONNX export.

---

## 6. Troubleshooting & FAQs

### Q1: `javac: Invalid filename: java-core/src/main/java/com/traffic/core/*.java` on PowerShell
**Reason**: Native Windows PowerShell does not expand Unix-style wildcards (`*`) for external executables like `javac.exe`.  
**Solution**: Use `.\build_and_run.ps1` or `run.bat`. They use `(Get-ChildItem ...).FullName` and `@sources.txt` to pass resolved paths directly.

### Q2: Do I need a physical webcam or external video capture card?
**Reason**: No. The system was designed specifically for deterministic in-vehicle ECU evaluation. The built-in `VideoFolderPlayer` sequentially reads a folder of image frames (`data/images/test`) and streams them as simulated camera frames, avoiding OpenCV native driver issues.

### Q3: Why is detection mAP@0.5 below 0.80 on the baseline checkpoint?
**Reason**: The rapid baseline was trained for a limited number of epochs on CPU. To achieve mAP@0.5 $\ge 0.80$, run the turnkey Colab notebook [`train/vision_pipeline_colab.ipynb`](file:///C:/Users/HP/Desktop/Projects/traffic-sign-recognition-road-safety/train/vision_pipeline_colab.ipynb) for 100 epochs on GPU using the full dataset.

### Q4: Can I deploy this model in India?
**Reason**: **No, not without fine-tuning.** The model was trained on European signs (Vienna Convention). Indian road signage follows **IRC-67 standards** with multilingual text and unique hazard pictograms. Fine-tuning on the Indian Driving Dataset (IDD) is required before deploying on Indian roads.
