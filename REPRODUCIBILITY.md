# Reproducibility, Seed Management & Experiment Logging Specification

This document details the configuration management, fixed seed determinism, and CSV run logging framework for the **Traffic Sign Recognition & Road Safety ADAS Suite**, ensuring complete mathematical and operational reproducibility across Python ML training and Java embedded execution.

---

## 1. Centralized Hyperparameter Management

All hyperparameters, perception thresholds, training dynamics, and embedded runtime flags are centralized in standard configuration files:
- **YAML Specification**: [`configs/hyperparameters.yaml`](file:///c:/Users/HP/Desktop/Projects/traffic-sign-recognition-road-safety/configs/hyperparameters.yaml) (and root mirror [`HYPERPARAMETERS.yaml`](file:///c:/Users/HP/Desktop/Projects/traffic-sign-recognition-road-safety/HYPERPARAMETERS.yaml))
- **JSON Specification**: [`configs/hyperparameters.json`](file:///c:/Users/HP/Desktop/Projects/traffic-sign-recognition-road-safety/configs/hyperparameters.json) (and root mirror [`HYPERPARAMETERS.json`](file:///c:/Users/HP/Desktop/Projects/traffic-sign-recognition-road-safety/HYPERPARAMETERS.json))

### Configuration Schema Overview

```yaml
reproducibility:
  global_seed: 42
  deterministic_cudnn: true
  benchmark_cudnn: false
  python_hash_seed: 42
  java_random_seed: 42

perception_yolo:
  model_name: "yolov8n"
  imgsz: 640
  epochs: 100
  batch_size: 16
  lr0: 0.01
  lrf: 0.01
  optimizer: "SGD"
  momentum: 0.937
  weight_decay: 0.0005
  conf_threshold: 0.25
  iou_threshold: 0.50
  nms_threshold: 0.45
  device: "cpu"

classifier_resnet18:
  model_name: "resnet18"
  crop_size: 64
  batch_size: 32
  epochs: 10
  learning_rate: 0.001
  weight_decay: 0.0001
  optimizer: "Adam"

sequence_lstm_risk:
  model_name: "lstm_risk_v1"
  sequence_length: 10
  hidden_dim: 64
  num_layers: 2
  activation: "tanh"
  learning_rate: 0.001
  epochs: 20

java_app_runtime:
  default_speed_kmh: 68.0
  min_confidence: 0.50
  nms_iou_threshold: 0.45
  letterbox_size: 640
  letterbox_pad_color: [114, 114, 114]
  input_source: "data/images/test"
  output_directory: "data/output_frames"
```

---

## 2. Fixed Random Seed Enforcement

To guarantee identical numerical results across successive training and inference iterations, a fixed global seed (`seed = 42`) is enforced:

### 2.1 Python ML Pipeline (`python/reproducibility.py`)
```python
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
```

### 2.2 Java Embedded Runtime (`java-app/src/main/java/com/traffic/app/Main.java`)
- `Main.java` reads the `--seed` CLI argument or extracts `global_seed: 42` directly from `configs/hyperparameters.json`.
- All pseudorandom tie-breakers in Non-Maximum Suppression and synthetic baseline generators initialize with `new Random(seed)`.

---

## 3. Automated CSV Run Logging

Every run (ML training, robustness evaluation, Java perception execution, and benchmarking) automatically appends a structured record to persistent CSV logs:

### 3.1 Python Experiment Log: `logs/experiments_log.csv` & `runs/experiment_runs.csv`
| Column Header | Description | Example Value |
| :--- | :--- | :--- |
| `run_id` | Unique run identifier with timestamp | `reproducibility_audit_20261117_192817` |
| `timestamp` | Execution date and time | `2026-11-17 19:28:17` |
| `component` | Executed pipeline module | `yolo_test_eval`, `resnet18_train`, `lstm_risk` |
| `model_name` | Model architecture name | `yolov8n`, `resnet18` |
| `seed` | Deterministic random seed | `42` |
| `dataset` | Dataset split evaluated | `GTSDB/GTSRB` |
| `status` | Run outcome status | `SUCCESS` |
| `git_commit` | Git commit hash for exact traceability | `dfa095c` |
| `metrics_json` | Performance metrics dictionary | `{"mAP@0.5": 0.6833, "missed_rate": 0.0}` |
| `hyperparameters_json` | Exact hyperparameters used | `{"lr": 0.01, "batch": 16, "imgsz": 640}` |

### 3.2 Java Execution Log: `logs/java_runs_log.csv` & `runs/java_runs_log.csv`
| Column Header | Description | Example Value |
| :--- | :--- | :--- |
| `run_id` | Java execution run identifier | `java_detect_1794924054566` |
| `timestamp` | Execution date and time | `2026-11-17 19:30:54` |
| `command` | CLI command executed | `detect`, `benchmark`, `demo-neurons` |
| `input_path` | Input folder or video stream | `data\images\test` |
| `frames_processed`| Total video frames evaluated | `15` |
| `detections_count`| Total traffic signs detected | `1734` |
| `alerts_count` | Total active safety alerts issued | `1734` |
| `speed_kmh` | Vehicle telemetry speed | `75.0 km/h` |
| `avg_latency_ms` | End-to-end latency per frame | `44.13 ms` (Benchmark: `3.95 ms`) |
| `peak_heap_mb` | Peak active JVM heap memory | `83.07 MB` (Benchmark: `121.89 MB`) |
| `seed` | Configured deterministic seed | `42` |
| `status` | Execution status | `SUCCESS` |

---

## 4. How to Run with Reproducibility

### 4.1 Running the Python Reproducibility Audit
```bash
python python/reproducibility.py
```

### 4.2 Running the Java Pipeline with Config and Fixed Seed
```bash
# Windows PowerShell
.\build_and_run.ps1 detect --config configs/hyperparameters.json --speed 75.0 --seed 42

# Maven Execution
mvn clean test
```
All outputs are guaranteed identical, and new execution rows are automatically recorded in [`logs/java_runs_log.csv`](file:///c:/Users/HP/Desktop/Projects/traffic-sign-recognition-road-safety/logs/java_runs_log.csv).
