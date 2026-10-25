# System Architecture

The Traffic Sign Recognition and Road Safety system is designed as a hybrid, multi-tier architecture separating high-throughput deep learning perception from deterministic, low-latency safety decision logic.

```mermaid
flowchart TD
    subgraph PerceptionLayer["Perception Layer (Python / Deep Learning)"]
        A["Camera / Video Feed"] --> B["Image Preprocessor"]
        B --> C["CNN Model / ONNX Runtime"]
        C --> D["Bounding Box & Confidence Filter"]
        D --> E["Detection JSON Payload"]
    end

    subgraph DecisionLayer["Decision & Safety Layer (Java Core)"]
        E --> F["Detection Model"]
        G["Vehicle Telemetry (Speed, GPS)"] --> H["SafetyMonitor"]
        F --> H
        H --> I{"Rule Engine Evaluation"}
        I -->|"Speed Limit Exceeded"| J["OVERSPEED Alert (WARN / CRITICAL)"]
        I -->|"Stop / Yield Ahead"| K["MANDATORY STOP Alert"]
        I -->|"Hazard Sign"| L["CAUTION Alert"]
    end

    subgraph ApplicationLayer["Application Layer (Java App)"]
        J --> M["Dashboard Display / Driver Audio Warning"]
        K --> M
        L --> M
        M --> N["Safety Audit Log"]
    end
```

## Component Breakdown

### 1. Perception Layer (`train/` and `python/`)
- **`train/train.py`**: Trains deep convolutional neural network backbones using PyTorch on labeled traffic sign datasets (e.g. GTSRB).
- **`train/export_model.py`**: Converts PyTorch weights into ONNX (`.onnx`) or TorchScript representations for deployment across multiple runtime targets.
- **`python/inference.py`**: Performs inference over live camera streams or pre-recorded dashcam footage, identifying sign classes and bounding coordinates.

### 2. Core Safety Engine (`java-core/`)
- **`Detection.java`**: Immutable domain model capturing class ID, natural language label, confidence score, bounding box coordinates, and detection timestamp.
- **`SafetyMonitor.java`**: State-aware safety evaluation engine that correlates detected signage with current vehicle speed and road conditions, firing prioritized alerts (`INFO`, `WARNING`, `CRITICAL`).

### 3. Application Execution (`java-app/`)
- **`Main.java`**: Driver application entry point orchestrating safety monitoring lifecycle and telemetry stream ingestion.
