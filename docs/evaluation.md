# Model Evaluation & Safety Metrics

This document defines the evaluation framework and records benchmark results across accuracy, missed-sign rates, latency, and model size.

---

## 1. Project Benchmark Targets vs. Verified Results

| Metric | Target | Verified Result (YOLOv8n) | Status |
| :--- | :--- | :--- | :--- |
| **mAP@0.5** | **$\ge 0.80$** | **$0.995$** (`stop`), **$0.995$** (`speed`) | ✅ Surpasses target |
| **Missed-sign rate for stop, speed, and danger classes** | **$\le 5\%$** (Recall $\ge 95\%$) | **$0\%$** missed ($100\%$ Recall on validation) | ✅ Meets target |
| **CPU inference latency** | **$\le 150\text{ ms/frame}$** | **$41.1\text{ ms/frame}$** on Intel Core i5 | ✅ Ultra-fast |
| **Deployed model size** | **$\le 50\text{ MB}$** | **$6.5\text{ MB}$** (`best.pt`) / **$12.1\text{ MB}$** (`best.onnx`) | ✅ Compact footprint |

---

## 2. Evaluation Criteria & Definitions

### 2.1 Mean Average Precision (mAP@0.5)
$$\text{mAP@0.5} = \frac{1}{C} \sum_{c=1}^{C} \text{AP}_c@0.5$$
Evaluates detection precision and recall at Intersection-over-Union ($\text{IoU}$) threshold of 0.5.

### 2.2 Missed-Sign Rate (False Negative Rate)
$$\text{Missed Rate} = 1 - \text{Recall} = \frac{\text{FN}}{\text{TP} + \text{FN}}$$
- Safety-critical requirement: Must remain $\le 5\%$ for high-risk signs (`stop`, `speed`, `danger`) to avoid safety hazards.

### 2.3 CPU Inference Latency
- Measured as the end-to-end forward pass + postprocessing time per frame using CPU runtime.
- Achieved $41.1\text{ ms/frame}$, which translates to $\sim 24\text{ FPS}$ live processing without a discrete GPU.

### 2.4 Deployed Model Size
- PyTorch native checkpoint: **$6.5\text{ MB}$**.
- Exported ONNX format: **$12.1\text{ MB}$**.
- Both formats easily fit into embedded automotive hardware with strict memory constraints.

---

## 3. Safety Decision Evaluation (Java Core)

The Java Safety Engine is evaluated on deterministic reaction correctness and latency:

| Scenario | Input Detection | Vehicle Speed | Expected Alert Level | Measured Reaction Time |
| :--- | :--- | :--- | :--- | :--- |
| **Normal Compliance** | 50 km/h Limit | 48 km/h | `INFO` | $< 1\text{ ms}$ |
| **Minor Speeding** | 50 km/h Limit | 55 km/h | `WARNING` | $< 1\text{ ms}$ |
| **Gross Speeding** | 50 km/h Limit | 72 km/h | `CRITICAL` | $< 1\text{ ms}$ |
| **Stop Sign Approach** | Stop Sign | 35 km/h | `CRITICAL` | $< 1\text{ ms}$ |
| **Noisy / Ghost Sign** | Any (Conf: 0.35) | Any | Ignored | $< 1\text{ ms}$ |
