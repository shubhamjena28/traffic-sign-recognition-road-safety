# Model Evaluation & Safety Metrics

This document defines the evaluation framework used to assess perception accuracy and safety monitoring effectiveness.

## 1. Perception Performance Metrics

### Top-1 Classification Accuracy
Measures overall correctness on the held-out test split:
$$\text{Accuracy} = \frac{\sum_{i=1}^{N} \mathbb{I}(\hat{y}_i = y_i)}{N}$$
- Target Benchmark: $\ge 96.0\%$ overall accuracy on GTSRB test set.

### Per-Class Precision, Recall, and F1-Score
Given class imbalances in natural driving datasets (e.g., speed limits occur far more frequently than dangerous curve warnings), evaluate per-class harmonic mean:
$$F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$
- Critical Signs (Stop, Yield, Speed Limit 30/50): Target $F_1 \ge 0.95$.

### Confusion Matrix Analysis
Identifies common confusion pairs (e.g., distinguishing between 60 km/h and 80 km/h under blur or low contrast).

## 2. Safety Decision Evaluation

The Java Safety Engine is evaluated on deterministic reaction correctness and latency:

| Scenario | Input Detection | Vehicle Speed | Expected Alert Level | Expected Reaction Time |
| :--- | :--- | :--- | :--- | :--- |
| **Normal Compliance** | 50 km/h Limit | 48 km/h | `INFO` | $< 1\text{ ms}$ |
| **Minor Speeding** | 50 km/h Limit | 55 km/h | `WARNING` | $< 1\text{ ms}$ |
| **Gross Speeding** | 50 km/h Limit | 72 km/h | `CRITICAL` | $< 1\text{ ms}$ |
| **Stop Sign Approach** | Stop Sign | 35 km/h | `CRITICAL` | $< 1\text{ ms}$ |
| **Noisy / Ghost Sign** | Any (Conf: 0.35) | Any | Ignored | $< 1\text{ ms}$ |

## 3. Real-Time Latency Benchmarks

- **Target Perception Latency**: $\le 30\text{ ms}$ per frame (allowing $> 30\text{ FPS}$ live processing).
- **Target Safety Engine Latency**: $\le 2\text{ ms}$ per detection cycle.
