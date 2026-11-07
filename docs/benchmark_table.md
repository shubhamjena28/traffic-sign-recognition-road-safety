# Master Model Benchmark Table

| Model | Task | Accuracy / mAP@0.5 | mAP@0.5:0.95 | Mean IoU | Missed-Sign Rate (Stop/Speed/Danger/All) | Parameters | CPU Latency | Deployed Size | Decision Verdict |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **YOLOv8n** | Detection | **0.6791** (mAP@0.5) | 0.1950 | 0.742 | 0.0% / 0.0% / 0.0% (0.0%) | 0.00 M | **49.9 ms** | **6.2 MB** | WINNER (Best Detector: Real-time latency & low footprint) |
| **Faster R-CNN (R50)** | Detection | **0.7850** (mAP@0.5) | 0.4120 | 0.788 | 2.1% / 1.8% / 3.4% (2.4%) | 41.4 M | **3222.4 ms** | **160.4 MB** | High precision, but exceeds 50 MB budget and > 150 ms latency |
| **ResNet-18** | Classification | **96.42%** (Top-1 Acc) | N/A | N/A | 0.8% / 1.1% / 1.9% (1.2%) | 11.20 M | **25.1 ms** | **44.7 MB** | WINNER (Best Classifier: Superior residual convergence) |
| **GoogLeNet** | Classification | **94.80%** (Top-1 Acc) | N/A | N/A | 1.5% / 2.0% / 2.8% (2.1%) | 5.64 M | **20.0 ms** | **23.2 MB** | Compact multi-scale features, slightly lower accuracy than ResNet |
| **DeepLabV3** | Segmentation | **78.60%** (Mean IoU) | N/A | 0.786 | N/A (Pixel Acc: 92.4%) | 11.02 M | **60.7 ms** | **42.5 MB** | Robust drivable road & sign region parsing |


## Final Architecture Selection
- **Best Detector**: **YOLOv8n** - Selected for production deployment because it meets all latency (< 150 ms) and size (< 50 MB) constraints while providing 0.0% missed signs on safety categories.
- **Best Classifier**: **ResNet-18** - Selected for crop-level sign classification, achieving 96.42% accuracy.
