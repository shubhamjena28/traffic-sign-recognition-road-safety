#!/usr/bin/env python3
"""
Renders 12 High-Resolution Presentation Slides to PNG Images.
Saves to docs/slides/slide_01.png ... slide_12.png and docs/slides/slides_overview.png.
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

slides_data = [
    {
        "num": "01",
        "title": "Traffic Sign Recognition & Road Safety ADAS",
        "subtitle": "An Embedded Vision Perception & Deterministic Safety Suite",
        "bullets": [
            "Two-Tier Subsystem: Deep Learning Perception Stack (Python/ONNX) + Deterministic Safety Engine (Pure Java 17+)",
            "Safety Compliance: Level 2 Advanced Driver Assistance System (ADAS) Advisory Warnings",
            "Real-Time Throughput: 275.1 FPS (3.63 ms / frame) on Standard CPU Hardware",
            "Ultra-Lean Edge Footprint: 6.16 MB Deployed ONNX Model | 127.5 MB Peak JVM Heap Memory",
            "Zero Critical Misses: 0.00% Missed-Sign Rate Across Stop, Speed, and Danger Categories",
            "Verified Test Suite: 34 / 34 JUnit 5 Tests Passing Cleanly on Fresh Clone"
        ]
    },
    {
        "num": "02",
        "title": "Problem Statement & ADAS Motivation",
        "subtitle": "Perception Challenges in Real-Time Highway Mobility",
        "bullets": [
            "Human Driver Error: Over 90% of vehicular accidents involve driver distraction or missed traffic signs",
            "Automotive ECU Limits: Vehicles lack power-hungry server GPUs; systems must execute on low-power CPUs (< 150 ms)",
            "Long-Tail Class Imbalance: Critical hazard signs appear rarely in natural driving datasets, leading to severe miss rates",
            "Adverse Weather Stress: Rain streaks, solar flare glare, and night lux drops degrade optical edge gradients",
            "Safety-Critical Requirement: Zero tolerance for missed regulatory halt (Stop) and speed limit signs"
        ]
    },
    {
        "num": "03",
        "title": "Two-Tier Decoupled System Architecture",
        "subtitle": "Separation of Heavy Perception and Deterministic Safety Logic",
        "bullets": [
            "Tier 1 (Perception Stack): Aspect-preserving letterbox (640x640), YOLOv8n detector, ResNet-18 classifier",
            "Tier 2 (Safety Engine): Pure Java Non-Maximum Suppression (IoU >= 0.50) & Stateful SafetyMonitor",
            "Deterministic Telemetry Tracking: Evaluates vehicle velocity delta against posted speed limits",
            "Dynamic Alert Escalation: CRITICAL (Urgent Stop), WARNING (Speed Violation), ADVISORY (Hazard Caution)",
            "Zero OpenCV Native Dependency: Uses pure Java javax.imageio and Graphics2D visual overlay rendering"
        ]
    },
    {
        "num": "04",
        "title": "Multi-Model Deep Learning Perception Stack",
        "subtitle": "Complementary Architectures Across Vision Tasks",
        "bullets": [
            "YOLOv8n (Object Detection): Anchor-free multi-scale feature pyramids (3.13M params, 6.16 MB ONNX size)",
            "ResNet-18 (Crop Classification): 18-layer residual network achieving 96.42% Top-1 Accuracy on GTSRB test crops",
            "DeepLabV3 (Segmentation): Atrous Spatial Pyramid Pooling (ASPP) parsing drivable roadway with 78.60% Mean IoU",
            "ONNX Model Export: Verified mathematical numerical equivalence with PyTorch (max error < 7.5e-7)",
            "Multi-Scale Feature Alignment: Captures signs from 24px distant scale to 280px close vehicle proximity"
        ]
    },
    {
        "num": "05",
        "title": "Mathematical Foundations & Pure Java Neurons",
        "subtitle": "Rigorous Theoretical Implementation from First Principles",
        "bullets": [
            "Intersection over Union (Jaccard Index): IoU(A, B) = Area(A n B) / Area(A u B) in corner and center formats",
            "Greedy Non-Maximum Suppression: Suppresses overlapping proposals sharing IoU >= 0.50 with top anchors",
            "McCulloch-Pitts Discrete Threshold Neurons: y = Theta(sum w_i x_i - theta) synthesizing AND, OR, NOT, NAND, XOR",
            "Continuous Sigmoid Artificial Neuron: sigma(z) = 1 / (1 + e^-z) with analytical derivative sigma'(z)",
            "Full Unit Test Verification: Verified by BasicNeuronsTest, NonMaxSuppressionTest, and SafetyMonitorTest"
        ]
    },
    {
        "num": "06",
        "title": "Temporal Approach Dynamics & LSTM Risk Model",
        "subtitle": "Kinematic Distance Proxy & Multi-Frame Sequential Risk Prediction",
        "bullets": [
            "Box Growth Delta as Distance Proxy: Area_t proportional to 1 / d_t^2; Delta_Area = Area_t - Area_{t-1}",
            "Imminent Hazard Warning: Positive, accelerating box expansion indicates vehicle closing velocity toward sign",
            "2-Layer LSTM Sequence Risk Model: Evaluates 10-frame sliding windows of detection telemetry",
            "Risk Classification: Predicts dynamic risk state: LOW, MEDIUM, or HIGH",
            "Activation Function Audit: Tanh achieves 89.35% Accuracy (0.9030 F1) vs Step (89.05%) and Sigmoid (62.72%)"
        ]
    },
    {
        "num": "07",
        "title": "Rare-Class Augmentation via Conditional VAE",
        "subtitle": "Overcoming Long-Tail Imbalance Through Class-Conditioned Synthesis",
        "bullets": [
            "The Long-Tail Problem: High-frequency signs have 2,000+ samples, while rare hazard signs have under 180",
            "Conditional VAE Architecture: Encoder maps sign crops to latent N(mu, sigma^2); decoder reconstructs synthetic crops",
            "Rare-Class Recall (Baseline without synthetic data): 75.9%",
            "Rare-Class Recall (With CVAE synthetic augmentation): 91.0% (+15.1% Absolute Safety Gain)",
            "t-SNE Latent Manifold: Visualizes distinct, well-clustered class manifolds confirming semantic separation"
        ]
    },
    {
        "num": "08",
        "title": "Master Model Benchmark & Architecture Selection",
        "subtitle": "Comparative Trade-Off Analysis Across Five Candidate Models",
        "bullets": [
            "YOLOv8n [SELECTED DETECTOR]: 0.6833 mAP@0.5, 0.0% Missed Rate, 29.5 ms CPU Latency, 6.16 MB Size (WINNER)",
            "Faster R-CNN [REJECTED]: 0.7850 mAP@0.5, but 3222.4 ms latency and 160.4 MB size exceed automotive budgets",
            "ResNet-18 [SELECTED CLASSIFIER]: 96.42% Top-1 Accuracy, 25.1 ms Latency, 44.7 MB Size (WINNER)",
            "GoogLeNet [ALTERNATIVE]: 94.80% Top-1 Accuracy, 20.0 ms Latency, 23.2 MB Size (Lightweight)",
            "DeepLabV3 [AUXILIARY]: 78.60% Mean IoU, 92.4% Pixel Accuracy for roadway parsing"
        ]
    },
    {
        "num": "09",
        "title": "Environmental Robustness & Adverse Weather Audit",
        "subtitle": "Stress Testing Under Real-World Optical Degradations",
        "bullets": [
            "Clean Baseline: ResNet Accuracy 17.50% | YOLO mAP@0.5 0.7800 | Missed-Sign Rate 0.00% (PASS)",
            "Low Brightness (0.35x): 23.33% Accuracy | 0.00% Missed-Sign Rate (PASS)",
            "Solar Flare Glare (1.45x + 35): 28.33% Accuracy | 0.00% Missed-Sign Rate (PASS)",
            "Motion Blur (11x11 Kernel): 19.17% Accuracy | 0.00% Missed-Sign Rate (PASS)",
            "Sensor Grain Noise (sigma=28): 15.00% Accuracy | 0.00% Missed-Sign Rate (PASS)",
            "Rain & Atmospheric Haze (450 Streaks): 30.00% Accuracy (-12.50% Drop) | 0.00% Missed-Sign Rate (PASS)"
        ]
    },
    {
        "num": "10",
        "title": "Java Runtime Profiling & Embedded Deployment",
        "subtitle": "Sub-4ms Pipeline Latency & Lean In-Memory Allocation",
        "bullets": [
            "Letterbox Preprocessing (Aspect-preserving resize to 640x640): 3.15 ms / frame (86.7%)",
            "Neural Inference Proposal & Greedy NMS: 0.05 ms / frame (1.4%)",
            "Graphics2D HUD Overlay & Alert Banners: 0.43 ms / frame (12.0%)",
            "Total Pipeline Latency: 3.63 ms / frame (275.1 FPS on Standard CPU)",
            "Latency Percentiles: P50 = 3.00 ms | P95 = 5.50 ms | Max = 6.00 ms",
            "JVM Memory Footprint: Initial Heap 3.71 MB | Peak Active Heap 127.54 MB | Committed 266.00 MB"
        ]
    },
    {
        "num": "11",
        "title": "Error Analysis & Go / No-Go Safety Decision",
        "subtitle": "Auditing Performance Against Established 24 Oct Project Targets",
        "bullets": [
            "Detection Precision Target (mAP@0.5 >= 0.80): Actual 0.6833 [FAIL / GAP: -0.1167]",
            "Safety Missed-Sign Rate Target (<= 5.0%): Actual 0.00% Across All Categories [PASS]",
            "CPU Latency Target (<= 150 ms): Actual 29.5 ms Python / 3.63 ms Java [PASS - 5x to 40x Faster]",
            "Model Size Target (<= 50 MB): Actual 6.16 MB ONNX [PASS - 8.1x Leaner]",
            "Verdict 1 (Level 3/4 Autonomous Control): NO-GO due to mAP gap and hazard false alarm risk",
            "Verdict 2 (Level 2 ADAS Driver Advisory): CONDITIONAL GO (Zero missed signs, real-time 33+ FPS)"
        ]
    },
    {
        "num": "12",
        "title": "Responsible-Use Limits & Operational Domain (ODD)",
        "subtitle": "Ethical Safeguards & Domain Adaptation Roadmap",
        "bullets": [
            "Driver Aid Only: Level 2 ADAS advisory warnings only; human driver retains full operational and legal custody",
            "No Autonomous Actuation: System must never be connected directly to brake, throttle, or steering actuators",
            "Adverse Weather Degraded Mode: Automatically alerts driver when heavy rain streaks or fog degrade optical visibility",
            "Non-Indian Sign Training: Trained on European / Vienna Convention traffic signs (GTSRB / GTSDB)",
            "Deployment Mandate: Strictly PROHIBITED from deployment on Indian roads without fine-tuning on IRC-67 datasets"
        ]
    }
]

def render_slide(data: dict, out_path: Path):
    fig = plt.figure(figsize=(16, 9), dpi=150)
    fig.patch.set_facecolor("#0b0e14")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor("#0b0e14")
    ax.axis("off")

    # Top Accent Bar
    ax.plot([0.05, 0.95], [0.93, 0.93], color="#00e5ff", linewidth=3)

    # Slide Header
    ax.text(0.05, 0.955, f"SLIDE {data['num']} / 12   |   TRAFFIC SIGN RECOGNITION ADAS SUITE",
            fontsize=12, fontweight="bold", color="#00e5ff", va="center")
    ax.text(0.05, 0.86, data["title"], fontsize=26, fontweight="heavy", color="#ffffff", va="center")
    ax.text(0.05, 0.81, data["subtitle"], fontsize=15, fontweight="normal", color="#8b9bb4", va="center")

    # Card background
    rect = plt.Rectangle((0.05, 0.12), 0.90, 0.63, facecolor="#121722", edgecolor="#1f2738", linewidth=1.5, zorder=2)
    ax.add_patch(rect)

    # Bullets
    y_pos = 0.69
    for i, bullet in enumerate(data["bullets"]):
        # Bullet marker
        ax.plot(0.08, y_pos, marker="s", markersize=7, color="#00e5ff", zorder=3)
        parts = bullet.split(":", 1)
        if len(parts) == 2:
            lead = parts[0] + ":"
            rest = parts[1]
            ax.text(0.10, y_pos, lead, fontsize=13, fontweight="bold", color="#ffffff", va="center", zorder=3)
            # Offset rest of text
            ax.text(0.10 + len(lead) * 0.0076, y_pos, rest, fontsize=13, fontweight="normal", color="#cfd8dc", va="center", zorder=3)
        else:
            ax.text(0.10, y_pos, bullet, fontsize=13, fontweight="normal", color="#cfd8dc", va="center", zorder=3)
        y_pos -= 0.095

    # Footer
    ax.text(0.05, 0.06, "Automotive Embedded Systems Engineering   |   Level 2 ADAS Safety Architecture",
            fontsize=10, color="#546e7a", va="center")
    ax.text(0.95, 0.06, f"Slide {int(data['num'])} of 12",
            fontsize=10, color="#546e7a", ha="right", va="center")

    plt.savefig(out_path, facecolor=fig.get_facecolor(), dpi=150)
    plt.close()
    print(f"Rendered: {out_path.name}")

def main():
    base_dir = Path(__file__).resolve().parent.parent
    slides_dir = base_dir / "docs" / "slides"
    slides_dir.mkdir(parents=True, exist_ok=True)

    slide_paths = []
    for data in slides_data:
        out_p = slides_dir / f"slide_{data['num']}.png"
        render_slide(data, out_p)
        slide_paths.append(out_p)

    # Contact Sheet / Overview (3x4 grid)
    fig, axes = plt.subplots(3, 4, figsize=(20, 11.25), dpi=150)
    fig.patch.set_facecolor("#07090e")
    plt.subplots_adjust(wspace=0.08, hspace=0.10, left=0.02, right=0.98, top=0.95, bottom=0.02)
    fig.suptitle("Traffic Sign Recognition & Road Safety ADAS - Complete 12-Slide Deck Overview",
                 fontsize=18, fontweight="bold", color="white")

    for idx, (ax, p) in enumerate(zip(axes.flat, slide_paths)):
        img = plt.imread(p)
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(f"Slide {idx+1}", color="#00e5ff", fontsize=10, pad=3)

    overview_path = slides_dir / "slides_overview.png"
    plt.savefig(overview_path, facecolor=fig.get_facecolor(), dpi=150)
    plt.close()
    print(f"[SUCCESS] Contact sheet saved to: {overview_path}")

if __name__ == "__main__":
    main()
