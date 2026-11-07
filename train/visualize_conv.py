#!/usr/bin/env python3
"""
Convolutional Neural Network Visualizations & Advanced Demos:
1. 2D Filter Kernels: First-layer learned weights from ResNet-18 (conv1: 7x7).
2. Feature Maps: Intermediate multi-scale activation maps across ResNet residual layers.
3. 3D Convolution Demo: Spatio-temporal convolution (nn.Conv3d) on a 5-frame video stack.
4. Instance Segmentation Demo: Polygonal mask & bounding box overlay demo (Mask R-CNN / YOLO-Seg style).
"""

import argparse
from pathlib import Path
from typing import List

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms


def visualize_2d_kernels(model: nn.Module, save_path: str = "checkpoints/resnet_conv1_kernels.png"):
    """Visualizes 2D filter kernels from the first convolutional layer (conv1)."""
    print("[1/4] Visualizing 2D Convolutional Kernels from ResNet-18 conv1...")
    # Conv1 weights shape: [64, 3, 7, 7]
    conv1_weights = model.conv1.weight.data.clone().cpu()
    num_filters = min(32, conv1_weights.size(0))

    fig, axes = plt.subplots(4, 8, figsize=(12, 6))
    fig.suptitle("ResNet-18 Layer 1 (conv1) 2D Spatial Filter Kernels (7x7 RGB)", fontsize=14)

    for i in range(num_filters):
        ax = axes[i // 8, i % 8]
        kernel = conv1_weights[i].numpy()  # [3, 7, 7]
        # Min-max normalization for visualization
        k_min, k_max = kernel.min(), kernel.max()
        kernel_norm = (kernel - k_min) / (k_max - k_min + 1e-8)
        kernel_rgb = np.transpose(kernel_norm, (1, 2, 0))

        ax.imshow(kernel_rgb)
        ax.axis("off")
        ax.set_title(f"F#{i+1}", fontsize=8)

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"  -> Saved to: {save_path}")


def visualize_feature_maps(model: nn.Module, sample_img: Image.Image, save_path: str = "checkpoints/resnet_feature_maps.png"):
    """Extracts and visualizes intermediate activation feature maps across ResNet residual blocks."""
    print("[2/4] Extracting and Visualizing Feature Maps across ResNet-18 Layers...")

    tf = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    tensor_in = tf(sample_img).unsqueeze(0)

    # Register hooks for conv1, layer1, layer2, layer3, layer4
    activation_maps = {}
    def hook_fn(name):
        def hook(module, inp, out):
            activation_maps[name] = out.detach().cpu()
        return hook

    handles = [
        model.conv1.register_forward_hook(hook_fn("conv1")),
        model.layer1.register_forward_hook(hook_fn("layer1")),
        model.layer2.register_forward_hook(hook_fn("layer2")),
        model.layer3.register_forward_hook(hook_fn("layer3")),
        model.layer4.register_forward_hook(hook_fn("layer4")),
    ]

    model.eval()
    with torch.no_grad():
        _ = model(tensor_in)

    for h in handles:
        h.remove()

    fig, axes = plt.subplots(5, 6, figsize=(14, 11))
    fig.suptitle("ResNet-18 Intermediate Feature Map Activations (Edge -> Texture -> Semantic)", fontsize=14)

    layer_names = ["conv1", "layer1", "layer2", "layer3", "layer4"]
    for row, layer_name in enumerate(layer_names):
        fmap = activation_maps[layer_name][0]  # [C, H, W]
        for col in range(6):
            ax = axes[row, col]
            if col < fmap.size(0):
                ch_map = fmap[col].numpy()
                ax.imshow(ch_map, cmap="magma")
            ax.axis("off")
            if col == 0:
                ax.set_title(f"{layer_name} (Res: {fmap.size(1)}x{fmap.size(2)})", fontsize=9, loc='left')

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"  -> Saved to: {save_path}")


def demo_3d_convolution_temporal_stack(save_path: str = "checkpoints/3d_conv_temporal_stack.png"):
    """Demonstrates Spatio-Temporal 3D Convolution (nn.Conv3d) on a 5-frame video stack."""
    print("[3/4] Running 3D Spatio-Temporal Convolution Demo on 5-Frame Approach Stack...")

    # Create 5 synthetic sequential video frames of a car approaching a STOP sign
    frames = []
    for t in range(5):
        frame = np.ones((112, 112, 3), dtype=np.uint8) * 130
        # Road perspective
        cv2.fillPoly(frame, [np.array([[20, 112], [92, 112], [65, 50], [47, 50]])], (60, 60, 60))
        # Approaching sign grows over time t
        r = 8 + t * 4
        cx, cy = 75, 45 - t * 2
        cv2.circle(frame, (cx, cy), r, (30, 30, 220), -1)
        cv2.circle(frame, (cx, cy), r - 2, (255, 255, 255), 1)
        cv2.putText(frame, "STOP", (cx - r + 3, cy + 3), cv2.FONT_HERSHEY_PLAIN, 0.6 + t * 0.1, (255, 255, 255), 1)
        frames.append(frame)

    # Convert to Tensor of shape [Batch, Channels, Time, Height, Width] -> (1, 3, 5, 112, 112)
    frames_tensor = torch.stack([transforms.functional.to_tensor(f) for f in frames], dim=1).unsqueeze(0)

    # Define 3D Convolution layer
    conv3d = nn.Conv3d(
        in_channels=3,
        out_channels=8,
        kernel_size=(3, 3, 3),
        stride=(1, 1, 1),
        padding=(1, 1, 1)
    )

    with torch.no_grad():
        out3d = conv3d(frames_tensor)  # [1, 8, 5, 112, 112]

    # Visualize original frames + 3D feature response over time
    fig, axes = plt.subplots(2, 5, figsize=(14, 6))
    fig.suptitle("3D Spatio-Temporal Convolution on 5-Frame Traffic Sign Approach Sequence", fontsize=14)

    for t in range(5):
        # Top row: Raw video input frames
        axes[0, t].imshow(frames[t])
        axes[0, t].set_title(f"Frame t={t+1} (Raw Video)")
        axes[0, t].axis("off")

        # Bottom row: 3D Conv activation feature map at time t
        act_map = out3d[0, 0, t].numpy()
        axes[1, t].imshow(act_map, cmap="viridis")
        axes[1, t].set_title(f"3D Conv Output t={t+1}")
        axes[1, t].axis("off")

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"  -> Saved to: {save_path}")


def demo_instance_segmentation(save_path: str = "checkpoints/instance_segmentation_demo.png"):
    """Demonstrates Mask R-CNN / YOLO-Seg Instance Segmentation with polygonal masks."""
    print("[4/4] Generating Instance Segmentation Demo (Bounding Box + Polygonal Mask)...")

    # Synthetic road scene image
    img = np.ones((320, 320, 3), dtype=np.uint8) * 140
    # Road
    cv2.fillPoly(img, [np.array([[0, 320], [320, 320], [200, 140], [120, 140]])], (70, 70, 70))

    # Instance 1: STOP Sign (Octagon)
    cx1, cy1, r1 = 80, 120, 35
    angles = np.linspace(0, 2 * np.pi, 9)[:-1] + (np.pi / 8)
    oct_pts = np.array([[int(cx1 + r1 * np.cos(a)), int(cy1 + r1 * np.sin(a))] for a in angles], np.int32)
    cv2.fillPoly(img, [oct_pts], (20, 20, 220))
    cv2.putText(img, "STOP", (cx1 - 25, cy1 + 6), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 2)

    # Instance 2: Speed Limit Sign (Circle)
    cx2, cy2, r2 = 230, 100, 32
    cv2.circle(img, (cx2, cy2), r2, (20, 20, 220), -1)
    cv2.circle(img, (cx2, cy2), r2 - 6, (255, 255, 255), -1)
    cv2.putText(img, "60", (cx2 - 16, cy2 + 8), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 0), 2)

    # Overlay Instance Segmentation Masks
    overlay = img.copy()
    mask_layer = np.zeros_like(img)

    # Mask 1 (Cyan tint for Stop sign)
    cv2.fillPoly(mask_layer, [oct_pts], (255, 100, 0))
    # Mask 2 (Yellow tint for Speed sign)
    cv2.circle(mask_layer, (cx2, cy2), r2, (0, 230, 255), -1)

    alpha = 0.45
    cv2.addWeighted(mask_layer, alpha, overlay, 1 - alpha, 0, overlay)

    # Draw bounding boxes and class labels
    # Box 1
    cv2.rectangle(overlay, (cx1 - r1, cy1 - r1), (cx1 + r1, cy1 + r1), (0, 255, 0), 2)
    cv2.putText(overlay, "stop: 0.98 [mask]", (cx1 - r1, cy1 - r1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Box 2
    cv2.rectangle(overlay, (cx2 - r2, cy2 - r2), (cx2 + r2, cy2 + r2), (0, 255, 255), 2)
    cv2.putText(overlay, "speed_60: 0.94 [mask]", (cx2 - r2, cy2 - r2 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(save_path, overlay)
    print(f"  -> Saved to: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Run Convolutional Visualizations & Demos")
    parser.add_argument("--sample-img", type=str, default="data/images/val/sample_0_000.jpg", help="Sample input image")
    parser.add_argument("--output-dir", type=str, default="checkpoints", help="Output directory")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load ResNet-18 (fallback to local init if offline)
    try:
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    except Exception:
        resnet = models.resnet18(weights=None)

    # 1. 2D Kernels
    visualize_2d_kernels(resnet, str(out_dir / "resnet_conv1_kernels.png"))

    # 2. Feature Maps
    sample_path = Path(args.sample_img)
    if sample_path.exists():
        sample = Image.open(sample_path).convert("RGB")
    else:
        sample = Image.fromarray(np.random.randint(50, 200, (128, 128, 3), dtype=np.uint8))
    visualize_feature_maps(resnet, sample, str(out_dir / "resnet_feature_maps.png"))

    # 3. 3D Conv Demo
    demo_3d_convolution_temporal_stack(str(out_dir / "3d_conv_temporal_stack.png"))

    # 4. Instance Segmentation Demo
    demo_instance_segmentation(str(out_dir / "instance_segmentation_demo.png"))

    print("\n[SUCCESS] All 4 convolution visualizations and demos generated successfully!")


if __name__ == "__main__":
    main()
