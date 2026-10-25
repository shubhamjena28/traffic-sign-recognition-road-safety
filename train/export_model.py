"""Export trained PyTorch models to ONNX and TorchScript formats.

Facilitates deployment to edge devices, OpenCV DNN, and Java runtime environments.
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).parent))
from train import TrafficSignCNN


def export_model(checkpoint_path: str, output_onnx: str, output_torchscript: str, img_size: int = 48):
    device = torch.device("cpu")
    ckpt_file = Path(checkpoint_path)

    if not ckpt_file.exists():
        print(f"[ERROR] Checkpoint file '{checkpoint_path}' does not exist.")
        return

    checkpoint = torch.load(ckpt_file, map_location=device)
    num_classes = checkpoint.get("num_classes", 43)

    model = TrafficSignCNN(num_classes=num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dummy_input = torch.randn(1, 3, img_size, img_size, device=device)

    # 1. Export to ONNX
    if output_onnx:
        onnx_path = Path(output_onnx)
        onnx_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Exporting model to ONNX: {onnx_path} ...")

        torch.onnx.export(
            model,
            dummy_input,
            str(onnx_path),
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        )

        try:
            import onnx
            onnx_model = onnx.load(str(onnx_path))
            onnx.checker.check_model(onnx_model)
            print(f"  [SUCCESS] ONNX model successfully verified: {onnx_path}")
        except ImportError:
            print("  [NOTE] 'onnx' package not installed for verification, file saved.")
        except Exception as e:
            print(f"  [WARNING] ONNX check warning: {e}")

    # 2. Export to TorchScript
    if output_torchscript:
        ts_path = Path(output_torchscript)
        ts_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Exporting model to TorchScript: {ts_path} ...")

        traced_script_module = torch.jit.trace(model, dummy_input)
        traced_script_module.save(str(ts_path))
        print(f"  [SUCCESS] TorchScript model exported to: {ts_path}")


def main():
    parser = argparse.ArgumentParser(description="Export Traffic Sign Recognition Model")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt", help="Path to .pt checkpoint")
    parser.add_argument("--output-onnx", type=str, default="checkpoints/traffic_sign_model.onnx", help="Output ONNX path")
    parser.add_argument("--output-torchscript", type=str, default="checkpoints/traffic_sign_model_ts.pt", help="Output TorchScript path")
    parser.add_argument("--img-size", type=int, default=48, help="Input image dimension (default: 48)")
    args = parser.parse_args()

    export_model(args.checkpoint, args.output_onnx, args.output_torchscript, args.img_size)


if __name__ == "__main__":
    main()
