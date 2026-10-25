"""Traffic Sign Recognition Inference Pipeline.

Runs predictions on static images or video streams using trained
PyTorch or ONNX models and outputs detected signs with confidence scores.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

# Ensure utils can be imported
sys.path.append(str(Path(__file__).parent))
from utils import get_label_name, preprocess_image, draw_detection, to_java_detection_payload


class TrafficSignPredictor:
    """Predictor supporting PyTorch and ONNX backends."""

    def __init__(self, model_path: str):
        self.model_path = Path(model_path)
        self.backend = None
        self.session = None
        self.torch_model = None

        if not self.model_path.exists():
            print(f"[WARNING] Model file '{model_path}' not found. Initializing in DEMO mode.")
            self.backend = "demo"
            return

        suffix = self.model_path.suffix.lower()
        if suffix == ".onnx":
            try:
                import onnxruntime as ort
                self.session = ort.InferenceSession(str(self.model_path))
                self.backend = "onnx"
                print(f"Loaded ONNX model from: {self.model_path}")
            except ImportError:
                print("[WARNING] onnxruntime not installed. Please install onnxruntime to run .onnx models.")
                self.backend = "demo"
        elif suffix in [".pt", ".pth"]:
            try:
                import torch
                from train.train import TrafficSignCNN
                ckpt = torch.load(str(self.model_path), map_location="cpu")
                num_classes = ckpt.get("num_classes", 43)
                self.torch_model = TrafficSignCNN(num_classes=num_classes)
                self.torch_model.load_state_dict(ckpt["model_state_dict"])
                self.torch_model.eval()
                self.backend = "torch"
                print(f"Loaded PyTorch model from: {self.model_path}")
            except Exception as e:
                print(f"[WARNING] Could not load PyTorch checkpoint ({e}). Running in DEMO mode.")
                self.backend = "demo"
        else:
            self.backend = "demo"

    def predict(self, image: np.ndarray) -> Tuple[int, str, float]:
        """Runs inference on a single BGR image crop containing a traffic sign.

        Returns (class_id, label_name, confidence).
        """
        if self.backend == "demo":
            # Return realistic simulated detection for demonstration
            return 2, get_label_name(2), 0.94

        tensor = preprocess_image(image)

        if self.backend == "onnx":
            input_name = self.session.get_inputs()[0].name
            outputs = self.session.run(None, {input_name: tensor})
            logits = outputs[0][0]
            exp_logits = np.exp(logits - np.max(logits))
            probs = exp_logits / exp_logits.sum()
            class_id = int(np.argmax(probs))
            conf = float(probs[class_id])
            return class_id, get_label_name(class_id), conf

        elif self.backend == "torch":
            import torch
            with torch.no_grad():
                t = torch.from_numpy(tensor)
                out = self.torch_model(t)
                probs = torch.softmax(out, dim=1).cpu().numpy()[0]
                class_id = int(np.argmax(probs))
                conf = float(probs[class_id])
                return class_id, get_label_name(class_id), conf

        return -1, "Unknown", 0.0


def run_image_inference(predictor: TrafficSignPredictor, image_path: str, output_path: Optional[str] = None):
    img = cv2.imread(image_path)
    if img is None:
        print(f"[ERROR] Could not load image from '{image_path}'")
        return

    class_id, label, conf = predictor.predict(img)
    payload = to_java_detection_payload(class_id, label, conf)

    print("\n--- Detection Result ---")
    print(f"Class ID:   {class_id}")
    print(f"Sign Label: {label}")
    print(f"Confidence: {conf * 100:.2f}%")
    print(f"Java JSON:  {json.dumps(payload)}")

    annotated = draw_detection(img, label, conf)
    if output_path:
        cv2.imwrite(output_path, annotated)
        print(f"Saved annotated image to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Run Traffic Sign Recognition Inference")
    parser.add_argument("--model", type=str, default="../train/checkpoints/best_model.pt", help="Path to .pt or .onnx model")
    parser.add_argument("--image", type=str, help="Path to input image file")
    parser.add_argument("--output", type=str, default="output.jpg", help="Path to save annotated image")
    args = parser.parse_args()

    predictor = TrafficSignPredictor(args.model)

    if args.image:
        run_image_inference(predictor, args.image, args.output)
    else:
        print("No input image specified. Running demo test inference:")
        dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
        class_id, label, conf = predictor.predict(dummy_img)
        payload = to_java_detection_payload(class_id, label, conf)
        print(f"Demo Detection: {label} (conf: {conf:.2f})")
        print(f"JSON Payload:   {json.dumps(payload, indent=2)}")


if __name__ == "__main__":
    main()
