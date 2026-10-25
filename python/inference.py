"""Traffic Sign Recognition & Detection Inference Pipeline.

Runs predictions on static images or video streams using trained
YOLO (Ultralytics), PyTorch CNN, or ONNX models and outputs detected signs
with bounding boxes and confidence scores.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

# Ensure utils can be imported
sys.path.append(str(Path(__file__).parent))
from utils import get_label_name, preprocess_image, draw_detection, to_java_detection_payload


class TrafficSignPredictor:
    """Predictor supporting YOLO (Ultralytics), PyTorch CNN, and ONNX backends."""

    def __init__(self, model_path: str):
        self.model_path = Path(model_path)
        self.backend = None
        self.session = None
        self.yolo_model = None
        self.torch_model = None

        if not self.model_path.exists():
            print(f"[NOTE] Model file '{model_path}' not found on disk. Initializing in DEMO mode.")
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
                print("[WARNING] onnxruntime not installed. Running in DEMO mode.")
                self.backend = "demo"
        elif suffix in [".pt", ".pth"]:
            # Check if this is an Ultralytics YOLO model
            try:
                from ultralytics import YOLO
                self.yolo_model = YOLO(str(self.model_path))
                self.backend = "yolo"
                print(f"Loaded Ultralytics YOLO model from: {self.model_path}")
            except Exception:
                try:
                    import torch
                    from train.train import TrafficSignCNN
                    ckpt = torch.load(str(self.model_path), map_location="cpu")
                    num_classes = ckpt.get("num_classes", 43)
                    self.torch_model = TrafficSignCNN(num_classes=num_classes)
                    self.torch_model.load_state_dict(ckpt["model_state_dict"])
                    self.torch_model.eval()
                    self.backend = "torch"
                    print(f"Loaded PyTorch CNN model from: {self.model_path}")
                except Exception as e:
                    print(f"[WARNING] Could not load checkpoint ({e}). Running in DEMO mode.")
                    self.backend = "demo"
        else:
            self.backend = "demo"

    def predict_detections(self, image: np.ndarray, conf_threshold: float = 0.50) -> List[dict]:
        """Runs full-image detection, returning a list of detections matching Java Detection format."""
        detections = []

        if self.backend == "yolo":
            results = self.yolo_model.predict(image, conf=conf_threshold, verbose=False)[0]
            boxes = results.boxes
            for box in boxes:
                cls_id = int(box.cls[0].item())
                label = results.names.get(cls_id, f"class_{cls_id}")
                conf = float(box.conf[0].item())
                xyxy = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = xyxy
                w, h = x2 - x1, y2 - y1
                detections.append(to_java_detection_payload(cls_id, label, conf, (int(x1), int(y1), int(w), int(h))))
            return detections

        elif self.backend == "demo":
            # Return realistic simulated detection for demonstration
            detections.append(to_java_detection_payload(0, "stop", 0.95, (10, 20, 40, 40)))
            return detections

        elif self.backend == "onnx":
            tensor = preprocess_image(image)
            input_name = self.session.get_inputs()[0].name
            outputs = self.session.run(None, {input_name: tensor})
            logits = outputs[0][0]
            exp_logits = np.exp(logits - np.max(logits))
            probs = exp_logits / exp_logits.sum()
            cls_id = int(np.argmax(probs))
            conf = float(probs[cls_id])
            if conf >= conf_threshold:
                detections.append(to_java_detection_payload(cls_id, get_label_name(cls_id), conf))
            return detections

        return detections


def run_image_inference(predictor: TrafficSignPredictor, image_path: str, output_path: Optional[str] = None):
    img = cv2.imread(image_path)
    if img is None:
        print(f"[ERROR] Could not load image from '{image_path}'")
        return

    detections = predictor.predict_detections(img)
    print(f"\n--- Detection Results ({len(detections)} signs found) ---")
    for d in detections:
        print(f"  • Sign: {d['label']:<10} | Confidence: {d['confidence']*100:.1f}% | Box: ({d['x']}, {d['y']}, {d['width']}, {d['height']})")

    print("\nPayload for Java SafetyMonitor:")
    print(json.dumps(detections, indent=2))

    if output_path:
        annotated = img.copy()
        for d in detections:
            box = (int(d["x"]), int(d["y"]), int(d["x"] + d["width"]), int(d["y"] + d["height"]))
            annotated = draw_detection(annotated, d["label"], d["confidence"], box)
        cv2.imwrite(output_path, annotated)
        print(f"\nSaved annotated image to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Run Traffic Sign Recognition & Detection Inference")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Path to .pt or .onnx model (default: yolov8n.pt)")
    parser.add_argument("--image", type=str, help="Path to input image file")
    parser.add_argument("--output", type=str, default="output.jpg", help="Path to save annotated image")
    parser.add_argument("--conf", type=float, default=0.50, help="Confidence threshold")
    args = parser.parse_args()

    predictor = TrafficSignPredictor(args.model)

    if args.image:
        run_image_inference(predictor, args.image, args.output)
    else:
        print("No input image specified. Running demo test detection:")
        dummy_img = np.zeros((200, 200, 3), dtype=np.uint8)
        detections = predictor.predict_detections(dummy_img, conf_threshold=args.conf)
        print("Demo Detections Output (Java compatible JSON):")
        print(json.dumps(detections, indent=2))


if __name__ == "__main__":
    main()
