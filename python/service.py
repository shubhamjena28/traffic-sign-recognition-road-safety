"""Lightweight HTTP Inference Service for Traffic Sign Recognition.

Receives road images from the Java application, runs YOLO/CNN inference,
and returns structured detection JSON payloads.
"""

import argparse
import io
import json
import os
import sys
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse

import cv2
import numpy as np

# Import predictor from inference module
sys.path.append(str(Path(__file__).parent))
from inference import TrafficSignPredictor


class InferenceRequestHandler(BaseHTTPRequestHandler):
    predictor: TrafficSignPredictor = None

    def _set_headers(self, status=200, content_type="application/json"):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_OPTIONS(self):
        self._set_headers(200)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/health":
            self._set_headers(200)
            resp = {
                "status": "UP",
                "backend": self.predictor.backend,
                "model": str(self.predictor.model_path)
            }
            self.wfile.write(json.dumps(resp).encode("utf-8"))
        elif parsed.path == "/demo":
            self._set_headers(200)
            dummy = np.zeros((200, 200, 3), dtype=np.uint8)
            detections = self.predictor.predict_detections(dummy)
            self.wfile.write(json.dumps(detections).encode("utf-8"))
        else:
            self._set_headers(404)
            self.wfile.write(json.dumps({"error": "Endpoint not found"}).encode("utf-8"))

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path != "/detect":
            self._set_headers(404)
            self.wfile.write(json.dumps({"error": "Unknown endpoint. Use /detect"}).encode("utf-8"))
            return

        content_length = int(self.headers.get("Content-Length", 0))
        if content_length == 0:
            self._set_headers(400)
            self.wfile.write(json.dumps({"error": "Empty request body"}).encode("utf-8"))
            return

        body = self.rfile.read(content_length)

        try:
            # Decode image from raw binary buffer
            nparr = np.frombuffer(body, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if image is None:
                # Try decoding as JSON payload with filepath if not raw bytes
                try:
                    payload = json.loads(body.decode("utf-8"))
                    image_path = payload.get("image_path")
                    if image_path and os.path.exists(image_path):
                        image = cv2.imread(image_path)
                except Exception:
                    pass

            if image is None:
                self._set_headers(400)
                self.wfile.write(json.dumps({"error": "Failed to decode image from payload"}).encode("utf-8"))
                return

            # Run detection
            detections = self.predictor.predict_detections(image)
            self._set_headers(200)
            self.wfile.write(json.dumps(detections).encode("utf-8"))

        except Exception as e:
            self._set_headers(500)
            self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))

    def log_message(self, format, *args):
        # Override to log concise HTTP events
        print(f"[HTTP Service] {self.address_string()} - {format % args}")


def run_service(model_path: str, host: str = "0.0.0.0", port: int = 5000):
    predictor = TrafficSignPredictor(model_path)
    InferenceRequestHandler.predictor = predictor

    server_address = (host, port)
    httpd = HTTPServer(server_address, InferenceRequestHandler)
    print(f"\n=======================================================")
    print(f" Traffic Sign Inference Service running on http://{host}:{port}")
    print(f" Model Backend: {predictor.backend} ({model_path})")
    print(f" Endpoints:")
    print(f"   - GET  /health   -> Check service status")
    print(f"   - GET  /demo     -> Sample detections JSON")
    print(f"   - POST /detect   -> Send image binary -> receive JSON")
    print(f"=======================================================\n")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping HTTP Inference Service...")
        httpd.server_close()


def find_default_model() -> str:
    candidates = [
        "runs/detect/colab_yolo_model/weights/best.pt",
        "runs/detect/colab_yolo_traffic/weights/best.pt",
        "runs/detect/train-3/weights/best.pt",
        "runs/detect/train/weights/best.pt",
        "yolov8n.pt"
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return "yolov8n.pt"


def main():
    default_model = find_default_model()
    parser = argparse.ArgumentParser(description="Run Python Traffic Sign Inference HTTP Service")
    parser.add_argument("--model", type=str, default=default_model, help="Path to .pt or .onnx model")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Server host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=5000, help="Server port (default: 5000)")
    args = parser.parse_args()

    run_service(args.model, args.host, args.port)


if __name__ == "__main__":
    main()
