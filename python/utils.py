"""Utility functions and label definitions for Traffic Sign Recognition."""

from typing import Dict, Optional, Tuple
import cv2
import numpy as np

# Standard 43 classes from the German Traffic Sign Recognition Benchmark (GTSRB)
GTSRB_LABELS: Dict[int, str] = {
    0: "Speed limit (20km/h)",
    1: "Speed limit (30km/h)",
    2: "Speed limit (50km/h)",
    3: "Speed limit (60km/h)",
    4: "Speed limit (70km/h)",
    5: "Speed limit (80km/h)",
    6: "End of speed limit (80km/h)",
    7: "Speed limit (100km/h)",
    8: "Speed limit (120km/h)",
    9: "No passing",
    10: "No passing for vehicles over 3.5 metric tons",
    11: "Right-of-way at next intersection",
    12: "Priority road",
    13: "Yield",
    14: "Stop",
    15: "No vehicles",
    16: "Vehicles over 3.5 metric tons prohibited",
    17: "No entry",
    18: "General caution",
    19: "Dangerous curve to the left",
    20: "Dangerous curve to the right",
    21: "Double curve",
    22: "Bumpy road",
    23: "Slippery road",
    24: "Road narrows on the right",
    25: "Road work",
    26: "Traffic signals",
    27: "Pedestrians",
    28: "Children crossing",
    29: "Bicycles crossing",
    30: "Beware of ice/snow",
    31: "Wild animals crossing",
    32: "End of all speed and passing limits",
    33: "Turn right ahead",
    34: "Turn left ahead",
    35: "Ahead only",
    36: "Go straight or right",
    37: "Go straight or left",
    38: "Keep right",
    39: "Keep left",
    40: "Roundabout mandatory",
    41: "End of no passing",
    42: "End of no passing by vehicles over 3.5 metric tons",
}


def get_label_name(class_id: int) -> str:
    """Returns human-readable sign label for a class ID."""
    return GTSRB_LABELS.get(class_id, f"Unknown Sign ({class_id})")


def preprocess_image(image: np.ndarray, target_size: Tuple[int, int] = (48, 48)) -> np.ndarray:
    """Preprocesses an OpenCV BGR image for model inference.

    Returns a normalized float32 tensor of shape (1, 3, H, W).
    """
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, target_size, interpolation=cv2.INTER_AREA)

    # Normalize to [0, 1] then standardize with ImageNet mean/std
    normalized = resized.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    standardized = (normalized - mean) / std

    # HWC to CHW and add batch dimension
    transposed = np.transpose(standardized, (2, 0, 1))
    batched = np.expand_dims(transposed, axis=0)
    return batched


def draw_detection(
    image: np.ndarray,
    label: str,
    confidence: float,
    box: Optional[Tuple[int, int, int, int]] = None,
    color: Tuple[int, int, int] = (0, 255, 0),
) -> np.ndarray:
    """Draws bounding box and prediction text onto image."""
    annotated = image.copy()
    h, w = annotated.shape[:2]

    if box is not None:
        x1, y1, x2, y2 = box
    else:
        x1, y1, x2, y2 = 10, 10, w - 10, h - 10

    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

    text = f"{label} ({confidence * 100:.1f}%)"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    text_y = max(y1 - 8, th + 8)
    cv2.rectangle(annotated, (x1, text_y - th - 4), (x1 + tw + 6, text_y + 4), color, -1)
    cv2.putText(annotated, text, (x1 + 3, text_y), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

    return annotated


def to_java_detection_payload(class_id: int, label: str, confidence: float, box: Optional[Tuple[int, int, int, int]] = None) -> dict:
    """Formats detection output into a dictionary matching Java Core's Detection class."""
    import time
    x, y, w, h = box if box else (0, 0, 0, 0)
    return {
        "classId": class_id,
        "label": label,
        "confidence": round(float(confidence), 4),
        "x": float(x),
        "y": float(y),
        "width": float(w),
        "height": float(h),
        "timestamp": int(time.time() * 1000),
    }
