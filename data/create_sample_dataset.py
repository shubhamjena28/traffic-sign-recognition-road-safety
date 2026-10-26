"""Generates a small starter dataset with annotated traffic signs (stop, speed, danger)

in standard YOLO format so you can train and test immediately.
"""

from pathlib import Path
import cv2
import numpy as np


def create_stop_sign(img_size=320):
    """Draws a synthetic STOP sign on a natural road background."""
    img = np.random.randint(100, 160, (img_size, img_size, 3), dtype=np.uint8)
    cx, cy, r = img_size // 2, img_size // 2, 70

    # Draw Octagon
    angles = np.linspace(0, 2 * np.pi, 9)[:-1] + (np.pi / 8)
    pts = np.array([[int(cx + r * np.cos(a)), int(cy + r * np.sin(a))] for a in angles], np.int32)
    pts = pts.reshape((-1, 1, 2))

    cv2.fillPoly(img, [pts], (20, 20, 220))
    cv2.polylines(img, [pts], True, (255, 255, 255), 4)

    # Text
    cv2.putText(img, "STOP", (cx - 42, cy + 12), cv2.FONT_HERSHEY_DUPLEX, 1.1, (255, 255, 255), 3, cv2.LINE_AA)

    # Bounding box in YOLO format (norm cx, cy, w, h)
    bbox = (cx / img_size, cy / img_size, (2 * r) / img_size, (2 * r) / img_size)
    return img, 0, bbox


def create_speed_sign(img_size=320, speed="50"):
    """Draws a synthetic Speed Limit sign."""
    img = np.random.randint(100, 160, (img_size, img_size, 3), dtype=np.uint8)
    cx, cy, r = img_size // 2, img_size // 2, 70

    # Red outer circle, white inner
    cv2.circle(img, (cx, cy), r, (20, 20, 220), -1)
    cv2.circle(img, (cx, cy), r - 14, (255, 255, 255), -1)

    # Speed number
    cv2.putText(img, speed, (cx - 28, cy + 14), cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 0, 0), 3, cv2.LINE_AA)

    bbox = (cx / img_size, cy / img_size, (2 * r) / img_size, (2 * r) / img_size)
    return img, 1, bbox


def create_danger_sign(img_size=320):
    """Draws a synthetic Danger / Warning triangle sign."""
    img = np.random.randint(100, 160, (img_size, img_size, 3), dtype=np.uint8)
    cx, cy, r = img_size // 2, img_size // 2, 75

    # Equilateral triangle pointing up
    pt1 = (cx, cy - r)
    pt2 = (int(cx - r * 0.9), int(cy + r * 0.7))
    pt3 = (int(cx + r * 0.9), int(cy + r * 0.7))
    pts = np.array([pt1, pt2, pt3], np.int32).reshape((-1, 1, 2))

    cv2.fillPoly(img, [pts], (20, 20, 220))

    # Inner white triangle
    in_pt1 = (cx, cy - r + 16)
    in_pt2 = (int(cx - r * 0.75), int(cy + r * 0.58))
    in_pt3 = (int(cx + r * 0.75), int(cy + r * 0.58))
    in_pts = np.array([in_pt1, in_pt2, in_pt3], np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(img, [in_pts], (255, 255, 255))

    # Exclamation mark
    cv2.line(img, (cx, cy - 20), (cx, cy + 10), (0, 0, 0), 5)
    cv2.circle(img, (cx, cy + 24), 3, (0, 0, 0), -1)

    bbox = (cx / img_size, cy / img_size, (2 * r * 0.9) / img_size, (1.7 * r) / img_size)
    return img, 2, bbox


def generate_dataset(base_dir: Path):
    """Generates images and YOLO txt labels for train and val sets."""
    generators = [create_stop_sign, create_speed_sign, create_danger_sign]

    for split, count in [("train", 30), ("val", 9)]:
        img_dir = base_dir / "images" / split
        lbl_dir = base_dir / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for i in range(count):
            gen_func = generators[i % len(generators)]
            img, cls_id, (cx, cy, w, h) = gen_func()

            filename = f"sample_{cls_id}_{i:03d}"
            img_path = img_dir / f"{filename}.jpg"
            lbl_path = lbl_dir / f"{filename}.txt"

            cv2.imwrite(str(img_path), img)
            with open(lbl_path, "w") as f:
                f.write(f"{cls_id} {cx:.4f} {cy:.4f} {w:.4f} {h:.4f}\n")

    print(f"Sample YOLO dataset successfully created at: {base_dir}")
    print(f"  - Train: {base_dir / 'images' / 'train'} (30 images)")
    print(f"  - Val:   {base_dir / 'images' / 'val'} (9 images)")


if __name__ == "__main__":
    data_dir = Path(__file__).parent
    generate_dataset(data_dir)
