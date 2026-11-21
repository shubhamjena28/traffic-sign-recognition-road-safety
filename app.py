"""Traffic Sign Recognition & Road Safety Monitor.

Streamlit GUI for College PBL Viva & Demonstration.
Clean, minimal, and beginner-friendly implementation.
"""

import os
import sys
import time
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

# Locate project directory
ROOT_DIR = Path(__file__).resolve().parent
if not (ROOT_DIR / "runs").exists() and not (ROOT_DIR / "checkpoints").exists():
    fallback_proj = Path(r"C:\Users\HP\Desktop\Projects\traffic-sign-recognition-road-safety")
    if fallback_proj.exists():
        ROOT_DIR = fallback_proj

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# ==========================================
# PAGE CONFIGURATION & DARK THEME CSS
# ==========================================
st.set_page_config(
    page_title="Traffic Sign Recognition & Road Safety Monitor",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* Dark dashboard theme */
    .stApp {
        background-color: #0B0F19;
        color: #E2E8F0;
        font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, sans-serif;
    }
    
    /* Clean metric cards */
    .metric-box {
        background: #111827;
        border-radius: 8px;
        border-left: 4px solid #00D2FF;
        padding: 12px 14px;
        text-align: center;
        min-height: 82px;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #FFFFFF;
        line-height: 1.1;
    }
    .metric-label {
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: #94A3B8;
        margin-top: 4px;
    }
    
    /* Simple safety alert banners */
    .alert-banner {
        border-radius: 8px;
        padding: 12px 16px;
        margin: 8px 0;
        display: flex;
        align-items: center;
        gap: 14px;
    }
    .alert-stop {
        background: rgba(255, 42, 85, 0.12);
        border-left: 5px solid #FF2A55;
    }
    .alert-speed {
        background: rgba(0, 210, 255, 0.12);
        border-left: 5px solid #00D2FF;
    }
    .alert-danger {
        background: rgba(255, 165, 0, 0.12);
        border-left: 5px solid #FFA500;
    }
    .alert-clear {
        background: rgba(0, 230, 118, 0.12);
        border-left: 5px solid #00E676;
    }
    
    h1, h2, h3, h4 {
        color: #FFFFFF !important;
        font-weight: 600 !important;
    }
    
    [data-testid="stSidebar"] {
        background-color: #0E1422;
        border-right: 1px solid #1E293B;
    }
</style>
""", unsafe_allow_html=True)


# ==========================================
# MODEL LOADING & CACHING
# ==========================================
@st.cache_resource(show_spinner="Loading YOLOv8 Model...")
def load_model():
    """Loads and caches the trained YOLOv8 traffic sign model."""
    from ultralytics import YOLO
    model_path = ROOT_DIR / "runs" / "detect" / "train-3" / "weights" / "best.pt"
    if not model_path.exists():
        model_path = ROOT_DIR / "checkpoints" / "best_yolo.pt"
    if not model_path.exists():
        model_path = ROOT_DIR / "yolov8n.pt"
    return YOLO(str(model_path))

yolo_model = load_model()


# Color mapping for sign categories
SIGN_COLORS = {
    "stop": (50, 40, 245),      # Red (BGR)
    "speed": (255, 200, 0),     # Cyan (BGR)
    "danger": (0, 165, 255),    # Orange (BGR)
    "default": (0, 230, 118),   # Green (BGR)
}


def classify_sign_category(label: str) -> str:
    """Categorizes sign into stop, speed, or danger."""
    lbl = label.lower()
    if "stop" in lbl:
        return "stop"
    elif "speed" in lbl or "limit" in lbl:
        return "speed"
    elif any(k in lbl for k in ["danger", "caution", "warn", "curve", "cross", "work", "bump", "slip"]):
        return "danger"
    return "default"


def detect_traffic_signs(image_bgr: np.ndarray, conf_thresh: float = 0.40):
    """Runs YOLO detection and draws simple bounding boxes with labels."""
    results = yolo_model.predict(image_bgr, conf=conf_thresh, verbose=False)[0]
    annotated = image_bgr.copy()
    h, w = annotated.shape[:2]
    detections = []

    for box in results.boxes:
        cls_id = int(box.cls[0].item())
        label = yolo_model.names.get(cls_id, f"Sign_{cls_id}")
        conf = float(box.conf[0].item())
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

        cat = classify_sign_category(label)
        color = SIGN_COLORS.get(cat, SIGN_COLORS["default"])

        # Draw bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

        # Draw clean text badge
        caption = f"{label.upper()} {conf*100:.1f}%"
        (tw, th), _ = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        by1 = max(0, y1 - th - 6)
        cv2.rectangle(annotated, (x1, by1), (x1 + tw + 8, y1), color, -1)
        text_color = (0, 0, 0) if cat in ["speed", "danger"] else (255, 255, 255)
        cv2.putText(annotated, caption, (x1 + 4, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)

        detections.append({
            "Label": label.title(),
            "Category": cat.upper(),
            "Confidence": f"{conf*100:.1f}%",
            "Box": f"[{x1}, {y1}, {x2}, {y2}]",
            "raw_cat": cat
        })

    return annotated, detections


# ==========================================
# SIDEBAR NAVIGATION & SETTINGS
# ==========================================
with st.sidebar:
    st.markdown("### 🚦 Traffic Sign & Road Safety")
    st.markdown("---")

    nav_choice = st.radio(
        "Navigation",
        ["📸 Image Detection", "🎥 Video Detection", "ℹ️ About Project"],
        key="nav_radio"
    )

    st.markdown("---")
    st.markdown("#### ⚙️ Detection Settings")

    conf_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.20,
        max_value=0.90,
        value=0.40,
        step=0.05,
        help="Minimum confidence percentage required to display a detected sign.",
        key="conf_slider"
    )

    st.caption("Model: **YOLOv8 Nano (Traffic Signs)**")


# ==========================================
# PAGE 1: IMAGE DETECTION
# ==========================================
if nav_choice == "📸 Image Detection":
    st.title("📸 Traffic Sign Recognition")
    st.markdown("Upload a road photo or pick a sample image to detect traffic signs and view safety alerts.")

    col1, col2 = st.columns([2, 1])

    with col1:
        uploaded_file = st.file_uploader(
            "Upload Road Scene Image",
            type=["jpg", "jpeg", "png", "bmp"],
            key="img_upload"
        )

    with col2:
        sample_dir = ROOT_DIR / "data" / "images" / "test"
        sample_files = sorted([f.name for f in sample_dir.glob("*.jpg")]) if sample_dir.exists() else []
        sample_options = ["None (Upload file)"] + sample_files[:10]
        chosen_sample = st.selectbox("Or Choose Sample Image:", sample_options, index=1 if sample_files else 0, key="sample_sel")

    # Load selected image
    input_bgr = None
    img_name = ""

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.getvalue()), dtype=np.uint8)
        input_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        img_name = uploaded_file.name
    elif chosen_sample != "None (Upload file)" and sample_files:
        sample_path = sample_dir / chosen_sample
        if sample_path.exists():
            input_bgr = cv2.imread(str(sample_path))
            img_name = chosen_sample

    if input_bgr is not None:
        # Run detection
        t0 = time.time()
        annotated_bgr, detections = detect_traffic_signs(input_bgr, conf_thresh=conf_threshold)
        latency_ms = (time.time() - t0) * 1000

        total_count = len(detections)
        stop_count = sum(1 for d in detections if d["raw_cat"] == "stop")
        speed_count = sum(1 for d in detections if d["raw_cat"] == "speed")
        danger_count = sum(1 for d in detections if d["raw_cat"] == "danger")

        # Metric summary cards
        st.markdown("### 📊 Detection Summary")
        m1, m2, m3, m4, m5 = st.columns(5)
        with m1:
            st.markdown(f'<div class="metric-box" style="border-left-color:#00D2FF;"><div class="metric-value">{total_count}</div><div class="metric-label">Total Signs</div></div>', unsafe_allow_html=True)
        with m2:
            st.markdown(f'<div class="metric-box" style="border-left-color:#FF2A55;"><div class="metric-value">{stop_count}</div><div class="metric-label">Stop Signs</div></div>', unsafe_allow_html=True)
        with m3:
            st.markdown(f'<div class="metric-box" style="border-left-color:#00D2FF;"><div class="metric-value">{speed_count}</div><div class="metric-label">Speed Limit</div></div>', unsafe_allow_html=True)
        with m4:
            st.markdown(f'<div class="metric-box" style="border-left-color:#FFA500;"><div class="metric-value">{danger_count}</div><div class="metric-label">Hazard / Caution</div></div>', unsafe_allow_html=True)
        with m5:
            st.markdown(f'<div class="metric-box" style="border-left-color:#00E676;"><div class="metric-value">{latency_ms:.1f} ms</div><div class="metric-label">Inference Time</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Simple road safety alerts
        st.markdown("### 🚨 Road Safety Alerts")
        if total_count == 0:
            st.markdown("""
            <div class="alert-banner alert-clear">
                <span style="font-size: 1.5rem;">✅</span>
                <div>
                    <strong>Road Clear:</strong> No traffic signs detected in current view.
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            for d in detections:
                cat = d["raw_cat"]
                lbl = d["Label"]
                conf = d["Confidence"]
                if cat == "stop":
                    st.markdown(f"""
                    <div class="alert-banner alert-stop">
                        <span style="font-size: 1.5rem;">🛑</span>
                        <div><strong>STOP ALERT ({conf}):</strong> Stop sign detected ahead. Bring vehicle to a complete halt before the stop line.</div>
                    </div>
                    """, unsafe_allow_html=True)
                elif cat == "speed":
                    st.markdown(f"""
                    <div class="alert-banner alert-speed">
                        <span style="font-size: 1.5rem;">⚡</span>
                        <div><strong>SPEED LIMIT ALERT ({conf}):</strong> {lbl} detected. Maintain legal driving speed.</div>
                    </div>
                    """, unsafe_allow_html=True)
                elif cat == "danger":
                    st.markdown(f"""
                    <div class="alert-banner alert-danger">
                        <span style="font-size: 1.5rem;">⚠️</span>
                        <div><strong>HAZARD WARNING ({conf}):</strong> {lbl} detected ahead. Drive with caution and maintain safe following distance.</div>
                    </div>
                    """, unsafe_allow_html=True)

        # Side-by-side display
        st.markdown("### 🖼️ Detection View")
        c_orig, c_pred = st.columns(2)
        with c_orig:
            st.markdown(f"**Original Image ({img_name})**")
            st.image(cv2.cvtColor(input_bgr, cv2.COLOR_BGR2RGB), width="stretch")
        with c_pred:
            st.markdown(f"**Detected Signs ({total_count} Found)**")
            st.image(cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB), width="stretch")

        # Download button
        ok, buf = cv2.imencode(".jpg", annotated_bgr)
        if ok:
            st.download_button("📥 Download Annotated Image", buf.tobytes(), "annotated_signs.jpg", "image/jpeg", key="dl_btn")

        # Table of detected signs
        if detections:
            st.markdown("### 📋 Detected Signs List")
            df = pd.DataFrame([{
                "Sign Name": d["Label"],
                "Category": d["Category"],
                "Confidence": d["Confidence"],
                "Bounding Box [x1, y1, x2, y2]": d["Box"]
            } for d in detections])
            st.dataframe(df, width="stretch", hide_index=True)

    else:
        st.info("👈 Please upload an image or choose a sample image from above to begin.")


# ==========================================
# PAGE 2: VIDEO DETECTION
# ==========================================
elif nav_choice == "🎥 Video Detection":
    st.title("🎥 Video Traffic Sign Detection")
    st.markdown("Process video files or test clips frame-by-frame with real-time bounding boxes.")

    col1, col2 = st.columns([2, 1])
    with col1:
        video_file = st.file_uploader("Upload Dashcam Video (.mp4, .avi)", type=["mp4", "avi"], key="vid_upload")
    with col2:
        sample_vid = ROOT_DIR / "data" / "sample_dashcam.mp4"
        use_sample = st.checkbox("Use Sample Video Clip", value=(video_file is None and sample_vid.exists()), key="vid_sample_chk")

    video_path = None
    if video_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(video_file.getvalue())
        video_path = tfile.name
    elif use_sample and sample_vid.exists():
        video_path = str(sample_vid)

    if video_path and os.path.exists(video_path):
        cap = cv2.VideoCapture(video_path)
        total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        st.markdown(f"**Video Loaded:** {total_f} frames available.")

        if st.button("🚀 Run Video Detection", type="primary", key="btn_run_vid"):
            progress_bar = st.progress(0)
            frame_view = st.empty()
            status = st.empty()

            frame_idx = 0
            max_frames = min(total_f, 80)
            detections_found = 0

            while cap.isOpened() and frame_idx < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % 2 == 0:
                    ann_frame, dets = detect_traffic_signs(frame, conf_thresh=conf_threshold)
                    detections_found += len(dets)
                    frame_view.image(cv2.cvtColor(ann_frame, cv2.COLOR_BGR2RGB), width="stretch")
                    progress_bar.progress(min(1.0, (frame_idx + 1) / max_frames))

                frame_idx += 1

            cap.release()
            status.success(f"Video analysis complete! Processed {frame_idx} frames. Total sign encounters: {detections_found}")
    else:
        st.info("👈 Upload a video or check 'Use Sample Video Clip' to start.")


# ==========================================
# PAGE 3: ABOUT PROJECT (VIVA READY)
# ==========================================
elif nav_choice == "ℹ️ About Project":
    st.title("ℹ️ Project Overview")
    st.markdown("""
    ### 🚦 Traffic Sign Recognition & Road Safety Monitor
    
    #### 1. Aim of the Project
    To build a deep learning system that detects and recognizes traffic signs in real time from road images 
    and dashcam video streams, and provides basic safety alerts (Stop, Speed Limit, Hazard) to assist the driver.

    #### 2. Key Components
    - **Vision Model:** YOLOv8 Nano (trained on traffic sign datasets).
    - **Classes Detected:** Stop Signs, Speed Limit Signs, and Danger/Hazard Warning Signs.
    - **Frontend Interface:** Streamlit dashboard with real-time bounding box annotations and confidence scores.
    - **Safety Advisories:** Color-coded driver alerts based on detected sign categories.

    #### 3. Technology Stack
    - **Language:** Python 3
    - **Deep Learning:** PyTorch & Ultralytics YOLOv8
    - **Computer Vision:** OpenCV
    - **User Interface:** Streamlit
    """)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #64748B; font-size: 0.85rem;'>"
    "Traffic Sign Recognition and Road Safety Monitor | PBL Project"
    "</div>",
    unsafe_allow_html=True
)
