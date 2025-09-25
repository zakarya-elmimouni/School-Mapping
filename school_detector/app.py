import io
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
import streamlit as st
from ultralytics import YOLO
from typing import Optional

# ------------------------------
# Config Streamlit
# ------------------------------
st.set_page_config(page_title="School Detection (YOLO)", layout="wide")

# ------------------------------
# Available models (update paths)
# ------------------------------
MODELS = {
    "togo":        "models/togo/best.pt",
    "peru":        "models/peru/best.pt",
    "brazil":      "models/brazil/best.pt",
    "mali":        "models/mali/best.pt",
    "bangladesh":  "models/bangladesh/best.pt",
    "colombia":    "models/colombia/best.pt",
    "lesotho":     "models/lesotho/best.pt",
    "nigeria":     "models/nigeria/best.pt",
    "big_global":  "models/big_global_model/best.pt",
    "africa_regional":  "models/africa_regional_model/best.pt"
    
}

# ------------------------------
# Cached loaders
# ------------------------------
@st.cache_resource(show_spinner=False)
def load_model(model_path: str) -> YOLO:
    return YOLO(model_path)

@st.cache_data(show_spinner=False)
def resize_to_500(img_rgb: np.ndarray) -> np.ndarray:
    return cv2.resize(img_rgb, (500, 500), interpolation=cv2.INTER_AREA)

# ------------------------------
# Utils
# ------------------------------
def pil_to_rgb_array(pil_img: Image.Image) -> np.ndarray:
    return np.array(pil_img.convert("RGB"))

def draw_boxes_on_rgb(img_rgb: np.ndarray,
                      boxes_xyxy: np.ndarray,
                      confs: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Draw red rectangles on an RGB image using OpenCV (convert to BGR then back to RGB).
    """
    annotated_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i, (x1, y1, x2, y2) in enumerate(boxes_xyxy):
        cv2.rectangle(annotated_bgr, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 3)  # red in BGR
        if confs is not None:
            txt = f"school {float(confs[i]):.2f}"
            cv2.putText(annotated_bgr, txt, (int(x1), max(0, int(y1) - 8)), font, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
    return cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

def to_png_bytes(img_rgb: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(img_rgb).save(buf, format="PNG")
    return buf.getvalue()

def run_detection(img_rgb: np.ndarray, model: YOLO, conf: float = 0.25, max_det: int = 1):
    """
    Returns (annotated_rgb, has_detection, boxes, confs).
    - Resizes to 500x500 before inference as requested.
    """
    resized = resize_to_500(img_rgb)
    results = model(resized, conf=conf, max_det=max_det)

    boxes = results[0].boxes.xyxy.cpu().numpy() if results and results[0].boxes is not None else np.empty((0, 4))
    confs = results[0].boxes.conf.cpu().numpy() if results and results[0].boxes is not None else np.array([])

    if len(boxes) > 0:
        annotated = draw_boxes_on_rgb(resized, boxes, confs)
        return annotated, True, boxes, confs
    else:
        # No detection: return the resized original
        return resized, False, boxes, confs

# ------------------------------
# UI
# ------------------------------
st.title("🏫 School Detection on Satellite Images (YOLO)")

with st.sidebar:
    st.header("Settings")
    model_key = st.selectbox("Model", list(MODELS.keys()), index=0)
    conf_th = st.slider("Confidence threshold", 0.05, 0.95, 0.25, 0.01)
    max_det = st.number_input("Max detections", min_value=1, max_value=5, value=1, step=1)
    st.caption("All images are resized to **500×500** before inference.")

# Uploader
uploaded = st.file_uploader("Upload a satellite image (.png/.jpg)", type=["png", "jpg", "jpeg"])

col1, col2 = st.columns(2)
with col1:
    st.subheader("Input")
    if uploaded:
        st.image(uploaded, caption="Original image", use_container_width=True)

run = st.button("🚀 Run detection", disabled=uploaded is None)

with col2:
    st.subheader("Result")

    if run and uploaded:
        # Read image as RGB array
        pil_img = Image.open(uploaded)
        img_rgb = pil_to_rgb_array(pil_img)

        # Load model (cached)
        model_path = MODELS[model_key]
        if not Path(model_path).exists():
            st.error(f"Model weights not found: `{model_path}`")
        else:
            model = load_model(model_path)

            with st.spinner("Running YOLO inference..."):
                annotated_rgb, has_det, boxes, confs = run_detection(img_rgb, model, conf=conf_th, max_det=max_det)

            # Show result
            st.image(annotated_rgb, caption=("Detected" if has_det else "No detection"), use_container_width=True)

            # Small summary
            if has_det:
                st.success(f"Detected {len(boxes)} box(es). Showing up to {max_det}.")
            else:
                st.info("No school detected. Returned the resized original image.")

            # Download button
            st.download_button(
                label="⬇️ Download annotated image (PNG)",
                data=to_png_bytes(annotated_rgb),
                file_name="result_with_bbox.png",
                mime="image/png",
            )
    elif not uploaded:
        st.info("Upload an image to enable detection.")
