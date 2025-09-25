import os
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from pathlib import Path
from collections import defaultdict




MODEL_PATH = "path/to/weights/best.pt"
DATA_YAML = "path/to/data.yaml"
IMAGES_TEST_DIR = "path/to/images/test"
LABELS_TEST_DIR = "path/to/labels/test"
OUTPUT_METRICS_TXT = "path/to/output/metrics.txt"
OUTPUT_IMG_DIR = "path/to/output/visualizations"



NUM_IMAGES = 10
IMAGE_SIZE = 500

os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)

model = YOLO(MODEL_PATH)
metrics = model.val(split="test", data=DATA_YAML, max_det=1)

# === Save YOLO metrics ===
with open(OUTPUT_METRICS_TXT, "w") as f:
    f.write("=== YOLOv10 trained on yolo10 and Evaluated on golden Test Set ===\n")
    f.write(f"mAP@0.5        : {metrics.box.map50:.4f}\n")
    f.write(f"mAP@0.5:0.95   : {metrics.box.map:.4f}\n")
    f.write(f"Mean Precision : {metrics.box.mp:.4f}\n")
    f.write(f"Mean Recall    : {metrics.box.mr:.4f}\n\n")
    
    # Mean F1-score
    if (metrics.box.mp + metrics.box.mr) > 0:
        mean_f1 = 2 * metrics.box.mp * metrics.box.mr / (metrics.box.mp + metrics.box.mr)
    else:
        mean_f1 = 0.0
    f.write(f"Mean F1-score  : {mean_f1:.4f}\n\n")

    f.write("=== Per-class Metrics ===\n")
    for i, name in model.names.items():
        p, r, ap50, ap = metrics.box.class_result(i)
        if (p + r) > 0:
            f1 = 2 * p * r / (p + r)
        else:
            f1 = 0.0
        f.write(f"{name:10s}: Precision={p:.3f}, Recall={r:.3f}, F1-score={f1:.3f}, mAP50={ap50:.3f}, mAP50-95={ap:.3f}\n")


print(f"Metrics saved to: {OUTPUT_METRICS_TXT}")





