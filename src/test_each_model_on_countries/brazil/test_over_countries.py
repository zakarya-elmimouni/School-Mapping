import os
import random
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict
from ultralytics import YOLO


"""This scripts is used to evaluate the model trained on brazil over the other countries"""

# =====================
# CONFIGURATION
# =====================
MODEL_PATH = "path to the brazil model that you want to evaluate over the other countries"
COUNTRIES = ["peru", "togo", "nigeria", "mali", "colombia", "bangladesh", "lesotho"]
NUM_IMAGES = 10

# === Existing evaluation helper functions ===
def load_yolo_labels(label_path, img_shape):
    h, w = img_shape[:2]
    boxes = []
    if not os.path.exists(label_path):
        return boxes
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                _, x, y, bw, bh = map(float, parts)
                cx, cy = x * w, y * h
                bw, bh = bw * w, bh * h
                x1, y1 = int(cx - bw/2), int(cy - bh/2)
                x2, y2 = int(cx + bw/2), int(cy + bh/2)
                boxes.append((x1, y1, x2, y2))
    return boxes

def compute_iou(box1, box2):
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])
    inter_area = max(0, xB - xA) * max(0, yB - yA)
    if inter_area == 0:
        return 0.0
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return inter_area / float(box1_area + box2_area - inter_area)

def average_precision(recalls, precisions):
    precisions = np.maximum.accumulate(precisions[::-1])[::-1]
    return np.trapz(precisions, recalls)

def compute_map_iou(model, img_dir, label_dir, iou_thresh=0.5):
    all_detections = []
    all_annotations = defaultdict(list)
    img_paths = list(Path(img_dir).glob("*.png"))

    for img_path in img_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img_name = img_path.stem
        label_path = os.path.join(label_dir, img_name + ".txt")
        gt_boxes = load_yolo_labels(label_path, img.shape)
        results = model(img, max_det=1)
        pred_boxes = results[0].boxes.xyxy.cpu().numpy()
        scores = results[0].boxes.conf.cpu().numpy()
        all_annotations[img_name] = gt_boxes
        for i in range(len(pred_boxes)):
            all_detections.append((img_name, scores[i], pred_boxes[i]))

    all_detections.sort(key=lambda x: x[1], reverse=True)
    image_gt_flags = {k: np.zeros(len(v)) for k, v in all_annotations.items()}
    tp = np.zeros(len(all_detections))
    fp = np.zeros(len(all_detections))
    total_gts = sum(len(v) for v in all_annotations.values())

    for i, (img_name, conf, pred_box) in enumerate(all_detections):
        matched = False
        gt_boxes = all_annotations[img_name]
        for j, gt_box in enumerate(gt_boxes):
            iou = compute_iou(pred_box, gt_box)
            if iou >= iou_thresh and image_gt_flags[img_name][j] == 0:
                matched = True
                image_gt_flags[img_name][j] = 1
                break
        tp[i] = 1 if matched else 0
        fp[i] = 0 if matched else 1

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recalls = tp_cum / total_gts if total_gts else np.array([])
    precisions = tp_cum / (tp_cum + fp_cum + 1e-16)
    ap = average_precision(recalls, precisions)
    final_recall = recalls[-1] if len(recalls) else 0
    final_precision = precisions[-1] if len(precisions) else 0
    return ap, final_recall, final_precision

def compute_mse_pred_vs_gt(img_dir, label_dir, model):
    distances = []
    img_paths = list(Path(img_dir).glob("*.png"))
    for img_path in img_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img_name = img_path.stem
        label_path = os.path.join(label_dir, img_name + ".txt")
        gt_boxes = load_yolo_labels(label_path, img.shape)
        if not gt_boxes:
            continue
        results = model(img, max_det=1)
        pred_boxes = results[0].boxes.xyxy.cpu().numpy()
        if len(pred_boxes) == 0:
            continue
        x1_p, y1_p, x2_p, y2_p = pred_boxes[0]
        x1_g, y1_g, x2_g, y2_g = gt_boxes[0]
        cx_pred = (x1_p + x2_p) / 2
        cy_pred = (y1_p + y2_p) / 2
        cx_gt = (x1_g + x2_g) / 2
        cy_gt = (y1_g + y2_g) / 2
        distances.append((cx_pred - cx_gt) ** 2 + (cy_pred - cy_gt) ** 2)
    return np.mean(distances) if distances else None

# =====================
# EVALUATION LOOP
# =====================
model = YOLO(MODEL_PATH)

for country in COUNTRIES:
    print(f"\n=== Evaluating on {country.upper()} ===")

    data_yaml = f"dataset/{country}/manual_labeled_data/data.yaml"
    images_test_dir = f"dataset/{country}/manual_labeled_data/images/test"
    labels_test_dir = f"dataset/{country}/manual_labeled_data/labels/test"
    output_dir = f"results/brazil/rslt_yolo10n_finetuning_auto_on_golden_best_params/best/outputs_over_{country}" # change the saving directory as needed
    os.makedirs(os.path.join(output_dir, "yolo_predictions"), exist_ok=True)
    output_metrics = os.path.join(output_dir, "evaluation_metrics.txt")

    metrics = model.val(split="test", data=data_yaml, max_det=1)

    with open(output_metrics, "w") as f:
        f.write("=== YOLOv10 trained on Brazil and evaluated on golden test set ===\n")
        f.write(f"mAP@0.5        : {metrics.box.map50:.4f}\n")
        f.write(f"mAP@0.5:0.95   : {metrics.box.map:.4f}\n")
        f.write(f"Mean Precision : {metrics.box.mp:.4f}\n")
        f.write(f"Mean Recall    : {metrics.box.mr:.4f}\n")
        if (metrics.box.mp + metrics.box.mr) > 0:
            f1 = 2 * metrics.box.mp * metrics.box.mr / (metrics.box.mp + metrics.box.mr)
        else:
            f1 = 0.0
        f.write(f"Mean F1-score  : {f1:.4f}\n")

    # Custom evaluations
    map30, r30, p30 = compute_map_iou(model, images_test_dir, labels_test_dir, iou_thresh=0.3)
    map75, r75, p75 = compute_map_iou(model, images_test_dir, labels_test_dir, iou_thresh=0.75)
    iou_thresholds = np.arange(0.5, 1.0, 0.05)
    all_ap = [compute_map_iou(model, images_test_dir, labels_test_dir, iou)[0] for iou in iou_thresholds]
    map_50_95 = np.mean(all_ap)
    mse_center = compute_mse_pred_vs_gt(images_test_dir, labels_test_dir, model)

    with open(output_metrics, "a") as f:
        f.write("\n=== Full mAP@0.3 Evaluation ===\n")
        f.write(f"mAP@0.3       : {map30:.4f}\n")
        f.write(f"Precision@0.3 : {p30:.4f}\n")
        f.write(f"Recall@0.3    : {r30:.4f}\n")
        f.write("\n=== Full mAP@0.75 Evaluation ===\n")
        f.write(f"mAP@0.75      : {map75:.4f}\n")
        f.write(f"Precision@0.75: {p75:.4f}\n")
        f.write(f"Recall@0.75   : {r75:.4f}\n")
        f.write("\n=== Custom mAP@[0.5:0.95] Evaluation ===\n")
        f.write(f"mAP@[.5:.95]  : {map_50_95:.4f}\n")
        f.write("\n=== Center Distance MSE (Prediction vs Ground Truth) ===\n")
        if mse_center is not None:
            f.write(f"MSE_center_distance: {mse_center:.2f}\n")
            f.write(f"RMSE_center_distance: {np.sqrt(mse_center):.2f}\n")
        else:
            f.write("MSE_center_distance: N/A\n")

    print(f"Done. Metrics saved to {output_metrics}")
