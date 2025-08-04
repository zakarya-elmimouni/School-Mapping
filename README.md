# 🛰️ School Detection from Satellite Imagery

This repository contains the codebase for the weakly supervised object detection pipeline used to detect school buildings from satellite images, in the context of digital connectivity and infrastructure mapping. The approach relies on a large dataset of auto-labeled images and a smaller manually annotated set (golden data), with segmentation-based labeling, training, and fine-tuning strategies.

---

## 📦 Project Structure & Key Features

- Automatic data download from Google Static Maps API (requires API key)
- Outlier cleaning (blur, vegetation, sea, desert)
- Bounding box generation using LangSAM segmentation
- YOLO-compatible dataset construction and augmentation
- Fine-tuning with high-quality golden data
- Training, evaluation, and hyperparameter tuning (ECP)

---

## ⚙️ How to Run the Code

### 1. Install dependencies

run:

```bash
pip install -r requirements.txt

### 2. Download satellite images (due to Google Maps API restrictions)
