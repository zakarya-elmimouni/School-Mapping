# School Detection from Satellite Imagery

This repository contains the codebase for the weakly supervised object detection pipeline used to detect school buildings from satellite images, in the context of digital connectivity and infrastructure mapping. The approach relies on a large dataset of auto-labeled images and a smaller manually annotated set (golden data), with segmentation-based labeling, training, and fine-tuning strategies.

---

##  Project Structure & Key Features

- Automatic data download from Google Static Maps API (requires API key)
- Outlier cleaning (blur, vegetation, sea, desert)
- Bounding box generation using LangSAM segmentation
- YOLO-compatible dataset construction and augmentation
- Fine-tuning with high-quality golden data
- Training, evaluation, and hyperparameter tuning (ECP)

---

##  How to Run the Code

### 1. Install dependencies

run:

```bash
pip install -r requirements.txt
```
### 2. Download satellite images (due to Google Maps API restrictions)

⚠️ **We do not share image files directly.**  
Google's usage policy does not allow redistribution of Static Map images.  
To reproduce the dataset:

- Get a [Google Static Maps API key](https://developers.google.com/maps/documentation/maps-static/get-api-key)
- For each country run the code provided in scripts/download_data_from_static_maps_api.

### 2. Clean and generate automatic labels (bounding boxes) 
- Clean outlier images (blurred, vegetation, sea, desert) and generate the bounding boxes with segementation using the codes provided in the folder scripts/cleaning_scripts
### 3. Prepare YOLO dataset (Auto-labeled)
 - Build the YOLO-compatible dataset with auto-labeled bounding boxes and apply standard augmentations
### 4. Prepare Golden Dataset
-The golden dataset consists of manually annotated labels in YOLO format.
- You will find label files in:

 ```bash
dataset/{country}/manaully_labeled_data/labels
```
To use this dataset:
- Copy only the matching images from data/{country}/satellite/ (same filenames as labels).
- Build the golden dataset for each country
### 7. Train and Evaluate YOLO models
Once the datasets are ready you can lunch the training and evaluation.
### 8. Hyperparameters (ECP)

We optimized the training hyperparameters using the [ECP algorithm](https://arxiv.org/abs/2502.04290), a black-box optimization method well-suited for tuning costly deep learning models.

All optimal hyperparameter for the Finetuned models with ECP, are stored in the folder: hyperparamaters.


