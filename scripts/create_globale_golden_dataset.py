import os
import shutil
from glob import glob


""" this script creates a global golden dataset for YOLO from four countries' datasets."""

# === Configuration ===
countries = ["brazil", "peru", "colombia", "nigeria"]
splits = ["train", "val", "test"]
base_input_dir = "dataset"
base_output_dir = "dataset/global_golden_dataset"

# === Create destination folders ===
for split in splits:
    os.makedirs(os.path.join(base_output_dir, "images", split), exist_ok=True)
    os.makedirs(os.path.join(base_output_dir, "labels", split), exist_ok=True)

# === Copy function with optional renaming ===
def copy_files(image_paths, label_paths, country, split):
    for img_path in image_paths:
        img_name = os.path.basename(img_path)
        new_img_name = f"{country}_{img_name}"
        dest_img_path = os.path.join(base_output_dir, "images", split, new_img_name)
        shutil.copy(img_path, dest_img_path)

        # Copy label with same name (but .txt)
        label_name = os.path.splitext(img_name)[0] + ".txt"
        label_path = os.path.join(os.path.dirname(label_paths[0]), label_name)
        if os.path.exists(label_path):
            new_label_name = f"{country}_{label_name}"
            dest_label_path = os.path.join(base_output_dir, "labels", split, new_label_name)
            shutil.copy(label_path, dest_label_path)

# === Merge all datasets ===
for country in countries:
    for split in splits:
        image_dir = os.path.join(base_input_dir, country, "manual_labeled_data", "images", split)
        label_dir = os.path.join(base_input_dir, country, "manual_labeled_data", "labels", split)

        image_files = glob(os.path.join(image_dir, "*.jpg")) + glob(os.path.join(image_dir, "*.png"))
        label_files = glob(os.path.join(label_dir, "*.txt"))  # just to get label path directory

        if image_files and label_files:
            copy_files(image_files, label_files, country, split)

print("? Global YOLO dataset created successfully!")
