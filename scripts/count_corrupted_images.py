from PIL import Image, UnidentifiedImageError
import os

# === Configuration ===
image_root = "ibex/user/elmimoz/dataset/bangaldesh/dataset_yolo_auto_labeling/images"
subfolders = ["train", "val", "test"]  

total_images = 0
total_corrupted = 0

for sub in subfolders:
    folder_path = os.path.join(image_root, sub)
    corrupted_in_folder = 0

    if not os.path.exists(folder_path):
        print(f"? Folder not found: {folder_path}")
        continue

    files = [f for f in os.listdir(folder_path) if f.endswith(".png") or f.endswith(".jpg")]
    total_images += len(files)

    for fname in files:
        fpath = os.path.join(folder_path, fname)
        try:
            with Image.open(fpath) as img:
                img.verify()
        except (UnidentifiedImageError, OSError):
            print(f"? Corrupted: {fpath}")
            corrupted_in_folder += 1

    print(f"?? {sub}: {corrupted_in_folder} corrupted images out of {len(files)}")
    total_corrupted += corrupted_in_folder

print("==========")
print(f"?? Total images checked: {total_images}")
print(f"?? Total corrupted images: {total_corrupted}")
