import os
import cv2
import random
import shutil

# -------------------- CONFIG --------------------
images_dir = "dataset/brazil/manual_labeled_data/images/test"
labels_dir = "dataset/brazil/manual_labeled_data/labels/test"
output_dir = "dataset/brazil/manual_labeled_data/get_viz"

num_images = 30 # CHANGE this number as you wish

#images_dir = "test/dataset_test/images"
#labels_dir = "test/dataset_test/labels"
#output_dir = "test/dataset_test/get_viz"

os.makedirs(output_dir, exist_ok=True)

# -------------------- SELECT RANDOM IMAGES --------------------
imgs = [f for f in os.listdir(images_dir) if f.lower().endswith((".jpg",".jpeg",".png"))]
if num_images > len(imgs):
    raise ValueError(f"Requested {num_images} images, but only {len(imgs)} available.")

selected_imgs = random.sample(imgs, num_images)

for idx, fname in enumerate(selected_imgs, 1):
    img_path = os.path.join(images_dir, fname)
    label_path = os.path.join(labels_dir, os.path.splitext(fname)[0] + ".txt")

    img = cv2.imread(img_path)
    if img is None:
        print(f"Skipped {fname}: cannot read image.")
        continue

    H, W = img.shape[:2]

    if not os.path.exists(label_path):
        print(f"No label found for {fname}, skipping.")
        continue

    with open(label_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            continue  # skip malformed line

        _, cx, cy, w, h = map(float, parts)
        x_center, y_center = cx * W, cy * H
        bbox_w, bbox_h = w * W, h * H
        x1, y1 = int(x_center - bbox_w / 2), int(y_center - bbox_h / 2)
        x2, y2 = int(x_center + bbox_w / 2), int(y_center + bbox_h / 2)

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    output_path = os.path.join(output_dir, fname)
    cv2.imwrite(output_path, img)
    print(f"[{idx}/{num_images}] Saved image with bbox: {output_path}")

print(f"\nDone! Saved {num_images} images with bounding boxes in '{output_dir}'.")
