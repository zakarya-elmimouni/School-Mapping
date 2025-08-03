import os
import cv2
import random

# === CONFIGURATION ===
image_dir = "dataset/brazil/dataset_yolo_auto_labeling/images/train"
label_dir = "dataset/brazil/dataset_faster_rcnn/labels/train"
output_dir = "dataset/brazil/dataset_faster_rcnn/get_viz"
num_images = 10  # nombre d'images  visualiser
box_color = (0, 255, 0)  # green
box_thickness = 2

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Get image files that have corresponding label files
image_files = [f for f in os.listdir(image_dir) if f.endswith(".jpg") or f.endswith(".png")]
image_files = [f for f in image_files if os.path.exists(os.path.join(label_dir, f.replace('.jpg', '.txt').replace('.png', '.txt')))]

# Select random samples
sample_images = random.sample(image_files, min(num_images, len(image_files)))

for img_name in sample_images:
    img_path = os.path.join(image_dir, img_name)
    label_path = os.path.join(label_dir, img_name.replace('.jpg', '.txt').replace('.png', '.txt'))

    # Load image
    img = cv2.imread(img_path)

    # Load and draw labels
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                xmin, ymin, xmax, ymax, cls_id = map(int, parts)
                cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color=box_color, thickness=box_thickness)
                cv2.putText(img, str(cls_id), (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)

    # Save the image with bounding boxes
    save_path = os.path.join(output_dir, img_name)
    cv2.imwrite(save_path, img)

print(f"? Saved {len(sample_images)} visualized images to: {output_dir}")

