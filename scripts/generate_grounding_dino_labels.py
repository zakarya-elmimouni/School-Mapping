import os
import json
from glob import glob
from tqdm import tqdm

def convert_yolo_to_coco(yolo_labels_dir, images_dir, output_json_path, image_size=(500, 500), class_name="school"):
    image_files = sorted(glob(os.path.join(images_dir, "*.jpg")) + glob(os.path.join(images_dir, "*.png")))
    
    coco_dict = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": class_name}],
    }

    annotation_id = 1
    for image_id, image_path in enumerate(tqdm(image_files, desc=f"Processing {output_json_path}")):
        file_name = os.path.basename(image_path)
        image_info = {
            "id": image_id,
            "file_name": file_name,
            "width": image_size[0],
            "height": image_size[1],
        }
        coco_dict["images"].append(image_info)

        # Corresponding label file
        label_file = os.path.join(yolo_labels_dir, os.path.splitext(file_name)[0] + ".txt")
        if not os.path.exists(label_file):
            continue  # image sans bbox ? pas d'annotation

        with open(label_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue  # skip malformed

                class_id, x_center, y_center, width, height = map(float, parts)
                if class_id != 0:
                    continue  # skip autres classes (juste au cas)

                # convert normalized to absolute
                x_abs = (x_center - width / 2) * image_size[0]
                y_abs = (y_center - height / 2) * image_size[1]
                w_abs = width * image_size[0]
                h_abs = height * image_size[1]

                annotation = {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": 1,
                    "bbox": [x_abs, y_abs, w_abs, h_abs],
                    "area": w_abs * h_abs,
                    "iscrowd": 0,
                }
                coco_dict["annotations"].append(annotation)
                annotation_id += 1

    # Write output
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, "w") as f:
        json.dump(coco_dict, f, indent=4)
    print(f"Saved: {output_json_path}")


# === Apply to train, val, test ===
splits = ["train", "val", "test"]
for split in splits:
    label_dir = f"dataset/brazil/dataset_yolo_auto_labeling/labels/{split}"
    image_dir = f"dataset/brazil/dataset_yolo_auto_labeling/images/{split}"
    output_json = f"dataset/brazil/grounding_dino_labels/{split}.json"
    convert_yolo_to_coco(label_dir, image_dir, output_json)
