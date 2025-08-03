import os

# === CONFIGURATION ===
input_base = "dataset/brazil/dataset_yolo_auto_labeling/labels"
output_base = "dataset/brazil/dataset_faster_rcnn/labels"
splits = ["train", "val", "test"]

# Cre les dossiers de sortie
for split in splits:
    os.makedirs(os.path.join(output_base, split), exist_ok=True)

def convert_yolo_line(line, img_width=500, img_height=500):
    parts = line.strip().split()
    if len(parts) != 5:
        return None
    cls_id = int(parts[0]) + 1  # Faster R-CNN expects classes starting at 1
    x_center, y_center, w, h = map(float, parts[1:])
    xmin = (x_center - w / 2) * img_width
    ymin = (y_center - h / 2) * img_height
    xmax = (x_center + w / 2) * img_width
    ymax = (y_center + h / 2) * img_height
    return f"{int(xmin)} {int(ymin)} {int(xmax)} {int(ymax)} {cls_id}"

for split in splits:
    input_dir = os.path.join(input_base, split)
    output_dir = os.path.join(output_base, split)
    label_files = [f for f in os.listdir(input_dir) if f.endswith(".txt")]

    for file in label_files:
        input_path = os.path.join(input_dir, file)
        output_path = os.path.join(output_dir, file)

        with open(input_path, "r") as fin:
            lines = fin.readlines()

        new_lines = []
        for line in lines:
            converted = convert_yolo_line(line)
            if converted:
                new_lines.append(converted)

        with open(output_path, "w") as fout:
            fout.write("\n".join(new_lines))
