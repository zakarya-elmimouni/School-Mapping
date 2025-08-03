from pathlib import Path, PurePath
import random

IMG_DIR = Path("dataset/brazil/dataset_yolo_auto_labeling/images/train")
TRAIN_TXT = Path("dataset/brazil/dataset_yolo_auto_labeling/train.txt")
MANUAL_TAG = "manual_labeled"   # motif qui identifie les images manuelles
K = 4                                 # facteur de surchantillonnage

lines = []
for img in IMG_DIR.iterdir():
    if img.suffix.lower() not in {".jpg", ".jpeg", ".png"} or not img.is_file():
        continue
    line = str(img.resolve())        # YOLO accepte aussi les chemins relatifs
    if MANUAL_TAG in line:
        lines.extend([line]*K)       # rpeteter K fois
    else:
        lines.append(line)

random.shuffle(lines)                 # bon melange pour les batchs
TRAIN_TXT.write_text("\n".join(lines) + "\n")
print(f"train.txt written with {len(lines)} lines")