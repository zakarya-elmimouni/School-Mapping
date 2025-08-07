#!/usr/bin/env python3
import shutil
import random
from pathlib import Path

"""Prepare manual labeled data for Bangladesh school detection. labels are provided in .txt files in dataset/bangladesh/manual_labeled_data/labels"""

# ---------- CONFIGURATION ---------------------------------------------------
LABEL_SRC_DIR     = Path("path/to/labels")          # existing .txt labels
POS_IMAGE_SRC_DIR = Path("data/bangladesh/satellite/school")         # positive images
DEST_BASE_DIR     = Path("dataset/bangladesh/manual_labeled_data") # path to store the new dataset
LABEL_DEST_DIR    = DEST_BASE_DIR / "labels"
IMAGE_DEST_DIR    = DEST_BASE_DIR / "images"

IMG_EXTS = {".png"}    # accepted image types
NEGATIVE_LIMIT = 100
# ---------------------------------------------------------------------------

def ensure_dirs() -> None:
    LABEL_DEST_DIR.mkdir(parents=True, exist_ok=True)
    IMAGE_DEST_DIR.mkdir(parents=True, exist_ok=True)

def copy_positive_pairs() -> tuple[int, int]:
    n_labels, n_pos_imgs = 0, 0
    for label_path in LABEL_SRC_DIR.glob("*.txt"):
        stem = label_path.stem

        # Copy label
        shutil.copy2(label_path, LABEL_DEST_DIR / label_path.name)
        n_labels += 1

        # Find matching image
        for ext in IMG_EXTS:
            img_path = POS_IMAGE_SRC_DIR / f"{stem}{ext}"
            if img_path.exists():
                shutil.copy2(img_path, IMAGE_DEST_DIR / img_path.name)
                n_pos_imgs += 1
                break
        else:
            print(f"[WARNING] No positive image found for {stem}")
    return n_labels, n_pos_imgs



def main() -> None:
    ensure_dirs()

    pos_labels, pos_images = copy_positive_pairs()
    

    print(
        f"Finished.\n"
        f"  Positive labels copied : {pos_labels}\n"
        f"  Positive images copied : {pos_images}\n"
        #f"  Negative images added  : {neg_images} (max = {NEGATIVE_LIMIT})\n"
        f"Destination folders: {IMAGE_DEST_DIR} & {LABEL_DEST_DIR}"
    )

if __name__ == "__main__":
    main()
