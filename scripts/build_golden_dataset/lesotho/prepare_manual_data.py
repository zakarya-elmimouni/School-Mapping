#!/usr/bin/env python3
import shutil
import random
from pathlib import Path

# ---------- CONFIGURATION ---------------------------------------------------
LABEL_SRC_DIR     = Path("dataset/lesotho/manual_labeled_images")                       # existing .txt labels
POS_IMAGE_SRC_DIR = Path("data/lesotho/satellite/school")         # positive images
NEG_IMAGE_SRC_DIR = Path("dataset/lesotho/negative_lesotho")     # negative images

DEST_BASE_DIR     = Path("dataset/lesotho/manual_labeled_data")
LABEL_DEST_DIR    = DEST_BASE_DIR / "labels"
IMAGE_DEST_DIR    = DEST_BASE_DIR / "images"

IMG_EXTS = {".png"}    # accepted image types
NEGATIVE_LIMIT = 130
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

def copy_negative_images(limit: int = 100) -> int:
    #List and filter valid image files
    neg_images = [p for p in NEG_IMAGE_SRC_DIR.iterdir() if p.suffix.lower() in IMG_EXTS and p.is_file()]
    
    # Randomly sample
    sampled = random.sample(neg_images, min(len(neg_images), limit))

    for img_path in sampled:
        shutil.copy2(img_path, IMAGE_DEST_DIR / img_path.name)
        empty_label = LABEL_DEST_DIR / f"{img_path.stem}.txt"
        empty_label.touch(exist_ok=True)

    return len(sampled)

def main() -> None:
    ensure_dirs()

    pos_labels, pos_images = copy_positive_pairs()
    neg_images = copy_negative_images(limit=NEGATIVE_LIMIT)

    print(
        f"Finished.\n"
        f"  Positive labels copied : {pos_labels}\n"
        f"  Positive images copied : {pos_images}\n"
        f"  Negative images added  : {neg_images} (max = {NEGATIVE_LIMIT})\n"
        f"Destination folders: {IMAGE_DEST_DIR} & {LABEL_DEST_DIR}"
    )

if __name__ == "__main__":
    main()
