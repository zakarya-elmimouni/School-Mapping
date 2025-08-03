from pathlib import Path
import shutil

# Source directories
new_img_dir = Path("dataset/brazil/manual_labeled_data/images/train")
new_lbl_dir = Path("dataset/brazil/manual_labeled_data/labels/train")

# Target (train) directories
target_train_img = Path("dataset/brazil/dataset_yolo_auto_label/images/train")
target_train_lbl = Path("dataset/brazil/dataset_yolo_auto_label/labels/train")

# Directories to check for duplicates
val_dir = Path("dataset/brazil/manual_labeled_data/images/val")
test_dir = Path("dataset/brazil/manual_labeled_data/images/test")

# Collect image names (without extension) from val and test
existing_names = {
    p.stem for p in val_dir.glob("*.*")
}.union({
    p.stem for p in test_dir.glob("*.*")
})

added = 0
skipped = 0

# Loop through new images
for img_path in new_img_dir.glob("*.*"):
    if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
        continue
    name = img_path.stem
    if name in existing_names:
        print(f"Skipping {img_path.name} (already in val/test)")
        skipped += 1
        continue

    label_path = new_lbl_dir / f"{name}.txt"
    if not label_path.exists():
        print(f"??  Warning: no label found for {img_path.name}, skipping.")
        continue

    # Copy image and label to training set
    shutil.copy2(img_path, target_train_img / img_path.name)
    shutil.copy2(label_path, target_train_lbl / label_path.name)
    print(f" Added {img_path.name}")
    added += 1

print(f"\n Done: {added} images added, {skipped} skipped.")