from PIL import Image, UnidentifiedImageError
import os

# ==== CONFIGURATION ==========================================================
IMG_ROOT     = "dataset/brazil/dataset_yolo_auto_labeling/images"
YOLO_LABELS  = "dataset/brazil/dataset_yolo_auto_labeling/labels"
FRCNN_LABELS = "dataset/brazil/dataset_faster_rcnn/labels"          # only if you created these
DELETE_FRCNN_LABELS = True                  # set False if you want 
SUBFOLDERS   = ["train", "val", "test"]     # adjust to your splits
IMG_EXTS     = (".png", ".jpg", ".jpeg")     # accepted image extensions
# ============================================================================

total_checked   = 0
total_deleted   = 0
corrupted_files = []

for split in SUBFOLDERS:
    img_dir   = os.path.join(IMG_ROOT, split)
    yolo_dir  = os.path.join(YOLO_LABELS, split)
    frcnn_dir = os.path.join(FRCNN_LABELS, split) if DELETE_FRCNN_LABELS else ""

    if not os.path.isdir(img_dir):
        print(f"??  Skipping missing folder: {img_dir}")
        continue

    for fname in os.listdir(img_dir):
        if not fname.lower().endswith(IMG_EXTS):
            continue

        img_path   = os.path.join(img_dir, fname)
        yolo_label = os.path.join(yolo_dir,  fname.rsplit(".", 1)[0] + ".txt")
        frcnn_label= os.path.join(frcnn_dir, fname.rsplit(".", 1)[0] + ".txt")

        total_checked += 1
        try:
            with Image.open(img_path) as im:
                im.verify()            # validate without loading fully
        except (UnidentifiedImageError, OSError):
            # ---------- corrupted file ----------
            corrupted_files.append(img_path)
            # delete image
            os.remove(img_path)
            # delete YOLO label if it exists
            if os.path.isfile(yolo_label):
                os.remove(yolo_label)
            # delete Faster-R-CNN label if enabled & exists
            if DELETE_FRCNN_LABELS and os.path.isfile(frcnn_label):
                os.remove(frcnn_label)
            total_deleted += 1

# ===================== SUMMARY ==============================================
print("==============================================")
print(f"Images checked : {total_checked}")
print(f"Corrupted found: {total_deleted}")
if total_deleted:
    print("? All corrupted images and their labels have been removed.")
else:
    print("? No corrupted images detected.")
