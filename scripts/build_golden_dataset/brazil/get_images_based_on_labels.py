import os
import shutil

"""This script copies images from a source directory to destination directories based on labels provided in text files."""

# Folders
label_base = 'dataset/manual_labeled_data/labels' # change if needed
image_src = os.path.join('data', 'brazil', 'school') # change if needed
image_dst_base = 'images'
subdirs = ['train', 'val', 'test']

# Extensions 
img_exts = ['.jpg', '.png']

for sub in subdirs:
    label_dir = os.path.join(label_base, sub)
    dst_dir = os.path.join(image_dst_base, sub)
    os.makedirs(dst_dir, exist_ok=True)
    for label_file in os.listdir(label_dir):
        if label_file.endswith('.txt'):
            img_name = os.path.splitext(label_file)[0]
            found = False
            for ext in img_exts:
                img_file = img_name + ext
                src_img_path = os.path.join(image_src, img_file)
                if os.path.exists(src_img_path):
                    shutil.copy(src_img_path, dst_dir)
                    print(f"Copied: {img_file} -> {dst_dir}")
                    found = True
                    break
            if not found:
                print(f"Image not found for label: {label_file}")