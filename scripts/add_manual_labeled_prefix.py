import os

# Paths to image and label folders
image_dir = 'dataset/brazil/manual_labeled_data/images/val'
#label_dir = 'dataset/brazil/manual_labeled_data/labels/test'




# Prefix to add
prefix = 'manual_labeled_'

# Loop through all image files
for filename in os.listdir(image_dir):
    if filename.endswith('.png'):  # adapt this if needed
        # Original image path
        old_image_path = os.path.join(image_dir, filename)
        # New image name and path
        new_image_name = prefix + filename
        new_image_path = os.path.join(image_dir, new_image_name)

        # Rename image file
        os.rename(old_image_path, new_image_path)
        print("image renamed")

        # Rename corresponding label file
        #label_filename = os.path.splitext(filename)[0] + '.txt'
        #old_label_path = os.path.join(label_dir, label_filename)
        #new_label_name = prefix + os.path.splitext(filename)[0] + '.txt'
        #new_label_path = os.path.join(label_dir, new_label_name)

        #if os.path.exists(old_label_path):
         #   os.rename(old_label_path, new_label_path)
        #else:
         #   print(f" Missing label for {filename}")