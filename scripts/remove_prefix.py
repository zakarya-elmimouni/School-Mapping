import os

# Dossier contenant les images
folder_path = "dataset/brazil/manual_labeled_data/images/val"  # <-- remplace par ton chemin

# 
prefix_to_remove = "manual_labeled_"

# Parcours des fichiers
for filename in os.listdir(folder_path):
    if filename.startswith(prefix_to_remove):
        new_name = filename[len(prefix_to_remove):]
        src = os.path.join(folder_path, filename)
        dst = os.path.join(folder_path, new_name)
        os.rename(src, dst)
        print(f"Renamed: {filename} -> {new_name}")
