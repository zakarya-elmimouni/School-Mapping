# -*- coding: utf-8 -*-
import os
import pandas as pd
import requests
from urllib.parse import urlencode

# ------------ Configuration ------------
API_KEY   = 'Your API key"
CSV_PATH  = "data/nigeria/Data_nigeria_11000_dedup.csv"  # the csv file in data folder contains the coordinates of school and non-school.

BASE_DIR  = "data"
SAT_DIR   = os.path.join(BASE_DIR, "satellite")
LABELS_CSV = "files_nigeria.csv"  # downloaded labels files

# Create sub-folders: "school" and "non_school"
for sub in ("school", "non_school"):
    os.makedirs(os.path.join(SAT_DIR, sub), exist_ok=True)

# ------------ Load input CSV ------------
df = pd.read_csv(CSV_PATH)
          

# ------------ Download loop ------------
labels = []

for _, row in df.iterrows():
    lat, lon, label = row["Latitude"], row["Longitude"], row["label"]

    # Sub-folder and filename
    sub_folder = "school" if label == "school" else "non_school"
    filename   = f"{label}_lat={lat}_lon={lon}.png"
    filepath   = os.path.join(SAT_DIR, sub_folder, filename)

    # Google Static Maps URL
    params = {
        "center": f"{lat},{lon}",
        "zoom"  : 18,
        "size"  : "500x500",
        "maptype": "satellite",
        "key"   : API_KEY,
    }
    url = f"https://maps.googleapis.com/maps/api/staticmap?{urlencode(params)}"

    # Request and save
    try:
        r = requests.get(url, timeout=15)
        if r.ok:
            with open(filepath, "wb") as img:
                img.write(r.content)
            labels.append([filename, "satellite", label, lat, lon])
        else:
            print(f"Failed for({lat}, {lon}) HTTP {r.status_code}")
    except requests.RequestException as err:
        print(f"[ERROR] Request error for ({lat}, {lon}): {err}")

# ------------ Export labels.csv ------------
pd.DataFrame(
    labels,
    columns=["filename", "modality", "label", "latitude", "longitude"]
).to_csv(LABELS_CSV, index=False)

print(f"[INFO] Download complete. Labels saved to {LABELS_CSV}")
