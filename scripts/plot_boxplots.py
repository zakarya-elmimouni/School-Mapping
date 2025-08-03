import os
import cv2
import numpy as np
import random
import matplotlib.pyplot as plt

countries = ["brazil", "nigeria", "colombia", "peru", "bangladesh"]
num_images = 1000
crop_size = 200
random_seed = 42
epsilon = 1e-8
random.seed(random_seed)
np.random.seed(random_seed)

summary_metrics = {c: {} for c in countries}

for country in countries:
    print(f"\n=== Processing country: {country.upper()} ===")
    images_dir = f"ibex/user/elmimoz/data/{country}/satellite/school"
    hist_output_dir = f"home/elmimoz/Project/official_implementation/figures/{country}/threshold_histograms_{country}"
    boxplot_output_dir = f"home/elmimoz/Project/official_implementation/figures/{country}/threshold_boxplots_{country}"
    os.makedirs(hist_output_dir, exist_ok=True)
    os.makedirs(boxplot_output_dir, exist_ok=True)

    vegetation, sahara, sea, blur = [], [], [], []

    imgs = [f for f in os.listdir(images_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    if len(imgs) == 0:
        print(f"No images found for {country}, skipping.")
        continue
    selected_imgs = imgs if num_images > len(imgs) else random.sample(imgs, num_images)

    for idx, fname in enumerate(selected_imgs, 1):
        img_path = os.path.join(images_dir, fname)
        img = cv2.imread(img_path)
        if img is None:
            continue

        H, W = img.shape[:2]
        cx, cy = W // 2, H // 2
        half = crop_size // 2
        left, top = max(0, cx - half), max(0, cy - half)
        right, bottom = min(W, cx + half), min(H, cy + half)
        if right - left != crop_size or bottom - top != crop_size:
            continue

        crop = img[top:bottom, left:right]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

        green = ((hsv[..., 0] > 30) & (hsv[..., 0] < 106) & (hsv[..., 1] > 40))
        vegetation.append(green.sum() / float(crop_size**2))

        sah = ((hsv[..., 0] > 15) & (hsv[..., 0] < 35) & (hsv[..., 1] > 50))
        sahara.append(sah.sum() / float(crop_size**2))

        sea_mask = ((hsv[..., 0] > 90) & (hsv[..., 0] < 130) & (hsv[..., 1] > 30))
        sea.append(sea_mask.sum() / float(crop_size**2))

        blur_val = 1 / (cv2.Laplacian(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var() + epsilon)
        blur.append(blur_val)

    metrics = {"vegetation": vegetation, "sahara": sahara, "sea": sea, "blur": blur}

    for metric, values in metrics.items():
        # Histogram
        plt.figure(figsize=(8, 6))
        plt.hist(values, bins=50, color='blue', alpha=0.7, edgecolor='black')
        plt.title(f"{country.title()} - {metric.title()} Histogram", fontweight='bold')
        plt.xlabel(metric.title(), fontweight='bold')
        plt.ylabel("Frequency", fontweight='bold')
        plt.xticks(fontweight='bold')
        plt.yticks(fontweight='bold')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(hist_output_dir, f"{metric}_hist.png"))
        plt.close()

        # Boxplot
        percentiles = np.percentile(values, [50, 75, 85, 95, 98])
        plt.figure(figsize=(8, 6))
        plt.boxplot(values, vert=True, showfliers=True)
        for p, val in zip([85, 95, 98], percentiles[2:]):
            plt.axhline(val, linestyle='--', color='red', label=f'{p}th percentile' if p == 85 else None)
        plt.title(f"{country.title()} - {metric.title()} Boxplot", fontweight='bold')
        plt.ylabel(metric.title(), fontweight='bold')
        plt.xticks(fontweight='bold')
        plt.yticks(fontweight='bold')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(boxplot_output_dir, f"{metric}_boxplot.png"))
        plt.close()

        summary_metrics[country][metric] = {
            "values": values,
            "percentiles": percentiles
        }

print("\nGenerating combined summary plots...")

summary_dir = "home/elmimoz/Project/official_implementation/figures/summary"
os.makedirs(summary_dir, exist_ok=True)

for metric in ["vegetation", "sahara", "sea", "blur"]:
    plt.figure(figsize=(10, 6))
    for country in countries:
        if metric in summary_metrics[country]:
            vals = summary_metrics[country][metric]["values"]
            plt.hist(vals, bins=50, alpha=0.4, label=country.title(), density=True, histtype='step')
    plt.title(f"Combined {metric.title()} Histograms", fontweight='bold')
    plt.xlabel(metric.title(), fontweight='bold')
    plt.ylabel("Density", fontweight='bold')
    plt.xticks(fontweight='bold')
    plt.yticks(fontweight='bold')
    plt.legend(fontsize=10, prop={'weight': 'bold'})
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(summary_dir, f"{metric}_histograms.png"))
    plt.close()

    plt.figure(figsize=(10, 6))
    data = [summary_metrics[c][metric]["values"] for c in countries if metric in summary_metrics[c]]
    labels = [c.title() for c in countries if metric in summary_metrics[c]]
    plt.boxplot(data, labels=labels, showfliers=True)
    plt.title(f"Combined {metric.title()} Boxplots", fontweight='bold')
    plt.ylabel(metric.title(), fontweight='bold')
    plt.xticks(fontweight='bold')
    plt.yticks(fontweight='bold')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(summary_dir, f"{metric}_boxplots.png"))
    plt.close()

print("\nGenerating summary table...")

header = ["Countries"]
for metric in ["Vegetation", "Sahara", "Sea", "Blur"]:
    header.extend([f"{metric}-{p}" for p in [50, 75, 85, 95, 98]])
lines = ["\t".join(header)]

for country in countries:
    row = [country.title()]
    for metric in ["vegetation", "sahara", "sea", "blur"]:
        if metric in summary_metrics[country]:
            percs = summary_metrics[country][metric]["percentiles"]
            row.extend([f"{v:.4f}" for v in percs])
        else:
            row.extend(["NA"] * 5)
    lines.append("\t".join(row))

with open(os.path.join(summary_dir, "threshold_percentiles.txt"), "w") as f:
    f.write("\n".join(lines))

print(f"\nAll done! Summary saved in {summary_dir}")
