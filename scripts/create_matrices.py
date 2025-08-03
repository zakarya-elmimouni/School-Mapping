import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# === Create output folder ===
os.makedirs("matrices_across_countries", exist_ok=True)

# === mAP@50 data ===
df_map = pd.DataFrame({
    "Model": [
        "Brazil", "Peru", "Colombia", "Nigeria", "Bangladesh",
        "Global Model"
    ],
    "Brazil":     [0.660, 0.441, 0.331, 0.315, 0.203, 0.595],
    "Peru":       [0.681, 0.905, 0.611, 0.494, 0.293, 0.865],
    "Colombia":   [0.330, 0.353, 0.589, 0.348, 0.168, 0.550],
    "Nigeria":    [0.408, 0.518, 0.421, 0.619, 0.319, 0.658],
    "Bangladesh": [0.436, 0.269, 0.248, 0.490, 0.665, 0.530],
})

# === F1@50 data ===
df_f1 = pd.DataFrame({
    "Model": [
        "Brazil", "Peru", "Colombia", "Nigeria", "Bangladesh",
        "Global Model"
    ],
    "Brazil":     [0.657, 0.459, 0.368, 0.398, 0.234, 0.607],
    "Peru":       [0.660, 0.878, 0.595, 0.546, 0.319, 0.830],
    "Colombia":   [0.372, 0.422, 0.592, 0.411, 0.221, 0.573],
    "Nigeria":    [0.434, 0.478, 0.468, 0.644, 0.327, 0.650],
    "Bangladesh": [0.480, 0.344, 0.344, 0.500, 0.643, 0.529],
})

# === Columns to compute average
test_cols = ["Brazil", "Peru", "Colombia", "Nigeria", "Bangladesh"]

# === Add average column
df_map["Average"] = df_map[test_cols].mean(axis=1)
df_f1["Average"] = df_f1[test_cols].mean(axis=1)

# === Set model as index
df_map.set_index("Model", inplace=True)
df_f1.set_index("Model", inplace=True)

# === Function to plot and save heatmap ===
def plot_heatmap(data, title, filename, cmap):
    plt.figure(figsize=(10, 5.5))
    sns.heatmap(
        data,
        annot=True,
        fmt=".3f",
        cmap=cmap,
        linewidths=0.5,
        annot_kws={"weight": "bold", "fontsize": 10},
        cbar=True
    )
    plt.title(title, fontsize=14)
    plt.xlabel("Test Country", fontsize=12)
    plt.ylabel("Model", fontsize=12)
    plt.xticks(rotation=0, fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()
    plt.savefig(f"matrices_across_countries/{filename}")
    plt.close()

# === Plot and save both heatmaps
plot_heatmap(df_map, "Cross-country mAP@50", "cross_country_map50_with_avg.png", "YlGnBu")
plot_heatmap(df_f1, "Cross-country F1@50", "cross_country_f1_with_avg.png", "YlOrRd")
