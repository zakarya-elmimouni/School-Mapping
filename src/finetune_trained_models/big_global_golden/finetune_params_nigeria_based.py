import os
import sys
import csv
import numpy as np
from ultralytics import YOLO


DATA_YAML = 'dataset/big_global_golden_dataset/data.yaml'


# === Step 1: Clone ECP if not already present ===
if not os.path.exists("ECP"):
    print("🔁 Cloning ECP repository...")
    os.system("git clone https://github.com/fouratifares/ECP.git")

# Add ECP to Python path
sys.path.append(os.path.abspath("ECP"))
from optimizers.ECP import ECP

# === Step 2: Define Objective Function ===
class YOLOObjective:
    def __init__(self, log_path="results/big_global_golden/ecp_yolo10n_finetuning_nigeria_based_log.csv"):
        self.bounds = np.array([
          [1e-4, 1e-2],     # 0: lr0            (default: 0.001)
          [0.01, 0.1],      # 1: lrf            (default: 0.01)
          [0.9, 0.98],     # 2: momentum       (default: 0.937)
          [1e-5, 0.005],    # 3: weight_decay   (default: 0.0005)
          [7.0, 10.0],      # 4: box            (default: 7.5)
          [0.0, 0.3],       # 5: translate      (default: 0.1)
          [0.2, 1.5],       # 6: cls            (default: 0.5)
          [0.8, 2.5],       # 7: dfl            (default: 1.5)
          [0., 0.4],       # 8: dropout          (default: 0.)
          [0.1, 0.5],       # 9: erasing        (default: 0.4)
        ])
        self.dimensions = self.bounds.shape[0]
        self.log_path = log_path

        # Initialize CSV logger
        with open(self.log_path, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["AP50"] + [f"x{i}" for i in range(self.dimensions)])

    def __call__(self, x):
        # Unpack hyperparameters
        lr0, lrf, momentum, weight_decay, box, translate, cls, dfl, dropout, erasing = map(float, x)

        print("\n🔁 ECP round starting with:")
        print(f"  lr0={lr0:.5f}, lrf={lrf:.4f}, momentum={momentum:.4f}, weight_decay={weight_decay:.5f}")
        print(f"  box={box:.2f}, translate={translate:.2f}, cls={cls:.2f}, dfl={dfl:.2f}, dropout={dropout:.2f}, erasing={erasing:.2f}")

        # Define experiment name
        exp_name = (
            f"lr{lr0:.1e}_lrf{lrf:.2f}_m{momentum:.2f}_wd{weight_decay:.4f}"
            f"_box{box:.1f}_t{translate:.2f}_cls{cls:.2f}_dfl{dfl:.2f}_dr{dropout:.2f}_erase{erasing:.2f}"
        )
        save_dir="results/big_global_golden/rslt_yolo10n_finetuning_params_finetuned_nigeria"

        try:
            model = YOLO("results/nigeria/rslt_yolo10n_finetune_auto_on_golden/exp/weights/best.pt")

            results = model.train(
                data=DATA_YAML,
                epochs=50,
                imgsz=500,
                batch=64,
                lr0=lr0,
                lrf=lrf,
                momentum=momentum,
                weight_decay=weight_decay,
                box=box,
                translate=translate,
                cls=cls,
                dfl=dfl,
                dropout=dropout,
                erasing=erasing,
                seed=0,
                device=[0, 1],
                pretrained=True,
                patience=8,
                save=True,
                save_period=20,
                verbose=False,
                project="results/big_global_golden/rslt_yolo10n_finetuning_params_finetuned_nigeria",
                name=exp_name,
            )

            
            best_model = YOLO(f"{save_dir}/{exp_name}/weights/best.pt")
            val_results = best_model.val(data=DATA_YAML, split="val")
            ap50=val_results.box.map50
            print(f"AP@50: {ap50}")
            
            

        except Exception as e:
            print(f"❌ Training failed: {e}")
            ap50 = 0.0

        # Log result to CSV
        with open(self.log_path, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([ap50] + list(x))

        return ap50

# === Step 3: Run ECP Optimization ===
if __name__ == "__main__":
    objective = YOLOObjective()

    # Run ECP with 50 evaluations (increase if resources allow)
    n_evals = 25
    points, values, epsilons = ECP(objective, n=n_evals)

    best_index = np.argmax(values)
    best_point = points[best_index]
    best_ap50 = values[best_index].item()

    print("\n✅ ECP Optimization Complete")
    print(f"🏁 Best Hyperparameters: {best_point}")
    print(f"🏆 Best AP50: {best_ap50:.4f}")
    print(f"📁 Full log saved to: {objective.log_path}")
