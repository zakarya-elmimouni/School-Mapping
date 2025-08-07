from ultralytics import YOLO

# Path to your YAML file
DATA_YAML = 'dataset/peru/manual_labeled_data/data.yaml'


model = YOLO('results/peru/rslt_yolo10n_auto_labeling/exp/weights/best.pt') # existing model pretrained on auto-labeled data
# Training
results = model.train(
    data=DATA_YAML,
    epochs=50,
    imgsz=500 ,
    batch=64,
    lr0=0.001,
    lrf=0.01,
    pretrained=True,
    seed=0,
    device=[0,1],
    project='results/peru/rslt_yolo10n_finetune_auto_on_golden', # change this to your desired save path
    name='exp',
    save=True,
    plots=True,
    patience=6,  # early stopping
    save_period=10,
    verbose=True
   # translate=0.2,
   # degrees=15,
   # auto_augment=None
)

print(f"? Best model saved at: {results.save_dir}/weights/best.pt")
print(f"? All training plots saved at: {results.save_dir}")

best_model = YOLO(f"{results.save_dir}/weights/best.pt")

# �valuer sur le dataset de validation
val_results = best_model.val(data=DATA_YAML, split="val")

# Extraire le mAP@50
print(f"mAP@50: {val_results.box.map50:.4f}")