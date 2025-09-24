from ultralytics import YOLO

# Path to your YAML file
DATA_YAML = 'dataset/africa_global_golden_dataset/data.yaml'
#DATA_YAML = 'dataset/brazil/manual_labeled_data/data.yaml'
# Model: YOLO12n
#model = YOLO('yolov10n.pt')
model = YOLO('results/nigeria/rslt_yolo10n_finetune_auto_on_golden/exp/weights/best.pt')
# Training
results = model.train(
    data=DATA_YAML,
    epochs=100,
    imgsz=500 ,
    batch=64,
    lr0=0.001,
    lrf=0.01,
    pretrained=True,
    seed=0,
    device=[0,1],
    project='results/africa_global_golden/rslt_yolo10n_finetune_finetuned_nigeria_on_global_golden',
    name='exp',
    save=True,
    plots=True,
    patience=10,  # early stopping
    save_period=20,
    verbose=True
   # translate=0.2,
   # degrees=15,
   # auto_augment=None
)

print(f"? Best model saved at: {results.save_dir}/weights/best.pt")
print(f"? All training plots saved at: {results.save_dir}")