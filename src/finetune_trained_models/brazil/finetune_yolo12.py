from ultralytics import YOLO

# Path to your YAML file
DATA_YAML = 'dataset/brazil/manual_labeled_data/data.yaml'
#DATA_YAML = 'dataset/brazil/manual_labeled_data/data.yaml'
# Model: YOLO12n
#model = YOLO('yolo12n.pt')
model = YOLO('results/brazil/rslt_yolo12n_auto_labeling/exp/weights/best.pt')
# Training
results = model.train(
    data=DATA_YAML,
    epochs=50,
    imgsz=500 ,
    batch=64,
    lr0=0.0001,
    lrf=0.01,
    pretrained=True,
    seed=0,
    device=[0,1],
    project='results/brazil/rslt_yolo12n_finetune_auto_on_golden',
    name='exp',
    save=True,
    plots=True,
    patience=7,  # early stopping
    save_period=1,
    verbose=True
   # translate=0.2,
   # degrees=15,
   # auto_augment=None
)

print(f"? Best model saved at: {results.save_dir}/weights/best.pt")
print(f"? All training plots saved at: {results.save_dir}")