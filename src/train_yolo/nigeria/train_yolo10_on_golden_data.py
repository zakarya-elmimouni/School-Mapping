from ultralytics import YOLO

# Path to  YAML file
DATA_YAML = 'dataset/nigeria/manual_labeled_data/data.yaml' # path to data.yaml file , change as needed 

# Model: YOLO10n
model = YOLO('yolov10n.pt')

# Training
results = model.train(
    data=DATA_YAML,
    epochs=120,
    imgsz=500 ,
    batch=64,
    lr0=0.01,
    lrf=0.001,
    pretrained=True,
    seed=0,
    device=[0,1],
    project='results/nigeria/rslt_yolo10n_manaul_labeled_data', # path to saving directory (change as needed )
    name='exp',
    save=True,
    plots=True,
    patience=10,  # early stopping
    save_period=20,
    verbose=True

)

print(f"? Best model saved at: {results.save_dir}/weights/best.pt")
print(f"? All training plots saved at: {results.save_dir}")