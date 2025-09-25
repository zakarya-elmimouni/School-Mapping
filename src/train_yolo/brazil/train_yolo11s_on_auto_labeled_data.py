from ultralytics import YOLO

# Path to your YAML file
DATA_YAML = 'dataset/brazil/dataset_yolo_auto_labeling/data.yaml' # data.yaml file (change as needed)
# Model: YOLO11s
model = YOLO('yolo11s.pt')

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
    project='results/brazil/rslt_yolo11s_auto_labeling', # saving directory (chnage as needed)
    name='exp',
    save=True,
    plots=True,
    patience=10,  # early stopping
    save_period=20,
    verbose=True

)

print(f"? Best model saved at: {results.save_dir}/weights/best.pt")
print(f"? All training plots saved at: {results.save_dir}")