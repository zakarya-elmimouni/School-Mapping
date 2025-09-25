from ultralytics import YOLO

# Path to YAML file

DATA_YAML = 'dataset/brazil/manual_labeled_data/data.yaml' # path to data.yaml file (change as needed)
# Model: YOLO10n
model = YOLO('yolov10n.pt')

# Training
results = model.train(
    data=DATA_YAML,
    epochs=60,
    imgsz=500 ,
    batch=64,
    lr0=0.001,
    lrf=0.01,
    pretrained=True,
    seed=0,
    device=[0,1],
    project='path to where you want to store results', # saving directory
    name='exp',
    save=True,
    plots=True,
    patience=8,  # early stopping
    save_period=20,
    verbose=True

)

print(f"? Best model saved at: {results.save_dir}/weights/best.pt")
print(f"? All training plots saved at: {results.save_dir}")