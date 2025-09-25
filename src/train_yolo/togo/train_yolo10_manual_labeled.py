from ultralytics import YOLO

"""this scripts is used to train the yolov10n over data"""

# Path to your YAML file

DATA_YAML = 'dataset/togo/manual_labeled_data/data.yaml'  # data.yaml file (change as needed)
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
    project='path to where you want to store results ',
    name='exp',
    save=True,
    plots=True,
    patience=10,  # early stopping
    save_period=20,
    verbose=True
    #translate=0.2,
    #degrees=15,
    #auto_augment=None
)

print(f"? Best model saved at: {results.save_dir}/weights/best.pt")
print(f"? All training plots saved at: {results.save_dir}")