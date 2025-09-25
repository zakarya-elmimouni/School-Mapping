from ultralytics import YOLO

#path to the yaml file 
DATA_YAML = 'dataset/peru/dataset_yolo_auto_labeling/data.yaml' # data.yaml file (change as needed)


# Model: YOLO10n
model = YOLO('yolov10n.pt')

# Training yolo10 over auto_data
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
    project='results/peru/rslt_yolo10n_auto_labeling', # path to where you want to store results (change as needed)
    name='exp',
    save=True,
    plots=True,
    patience=10,  # early stopping
    save_period=20,
    verbose=True

)

print(f"? Best model saved at: {results.save_dir}/weights/best.pt")
print(f"? All training plots saved at: {results.save_dir}")