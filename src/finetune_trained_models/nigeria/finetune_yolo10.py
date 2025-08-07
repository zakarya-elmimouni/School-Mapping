from ultralytics import YOLO

# Path to your YAML file
DATA_YAML = 'dataset/peru/manual_labeled_data/data.yaml'


model = YOLO('results/nigeria/rslt_yolo10n_auto_labeling/exp/weights/best.pt') # the path to your pre-trained model on auto-dataset
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
    project='results/nigeria/rslt_yolo10n_finetune_auto_on_golden', # save results to this directory
    name='exp',
    save=True,
    plots=True,
    patience=6,  # early stopping
    save_period=10,
    verbose=True

)

print(f"? Best model saved at: {results.save_dir}/weights/best.pt")
print(f"? All training plots saved at: {results.save_dir}")