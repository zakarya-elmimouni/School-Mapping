import os
import torch
import torchvision
import torchvision.transforms as T
import torchvision.models.detection as detection
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np

patience = 6
counter = 0
num_epochs = 80

# === Dataset Class ===
class FasterRCNNDataset(Dataset):
    def __init__(self, image_dir, label_dir, transforms=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transforms = transforms
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith(".jpg") or f.endswith(".png")]

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        label_path = os.path.join(self.label_dir, img_name.replace(".jpg", ".txt").replace(".png", ".txt"))

        img = Image.open(img_path).convert("RGB")
        width, height = img.size

        boxes = []
        labels = []

        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) == 5:
                        xmin, ymin, xmax, ymax, cls = map(float, parts)
                        boxes.append([xmin, ymin, xmax, ymax])
                        labels.append(int(cls))

        # Fix for empty labels
        if len(boxes) == 0:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
            labels_tensor = torch.tensor(labels, dtype=torch.int64)
        
        target = {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "image_id": torch.tensor([idx])
        }

        if self.transforms:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.image_files)

# === Transforms ===
def get_transform(train=True):
    transforms = []
    if train:
        transforms.append(
            T.RandomAffine(
                degrees=10,              
                translate=(0.1, 0.1),    
                fill=0                   
            )
        )

    transforms.append(T.ToTensor())
    return T.Compose(transforms)


# === IoU and metric evaluation ===
def compute_iou(box1, box2):
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    box1Area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2Area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return interArea / float(box1Area + box2Area - interArea + 1e-6)

def evaluate_metrics(model, dataloader, device, iou_threshold=0.5):
    model.eval()
    TP, FP, FN = 0, 0, 0
    with torch.no_grad():
        for images, targets in dataloader:
            images = [img.to(device) for img in images]
            outputs = model(images)
            for pred, target in zip(outputs, targets):
                pred_boxes = pred['boxes'].cpu().numpy()
                scores = pred['scores'].cpu().numpy()
                gt_boxes = target['boxes'].cpu().numpy()
                matched = []
                for pb, score in zip(pred_boxes, scores):
                    if score < 0.5:
                        continue
                    match_found = False
                    for i, gb in enumerate(gt_boxes):
                        if i in matched:
                            continue
                        iou = compute_iou(pb, gb)
                        if iou >= iou_threshold:
                            TP += 1
                            matched.append(i)
                            match_found = True
                            break
                    if not match_found:
                        FP += 1
                FN += len(gt_boxes) - len(matched)
    precision = TP / (TP + FP + 1e-6)
    recall = TP / (TP + FN + 1e-6)
    #ap50 = precision * recall
    return  precision, recall

# === Main training logic ===
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_dataset = FasterRCNNDataset(
        image_dir="dataset/brazil/dataset_yolo_auto_labeling/images/train",
        label_dir="dataset/brazil/dataset_faster_rcnn/labels/train",
        transforms=get_transform()
    )
    val_dataset = FasterRCNNDataset(
        image_dir="dataset/brazil/dataset_yolo_auto_labeling/images/val",
        label_dir="dataset/brazil/dataset_faster_rcnn/labels/val",
        transforms=get_transform()
    )


        
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))
    
    log_file = open("results/brazil/faster_rcnn/training_log.txt", "w")
    def log(msg):
        print(msg)
        log_file.write(msg + "\n")

    model = detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes=2)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    best_ap = 0
    best_val_loss = float('inf')


    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        for images, targets in train_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            total_train_loss += losses.item()

        lr_scheduler.step()

        precision, recall = evaluate_metrics(model, val_loader, device)
        log(f"[Epoch {epoch+1}] Train Loss: {total_train_loss:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}")

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for images, targets in val_loader:
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                total_val_loss += losses.item()
        avg_val_loss = total_val_loss / len(val_loader)
        
        log(f"[Epoch {epoch+1}] Train Loss: {total_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "results/brazil/faster_rcnn/faster_rcnn.pth")
            log("? Validation loss improved model saved!")
            counter = 0
        else:
            counter += 1
            log(f"?? No improvement. Patience {counter}/{patience}")
            if counter >= patience:
                log("? Early stopping triggered!")
                break

if __name__ == "__main__":
    main()
