import os
import torch
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.train_utils import train_one_epoch, evaluate
from GroundingDINO.groundingdino.util.misc import get_optimizer_and_scheduler
from GroundingDINO.groundingdino.util.get_args import get_args_parser
from GroundingDINO.groundingdino.util.dist import init_distributed_mode


class BestCheckpointSaver:
    def __init__(self, save_path):
        self.best_loss = float("inf")
        self.save_path = save_path

    def update(self, model, val_loss):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            torch.save(model.state_dict(), self.save_path)
            print(f"? Saved new best model with val_loss = {val_loss:.4f}")


def main():
    # === Load config ===
    config_path = "GroundingDINO/configs/brazil_groundingdino.yaml"
    config = SLConfig.fromfile(config_path)
    config.device = "cuda"

    # === Build model ===
    model = build_model(config)
    model.to(config.device)

    # === Optimizer & LR scheduler ===
    optimizer, lr_scheduler = get_optimizer_and_scheduler(config, model)

    # === Data loaders ===
    from groundingdino.util.data_utils import build_dataloader
    dataloaders = build_dataloader(config)
    train_loader = dataloaders["train"]
    val_loader = dataloaders["val"]

    # === Training loop ===
    num_epochs = config.epochs
    output_dir = config.output_dir
    os.makedirs(output_dir, exist_ok=True)

    saver = BestCheckpointSaver(os.path.join(output_dir, "best_model.pth"))

    for epoch in range(num_epochs):
        print(f"\n Epoch {epoch+1}/{num_epochs}")
        train_one_epoch(model, optimizer, train_loader, device=config.device, epoch=epoch)

        val_stats = evaluate(model, val_loader, device=config.device)
        val_loss = val_stats["loss"]
        print(f"Validation loss = {val_loss:.4f}")
        saver.update(model, val_loss)

        lr_scheduler.step()


if __name__ == "__main__":
    main()
