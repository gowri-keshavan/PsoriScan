
import argparse
import json
import random

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

from psoriscan.config import CLASS_NAMES, IMAGENET_MEAN, IMAGENET_STD, IMG_SIZE, MODEL_DIR, MODEL_FILES, REPORT_DIR, SEED
from psoriscan.data import list_split
from psoriscan.imaging import load_image, resize_for_model
from psoriscan.reporting import save_history_plot
from psoriscan.torch_model import build_resnet18

NORMALIZE = T.Normalize(IMAGENET_MEAN, IMAGENET_STD)
TRAIN_TF = T.Compose([
    T.RandomResizedCrop(IMG_SIZE, scale=(0.75, 1.0), ratio=(0.9, 1.1)),
    T.RandomHorizontalFlip(),
    T.RandomRotation(15),
    T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
    T.ToTensor(),
    NORMALIZE,
])
EVAL_TF = T.Compose([T.ToTensor(), NORMALIZE])


class SkinDataset(Dataset):
    """Images are resized with the shared serving preprocessing before augmentation."""

    def __init__(self, split: str, transform) -> None:
        self.items = list_split(split)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i):
        path, label = self.items[i]
        return self.transform(resize_for_model(load_image(path))), torch.tensor(label, dtype=torch.float32)


def run_epoch(model, loader, device, criterion, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss, correct, n = 0.0, 0, 0
    probs, targets = [], []
    with torch.set_grad_enabled(training):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x).squeeze(1)
            loss = criterion(logits, y)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            p = torch.sigmoid(logits.detach())
            total_loss += loss.item() * len(y)
            correct += ((p >= 0.5).float() == y).sum().item()
            n += len(y)
            probs += p.cpu().tolist()
            targets += y.cpu().tolist()
    auc = roc_auc_score(targets, probs) if len(set(targets)) == 2 else float("nan")
    return total_loss / n, correct / n, auc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--epochs-head", type=int, default=3)
    parser.add_argument("--epochs-finetune", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr-head", type=float, default=1e-3)
    parser.add_argument("--lr-finetune", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--no-pretrained", action="store_true", help="skip the ImageNet weight download")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = DataLoader(SkinDataset("train", TRAIN_TF), args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(SkinDataset("val", EVAL_TF), args.batch_size, num_workers=args.num_workers)
    print(f"device={device} train={len(train_loader.dataset)} val={len(val_loader.dataset)}")

    model = build_resnet18(pretrained=not args.no_pretrained).to(device)
    criterion = nn.BCEWithLogitsLoss()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / MODEL_FILES["torch"]
    history = {k: [] for k in ("loss", "accuracy", "auc", "val_loss", "val_accuracy", "val_auc")}
    best_val, stale = float("inf"), 0

    phases = [("head", args.epochs_head, args.lr_head), ("finetune", args.epochs_finetune, args.lr_finetune)]
    for phase, epochs, lr in phases:
        if epochs == 0:
            continue
        for name, param in model.named_parameters():
            param.requires_grad = phase == "finetune" or name.startswith("fc.")
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        stale = 0
        for epoch in range(1, epochs + 1):
            tr = run_epoch(model, train_loader, device, criterion, optimizer)
            va = run_epoch(model, val_loader, device, criterion)
            scheduler.step()
            for key, value in zip(history, (*tr, *va)):
                history[key].append(float(value))
            print(f"[{phase} {epoch}/{epochs}] loss={tr[0]:.4f} acc={tr[1]:.3f} | val_loss={va[0]:.4f} val_acc={va[1]:.3f} val_auc={va[2]:.3f}")
            if va[0] < best_val:
                best_val, stale = va[0], 0
                torch.save({"arch": "resnet18", "state_dict": model.state_dict(), "img_size": IMG_SIZE,
                            "class_names": list(CLASS_NAMES), "pretrained": not args.no_pretrained}, model_path)
            else:
                stale += 1
                if phase == "finetune" and stale >= args.patience:
                    print("Early stopping.")
                    break

    out_dir = REPORT_DIR / "torch"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    save_history_plot(history, out_dir / "training_curves.png", "PyTorch ResNet18")
    print(f"Best model (val_loss={best_val:.4f}) saved to {model_path}")


if __name__ == "__main__":
    main()
