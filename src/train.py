"""
Training script for CIFAR-10.

Default: 100 epochs, SGD with cosine LR schedule, momentum=0.9, weight_decay=5e-4.
On a single modern GPU (~RTX 3060), each network trains in ~30-60 minutes.
On CPU, expect 4-8 hours per network — fine for an overnight run.

Usage:
    python train.py --config configs/resnet20.yaml
    python train.py --config configs/plainnet20.yaml
"""

import argparse
import os
import time
import yaml

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models import resnet20, plainnet20


MODELS = {
    "resnet20": resnet20,
    "plainnet20": plainnet20,
}


def get_dataloaders(batch_size=128, num_workers=2, data_dir="data"):
    """CIFAR-10 with standard augmentation."""
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)

    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_set = datasets.CIFAR10(data_dir, train=True, download=True, transform=train_tf)
    test_set = datasets.CIFAR10(data_dir, train=False, download=True, transform=test_tf)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                             num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, total_correct, total = 0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        total_correct += (out.argmax(1) == y).sum().item()
        total += x.size(0)
    return total_loss / total, total_correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, total_correct, total = 0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss = criterion(out, y)
        total_loss += loss.item() * x.size(0)
        total_correct += (out.argmax(1) == y).sum().item()
        total += x.size(0)
    return total_loss / total, total_correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    print(f"Config: {cfg}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader, test_loader = get_dataloaders(
        batch_size=cfg.get("batch_size", 128),
        num_workers=cfg.get("num_workers", 2),
    )

    model = MODELS[cfg["model"]]().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {cfg['model']} ({n_params:,} params)")

    optimizer = optim.SGD(
        model.parameters(),
        lr=cfg.get("lr", 0.1),
        momentum=cfg.get("momentum", 0.9),
        weight_decay=cfg.get("weight_decay", 5e-4),
        nesterov=True,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])
    criterion = nn.CrossEntropyLoss()

    os.makedirs("models", exist_ok=True)
    best_acc = 0

    for epoch in range(cfg["epochs"]):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - t0
        print(f"Epoch {epoch+1:3d}/{cfg['epochs']}  "
              f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
              f"test_loss={test_loss:.4f}  test_acc={test_acc:.4f}  "
              f"({elapsed:.1f}s)")

        # Save checkpoint each epoch (extension idea — visualize over training)
        if cfg.get("save_every_epoch", False):
            torch.save(model.state_dict(), f"models/{cfg['model']}_epoch{epoch+1:03d}.pt")

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), f"models/{cfg['model']}_best.pt")

    torch.save(model.state_dict(), f"models/{cfg['model']}_final.pt")
    print(f"\nBest test accuracy: {best_acc:.4f}")
    print(f"Saved final model to models/{cfg['model']}_final.pt")


if __name__ == "__main__":
    main()
