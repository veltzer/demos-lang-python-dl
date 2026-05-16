#!/usr/bin/env python

"""Train a small CNN on a subset of real MNIST and report test accuracy."""

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from torchvision import datasets, transforms


def main() -> tuple[list[float], float]:
    torch.manual_seed(0)

    tform = transforms.ToTensor()
    train_full = datasets.MNIST(root="data", train=True, download=True, transform=tform)
    test_full = datasets.MNIST(root="data", train=False, download=True, transform=tform)

    train_ds = Subset(train_full, range(2000))
    test_ds = Subset(test_full, range(1000))

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=256)

    model = nn.Sequential(
        nn.Conv2d(1, 8, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(8, 16, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(16 * 7 * 7, 10),
    )

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    epoch_losses: list[float] = []
    for epoch in range(3):
        model.train()
        total_loss = 0.0
        seen = 0
        for xb, yb in train_loader:
            logits = model(xb)
            loss = loss_fn(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(xb)
            seen += len(xb)
        avg = total_loss / seen
        epoch_losses.append(avg)
        print(f"epoch {epoch}: train_loss={avg:.4f}")

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for xb, yb in test_loader:
            preds = model(xb).argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += len(yb)
    test_acc = correct / total
    print(f"test_acc={test_acc:.4f}")

    return epoch_losses, test_acc


if __name__ == "__main__":
    main()
