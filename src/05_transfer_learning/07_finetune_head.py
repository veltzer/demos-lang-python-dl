#!/usr/bin/env python

"""Freeze DistilBERT, train a tiny classifier head on a few SST-2 examples."""

import torch
from torch import nn

from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer


def main() -> tuple[list[float], bool]:
    torch.manual_seed(0)

    name = "distilbert-base-uncased"
    tok = AutoTokenizer.from_pretrained(name)
    backbone = AutoModel.from_pretrained(name)

    for p in backbone.parameters():
        p.requires_grad = False
    backbone.eval()

    ds = load_dataset("glue", "sst2", split="train").select(range(64))
    enc = tok(
        ds["sentence"],
        padding="max_length",
        truncation=True,
        max_length=32,
        return_tensors="pt",
    )
    labels = torch.tensor(ds["label"], dtype=torch.long)

    hidden = backbone.config.hidden_size
    head = nn.Linear(hidden, 2)
    opt = torch.optim.Adam(head.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    losses: list[float] = []
    for epoch in range(5):
        with torch.no_grad():
            out = backbone(**enc)
            cls = out.last_hidden_state[:, 0]

        logits = head(cls)
        loss = loss_fn(logits, labels)

        opt.zero_grad()
        loss.backward()
        opt.step()

        losses.append(loss.item())
        print(f"epoch {epoch}: loss={loss.item():.4f}")

    backbone_frozen = all(p.grad is None for p in backbone.parameters())
    print("backbone gradients still None:", backbone_frozen)
    return losses, backbone_frozen


if __name__ == "__main__":
    main()
