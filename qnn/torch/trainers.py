# qnn/torch/trainers.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Callable

import torch
import torch.nn as nn
import torch.optim as optim

@dataclass
class TrainConfig:
    epochs: int = 30
    lr: float = 0.05
    bce_scale: float = 2.5  # scale expectation to logits for BCE

def train_binary_estimator(model: nn.Module, X: torch.Tensor, y01: torch.Tensor, cfg: TrainConfig = TrainConfig()):
    """Quick binary training loop for Estimator-based QNNs."""
    criterion = nn.BCEWithLogitsLoss()
    opt = optim.Adam(model.parameters(), lr=cfg.lr)
    for _ in range(cfg.epochs):
        opt.zero_grad()
        logits = model(X) * cfg.bce_scale
        loss = criterion(logits, y01.unsqueeze(-1).float())
        loss.backward()
        opt.step()
    with torch.no_grad():
        preds = (torch.sigmoid(model(X)*cfg.bce_scale) > 0.5).float().squeeze(-1)
        acc = (preds == y01.float()).float().mean().item()
    return acc
