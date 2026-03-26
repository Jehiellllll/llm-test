from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class TorchClassifierHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class TorchHeadWrapper:
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 128, lr: float = 1e-2, epochs: int = 25, batch_size: int = 64, device: str = "cpu"):
        self.device = torch.device(device)
        self.model = TorchClassifierHead(in_dim=in_dim, out_dim=out_dim, hidden_dim=hidden_dim).to(self.device)
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size

    def fit(self, x: np.ndarray, y: np.ndarray):
        x_t = torch.tensor(x, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.long)
        loader = DataLoader(TensorDataset(x_t, y_t), batch_size=self.batch_size, shuffle=True)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        loss_fn = nn.CrossEntropyLoss()
        self.model.train()
        for _ in range(self.epochs):
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                opt.zero_grad()
                loss = loss_fn(self.model(xb), yb)
                loss.backward()
                opt.step()
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            logits = self.model(torch.tensor(x, dtype=torch.float32, device=self.device))
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        return probs

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.predict_proba(x).argmax(axis=1)
