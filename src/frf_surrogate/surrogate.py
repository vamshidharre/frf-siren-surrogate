"""Training and inference for the coordinate surrogate.

Each design contributes n_freq rows: input [x_screened (standardised), gamma(f)],
target = standardised dB amplitude at f. The network is queried at every grid
frequency to rebuild a full FRF.
"""

from __future__ import annotations

import copy
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .encoding import FrequencyNormalizer, encoding_dim, fourier_encode
from .models import build_model, n_params


@dataclass
class Scalers:
    x_mean: np.ndarray
    x_std: np.ndarray
    y_mean: float
    y_std: float

    def to_dict(self):
        return {"x_mean": self.x_mean.tolist(), "x_std": self.x_std.tolist(),
                "y_mean": self.y_mean, "y_std": self.y_std}

    @classmethod
    def from_dict(cls, d):
        return cls(np.asarray(d["x_mean"]), np.asarray(d["x_std"]), d["y_mean"], d["y_std"])


@dataclass
class Surrogate:
    model: nn.Module
    features: list
    scalers: Scalers
    fnorm: FrequencyNormalizer
    n_bands: int
    model_cfg: dict
    history: list = field(default_factory=list)

    # ---------------------------------------------------------------- inference
    @torch.no_grad()
    def predict_db(self, X_raw: np.ndarray, freq: np.ndarray, device: str = "cpu",
                   batch: int = 262_144) -> np.ndarray:
        """X_raw: (n_designs, len(features)) unscaled parameters -> (n_designs, n_freq) dB."""
        self.model.eval().to(device)
        xs = (X_raw - self.scalers.x_mean) / self.scalers.x_std
        enc = fourier_encode(self.fnorm(freq), self.n_bands)
        inp = _coordinate_inputs(xs, enc)
        out = []
        for i in range(0, len(inp), batch):
            out.append(self.model(torch.as_tensor(inp[i:i + batch], device=device)).cpu().numpy())
        y = np.concatenate(out).reshape(len(xs), len(freq))
        return y * self.scalers.y_std + self.scalers.y_mean

    # ------------------------------------------------------------------- I/O
    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": self.model.state_dict(),
            "meta": {
                "features": self.features,
                "scalers": self.scalers.to_dict(),
                "fnorm": self.fnorm.state_dict(),
                "n_bands": self.n_bands,
                "model_cfg": self.model_cfg,
                "in_dim": len(self.features) + encoding_dim(self.n_bands),
            },
        }, path)
        (path.with_suffix(".history.json")).write_text(json.dumps(self.history))

    @classmethod
    def load(cls, path: str | Path) -> "Surrogate":
        ck = torch.load(path, map_location="cpu", weights_only=False)
        m = ck["meta"]
        model = build_model(m["model_cfg"], m["in_dim"])
        model.load_state_dict(ck["state_dict"])
        return cls(model, m["features"], Scalers.from_dict(m["scalers"]),
                   FrequencyNormalizer(**m["fnorm"]), m["n_bands"], m["model_cfg"])


def _coordinate_inputs(xs: np.ndarray, enc: np.ndarray) -> np.ndarray:
    n, nf = len(xs), len(enc)
    a = np.repeat(xs, nf, axis=0)
    b = np.tile(enc, (n, 1))
    return np.concatenate([a, b], 1).astype(np.float32)


def train_surrogate(X_tr: np.ndarray, Y_tr_db: np.ndarray, X_va: np.ndarray, Y_va_db: np.ndarray,
                    freq: np.ndarray, features: list, model_cfg: dict, train_cfg: dict,
                    enc_cfg: dict, device: str = "cpu", verbose: bool = True) -> Surrogate:
    torch.manual_seed(train_cfg.get("seed", 42))
    np.random.seed(train_cfg.get("seed", 42))

    x_mean, x_std = X_tr.mean(0), X_tr.std(0) + 1e-12
    y_mean, y_std = float(Y_tr_db.mean()), float(Y_tr_db.std())
    scalers = Scalers(x_mean, x_std, y_mean, y_std)

    fnorm = FrequencyNormalizer(freq.min(), freq.max(), enc_cfg.get("scale", "log"))
    n_bands = enc_cfg.get("n_bands", 8)
    enc = fourier_encode(fnorm(freq), n_bands)

    def tensors(X, Y):
        inp = _coordinate_inputs((X - x_mean) / x_std, enc)
        tgt = ((Y - y_mean) / y_std).reshape(-1).astype(np.float32)
        return torch.as_tensor(inp, device=device), torch.as_tensor(tgt, device=device)

    xi_tr, yt_tr = tensors(X_tr, Y_tr_db)
    xi_va, yt_va = tensors(X_va, Y_va_db)
    n_feat = X_tr.shape[1]

    model = build_model(model_cfg, xi_tr.shape[1]).to(device)
    if verbose:
        print(f"model: {model_cfg.get('type', 'siren')}  params: {n_params(model):,}  "
              f"rows: train {len(xi_tr):,} / val {len(xi_va):,}")

    epochs = train_cfg.get("epochs", 200)
    bs = train_cfg.get("batch_size", 4096)
    opt = torch.optim.AdamW(model.parameters(), lr=train_cfg.get("lr", 1e-3),
                            weight_decay=train_cfg.get("weight_decay", 1e-4))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=train_cfg.get("lr_min", 1e-5))
    sigma = train_cfg.get("input_noise", 0.0)
    patience = train_cfg.get("patience", 25)
    loss_fn = nn.MSELoss()

    best, best_state, bad, history = float("inf"), None, 0, []
    for ep in range(epochs):
        t0 = time.time()
        model.train()
        perm = torch.randperm(len(xi_tr), device=device)
        tot = 0.0
        for i in range(0, len(perm), bs):
            j = perm[i:i + bs]
            bx, by = xi_tr[j], yt_tr[j]
            if sigma > 0:
                bx = bx.clone()
                bx[:, :n_feat] += torch.randn_like(bx[:, :n_feat]) * sigma
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(bx), by)
            loss.backward()
            opt.step()
            tot += loss.item() * len(j)
        sched.step()

        model.eval()
        with torch.no_grad():
            va = sum(loss_fn(model(xi_va[i:i + 65536]), yt_va[i:i + 65536]).item() * len(yt_va[i:i + 65536])
                     for i in range(0, len(xi_va), 65536)) / len(xi_va)
        tr = tot / len(xi_tr)
        history.append({"epoch": ep + 1, "train_mse": tr, "val_mse": va})
        if verbose and (ep == 0 or (ep + 1) % max(1, epochs // 20) == 0):
            print(f"epoch {ep + 1:4d}  train {tr:.4f}  val {va:.4f}  ({time.time() - t0:.1f}s)")
        if va < best - 1e-6:
            best, best_state, bad = va, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                if verbose:
                    print(f"early stop at epoch {ep + 1}; best val {best:.4f}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)       # restore best weights
    return Surrogate(model.cpu(), list(features), scalers, fnorm, n_bands, model_cfg, history)
