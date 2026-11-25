"""Coordinate networks: [x_screened, gamma(f)] -> y_dB(f).

* ResidualSIREN : sine activations with SIREN initialisation (Sitzmann et al. 2020)
                  and residual blocks. Final model: omega0 = 2, 3 blocks, width 128
                  (~103k parameters with 10 inputs + K = 8 bands).
* CoordinateMLP : the same coordinate formulation with GELU / SiLU / tanh, used as
                  activation baselines.
"""

from __future__ import annotations

import math

import torch
from torch import nn


class SineLayer(nn.Module):
    def __init__(self, in_f: int, out_f: int, omega0: float = 2.0, is_first: bool = False):
        super().__init__()
        self.omega0 = omega0
        self.linear = nn.Linear(in_f, out_f)
        with torch.no_grad():
            if is_first:
                bound = 1.0 / in_f
            else:
                bound = math.sqrt(6.0 / in_f) / omega0
            self.linear.weight.uniform_(-bound, bound)

    def forward(self, x):
        return torch.sin(self.omega0 * self.linear(x))


class SirenResBlock(nn.Module):
    def __init__(self, width: int, omega0: float):
        super().__init__()
        self.l1 = SineLayer(width, width, omega0)
        self.l2 = SineLayer(width, width, omega0)

    def forward(self, x):
        return x + self.l2(self.l1(x))


class ResidualSIREN(nn.Module):
    def __init__(self, in_dim: int, width: int = 128, n_blocks: int = 3, omega0: float = 2.0,
                 omega0_first: float | None = None):
        super().__init__()
        self.inp = SineLayer(in_dim, width, omega0_first or omega0, is_first=True)
        self.blocks = nn.Sequential(*[SirenResBlock(width, omega0) for _ in range(n_blocks)])
        self.out = nn.Linear(width, 1)
        with torch.no_grad():
            b = math.sqrt(6.0 / width) / omega0
            self.out.weight.uniform_(-b, b)

    def forward(self, x):
        return self.out(self.blocks(self.inp(x))).squeeze(-1)


_ACTS = {"gelu": nn.GELU, "silu": nn.SiLU, "tanh": nn.Tanh, "relu": nn.ReLU}


class CoordinateMLP(nn.Module):
    def __init__(self, in_dim: int, width: int = 192, depth: int = 14, activation: str = "gelu"):
        super().__init__()
        act = _ACTS[activation]
        layers = [nn.Linear(in_dim, width), act()]
        for _ in range(depth - 2):
            layers += [nn.Linear(width, width), act()]
        layers += [nn.Linear(width, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def build_model(cfg: dict, in_dim: int) -> nn.Module:
    kind = cfg.get("type", "siren").lower()
    if kind == "siren":
        return ResidualSIREN(in_dim, cfg.get("width", 128), cfg.get("n_blocks", 3),
                             cfg.get("omega0", 2.0), cfg.get("omega0_first"))
    return CoordinateMLP(in_dim, cfg.get("width", 192), cfg.get("depth", 14), kind)


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
