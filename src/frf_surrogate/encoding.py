"""Multi-scale Fourier (octave) encoding of frequency.

gamma(f) = [sin(2^k * pi * f~), cos(2^k * pi * f~)]_{k=0..K-1}

with f~ in [0, 1] the normalised frequency. Normalisation can be linear or
logarithmic over [f_min, f_max]; log spreads the octaves evenly across the
25 Hz - 10 kHz range.
"""

from __future__ import annotations

import numpy as np
import torch


class FrequencyNormalizer:
    def __init__(self, f_min: float, f_max: float, scale: str = "log"):
        if scale not in ("log", "linear"):
            raise ValueError("scale must be 'log' or 'linear'")
        self.f_min, self.f_max, self.scale = float(f_min), float(f_max), scale

    def __call__(self, f):
        f = np.asarray(f, dtype=float)
        if self.scale == "log":
            return (np.log(f) - np.log(self.f_min)) / (np.log(self.f_max) - np.log(self.f_min))
        return (f - self.f_min) / (self.f_max - self.f_min)

    def state_dict(self):
        return {"f_min": self.f_min, "f_max": self.f_max, "scale": self.scale}


def fourier_encode(f_norm, n_bands: int = 8, include_raw: bool = False):
    """Encode normalised frequency into 2*n_bands (+1) features. Accepts numpy or torch."""
    is_torch = isinstance(f_norm, torch.Tensor)
    lib = torch if is_torch else np
    f = f_norm[..., None]
    k = lib.arange(n_bands, dtype=f.dtype) if is_torch else np.arange(n_bands, dtype=float)
    ang = (2.0 ** k) * np.pi * f
    parts = [lib.sin(ang), lib.cos(ang)]
    if include_raw:
        parts.insert(0, f)
    return lib.cat(parts, -1) if is_torch else np.concatenate(parts, -1)


def encoding_dim(n_bands: int, include_raw: bool = False) -> int:
    return 2 * n_bands + int(include_raw)
