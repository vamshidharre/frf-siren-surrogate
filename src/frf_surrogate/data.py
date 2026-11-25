"""Dataset assembly, dB transform and design-level splitting."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

EPS = 1e-30


def to_db(amplitude: np.ndarray, ref: float = 1.0) -> np.ndarray:
    """Amplitude -> dB (20 log10). Works for magnitude or complex input."""
    return 20.0 * np.log10(np.maximum(np.abs(amplitude), EPS) / ref)


def from_db(db: np.ndarray, ref: float = 1.0) -> np.ndarray:
    return ref * 10.0 ** (db / 20.0)


@dataclass
class FRFDataset:
    """One row per design variant.

    X      : (n_designs, n_params) material parameters (E_*, RHO_*, GE_*)
    Y_db   : (n_designs, n_freq) FRF amplitude in dB
    freq   : (n_freq,) frequency grid in Hz
    """
    X: pd.DataFrame
    Y_db: np.ndarray
    freq: np.ndarray

    @property
    def n_designs(self) -> int:
        return len(self.X)

    @classmethod
    def load(cls, data_dir: str | Path) -> "FRFDataset":
        """Load the standard layout written by scripts/build_dataset.py:

        data_dir/params.csv   design_id + one column per parameter
        data_dir/frf_amp.npy  (n_designs, n_freq) linear amplitude
        data_dir/freq.npy     (n_freq,) Hz
        """
        d = Path(data_dir)
        X = pd.read_csv(d / "params.csv", index_col=0)
        amp = np.load(d / "frf_amp.npy")
        freq = np.load(d / "freq.npy")
        if amp.shape != (len(X), len(freq)):
            raise ValueError(f"frf_amp {amp.shape} does not match params ({len(X)}) x freq ({len(freq)})")
        return cls(X=X, Y_db=to_db(amp), freq=freq)


def split_designs(n: int, frac=(0.70, 0.15, 0.15), seed: int = 42):
    """Split by design variant (whole curves), never by individual frequency points."""
    if abs(sum(frac) - 1.0) > 1e-9:
        raise ValueError("split fractions must sum to 1")
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_tr = int(round(frac[0] * n))
    n_va = int(round(frac[1] * n))
    return np.sort(idx[:n_tr]), np.sort(idx[n_tr:n_tr + n_va]), np.sort(idx[n_tr + n_va:])
