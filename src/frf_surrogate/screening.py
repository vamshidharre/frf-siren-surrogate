"""Consensus feature screening: 255 material parameters -> top-k.

Four importance measures are computed against band-level dB targets, each is
turned into a rank, and the mean rank decides the final ordering:

1. |Pearson correlation|  (max over bands, so a parameter that only drives one
   resonance region is not averaged away)
2. Mutual information     (max over bands; catches non-linear effects such as GE)
3. Permutation importance (random forest, mean over bands)
4. SHAP |value|           (TreeExplainer on the same forest; skipped if `shap`
                           is not installed)

Screening must only see training + validation designs. Never pass test rows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.inspection import permutation_importance


def band_targets(Y_db: np.ndarray, freq: np.ndarray, n_bands: int = 12) -> np.ndarray:
    """Average dB level in n_bands log-spaced frequency bands -> (n_designs, n_bands)."""
    edges = np.geomspace(freq.min(), freq.max() * (1 + 1e-9), n_bands + 1)
    cols = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (freq >= lo) & (freq < hi)
        if m.any():
            cols.append(Y_db[:, m].mean(1))
    return np.stack(cols, 1)


def _rank(scores: pd.Series) -> pd.Series:
    return scores.rank(ascending=False, method="average")


def screen(X: pd.DataFrame, Y_db: np.ndarray, freq: np.ndarray, top_k: int = 10,
           n_bands: int = 12, seed: int = 42, use_shap: bool = True, n_jobs: int = -1
           ) -> tuple[list[str], pd.DataFrame]:
    T = band_targets(Y_db, freq, n_bands)
    Xv = X.values
    Xs = (Xv - Xv.mean(0)) / (Xv.std(0) + 1e-12)
    Ts = (T - T.mean(0)) / (T.std(0) + 1e-12)

    corr = np.abs(Xs.T @ Ts / len(Xs)).max(1)
    mi = np.stack([mutual_info_regression(Xv, T[:, b], random_state=seed) for b in range(T.shape[1])], 1).max(1)

    rf = RandomForestRegressor(n_estimators=300, min_samples_leaf=3, random_state=seed, n_jobs=n_jobs)
    rf.fit(Xv, Ts)
    perm = permutation_importance(rf, Xv, Ts, n_repeats=5, random_state=seed, n_jobs=n_jobs).importances_mean

    table = pd.DataFrame({"corr": corr, "mi": mi, "perm": perm}, index=X.columns)

    if use_shap:
        try:
            import shap  # noqa: F401
            expl = shap.TreeExplainer(rf)
            sv = np.asarray(expl.shap_values(Xv))       # (n, p, bands) or (bands, n, p)
            if sv.ndim == 3 and sv.shape[0] == T.shape[1]:
                sv = np.moveaxis(sv, 0, -1)
            table["shap"] = np.abs(sv).mean(axis=(0, 2)) if sv.ndim == 3 else np.abs(sv).mean(0)
        except ImportError:
            pass

    ranks = table.apply(_rank)
    table["mean_rank"] = ranks.mean(axis=1)
    table = table.sort_values("mean_rank")
    return list(table.index[:top_k]), table
