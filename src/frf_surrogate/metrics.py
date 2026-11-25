"""FRF error metrics. All inputs are (n_designs, n_freq) arrays in dB unless noted."""

from __future__ import annotations

import numpy as np


def skill(y_true_db, y_pred_db, baseline_db) -> float:
    """1 - MSE_model / MSE_baseline, baseline = per-frequency mean of the training curves."""
    mse_m = np.mean((y_true_db - y_pred_db) ** 2)
    mse_b = np.mean((y_true_db - baseline_db[None, :]) ** 2)
    return float(1.0 - mse_m / mse_b)


def db_mae(y_true_db, y_pred_db) -> float:
    return float(np.mean(np.abs(y_true_db - y_pred_db)))


def r2_per_frequency(y_true_db, y_pred_db) -> float:
    """R^2 across designs at each frequency, averaged over frequency."""
    ss_res = ((y_true_db - y_pred_db) ** 2).sum(0)
    ss_tot = ((y_true_db - y_true_db.mean(0)) ** 2).sum(0)
    return float(np.mean(1.0 - ss_res / np.maximum(ss_tot, 1e-12)))


def frac(y_true_db, y_pred_db) -> np.ndarray:
    """Frequency Response Assurance Criterion per design, on linear magnitudes."""
    a = 10 ** (y_true_db / 20.0)
    b = 10 ** (y_pred_db / 20.0)
    num = (a * b).sum(1) ** 2
    den = (a * a).sum(1) * (b * b).sum(1)
    return num / den


def _top_peaks(y, k):
    # local maxima, highest k
    idx = np.where((y[1:-1] > y[:-2]) & (y[1:-1] > y[2:]))[0] + 1
    return idx[np.argsort(y[idx])[::-1][:k]]


def peak_errors(y_true_db, y_pred_db, freq, k: int = 5, search_hz: float = 100.0):
    """For the k highest true peaks of each design, find the nearest predicted local maximum
    within +-search_hz and report |delta f| (Hz) and |delta amplitude| (dB)."""
    df, da, missed = [], [], 0
    for yt, yp in zip(y_true_db, y_pred_db):
        pp = np.where((yp[1:-1] > yp[:-2]) & (yp[1:-1] > yp[2:]))[0] + 1
        for i in _top_peaks(yt, k):
            near = pp[np.abs(freq[pp] - freq[i]) <= search_hz] if len(pp) else pp
            if len(near) == 0:
                missed += 1
                continue
            j = near[np.argmin(np.abs(freq[near] - freq[i]))]
            df.append(abs(freq[j] - freq[i]))
            da.append(abs(yp[j] - yt[i]))
    return {
        "peak_freq_err_hz": float(np.median(df)) if df else float("nan"),
        "peak_amp_err_db": float(np.median(da)) if da else float("nan"),
        "peaks_missed_frac": missed / max(1, missed + len(df)),
    }


def report(y_true_db, y_pred_db, baseline_db, freq) -> dict:
    out = {
        "skill": skill(y_true_db, y_pred_db, baseline_db),
        "db_mae": db_mae(y_true_db, y_pred_db),
        "r2_per_freq": r2_per_frequency(y_true_db, y_pred_db),
        "frac_mean": float(frac(y_true_db, y_pred_db).mean()),
    }
    out.update(peak_errors(y_true_db, y_pred_db, freq))
    return out
