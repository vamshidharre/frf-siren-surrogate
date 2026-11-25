"""Screening -> training -> test evaluation, from one config file.

    python scripts/run_pipeline.py --config configs/default.yaml
    python scripts/run_pipeline.py --config configs/default.yaml --data-dir data/synthetic --epochs 20

Writes to out_dir: model.pt, screening.csv, split.json, metrics.json,
test_predictions.npz and a few plots.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from _common import load_config
from frf_surrogate.data import FRFDataset, split_designs
from frf_surrogate.metrics import report
from frf_surrogate.models import n_params
from frf_surrogate.screening import screen
from frf_surrogate.surrogate import train_surrogate


def run(cfg, verbose=True):
    out = Path(cfg["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    ds = FRFDataset.load(cfg["data_dir"])
    tr, va, te = split_designs(ds.n_designs, tuple(cfg["split"]["frac"]), cfg["split"]["seed"])
    (out / "split.json").write_text(json.dumps({"train": tr.tolist(), "val": va.tolist(), "test": te.tolist()}))
    if verbose:
        print(f"{ds.n_designs} designs x {ds.X.shape[1]} params, {len(ds.freq)} freqs | "
              f"split {len(tr)}/{len(va)}/{len(te)}")

    # 1. screening on train + val only (test designs never seen)
    sc = cfg["screening"]
    feats_path = out / "screening.csv"
    trva = np.concatenate([tr, va])
    features, table = screen(ds.X.iloc[trva], ds.Y_db[trva], ds.freq, sc["top_k"], sc.get("n_bands", 12),
                             cfg["split"]["seed"], sc.get("use_shap", True))
    table.to_csv(feats_path)
    if verbose:
        print("screened features:", features)

    # 2. train
    X = ds.X[features].values
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sur = train_surrogate(X[tr], ds.Y_db[tr], X[va], ds.Y_db[va], ds.freq, features,
                          cfg["model"], cfg["train"], cfg["encoding"], device, verbose)
    sur.save(out / "model.pt")

    # 3. evaluate
    baseline = ds.Y_db[tr].mean(0)
    metrics = {"n_params": n_params(sur.model), "features": features}
    for name, idx in (("train", tr), ("val", va), ("test", te)):
        pred = sur.predict_db(X[idx], ds.freq, device)
        metrics[name] = report(ds.Y_db[idx], pred, baseline, ds.freq)
        if name == "test":
            np.savez_compressed(out / "test_predictions.npz", design_idx=te, y_true_db=ds.Y_db[te],
                                y_pred_db=pred, freq=ds.freq, baseline_db=baseline)
    import time
    t0 = time.perf_counter()
    for _ in range(20):
        sur.predict_db(X[te[:1]], ds.freq, device)
    metrics["inference_ms_per_curve"] = (time.perf_counter() - t0) / 20 * 1e3
    metrics["device"] = device
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    if verbose:
        for k in ("train", "val", "test"):
            m = metrics[k]
            print(f"{k:5s} skill {m['skill']:+.3f}  dB MAE {m['db_mae']:.2f}  R2 {m['r2_per_freq']:.3f}  "
                  f"FRAC {m['frac_mean']:.3f}")
        print(f"inference: {metrics['inference_ms_per_curve']:.1f} ms per {len(ds.freq)}-point curve ({device})")
    return metrics


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--data-dir")
    p.add_argument("--out-dir")
    p.add_argument("--epochs", type=int)
    a = p.parse_args()
    cfg = load_config(a.config)
    if a.data_dir:
        cfg["data_dir"] = a.data_dir
    if a.out_dir:
        cfg["out_dir"] = a.out_dir
    if a.epochs:
        cfg["train"]["epochs"] = a.epochs
    run(cfg)
    try:
        from plot_results import plot_run
        plot_run(cfg["out_dir"])
    except ImportError:
        pass
