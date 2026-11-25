"""Plots for a finished run: loss curves, example test FRFs, error by frequency.

    python scripts/plot_results.py runs/siren_w2
"""

import json
import sys
from pathlib import Path

import numpy as np


def plot_run(run_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run = Path(run_dir)
    d = np.load(run / "test_predictions.npz")
    yt, yp, f, base = d["y_true_db"], d["y_pred_db"], d["freq"], d["baseline_db"]
    hist = json.loads((run / "model.history.json").read_text())

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.semilogy([h["train_mse"] for h in hist], label="train")
    ax.semilogy([h["val_mse"] for h in hist], label="validation")
    ax.set(xlabel="epoch", ylabel="MSE (standardised dB)", title="Training history")
    ax.legend()
    fig.tight_layout()
    fig.savefig(run / "loss.png", dpi=150)
    plt.close(fig)

    err = np.abs(yt - yp).mean(1)
    picks = {"best": int(np.argmin(err)), "median": int(np.argsort(err)[len(err) // 2]),
             "worst": int(np.argmax(err))}
    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
    for ax, (lab, i) in zip(axes, picks.items()):
        ax.plot(f, yt[i], lw=1.2, label="FE (ground truth)")
        ax.plot(f, yp[i], lw=1.2, label="surrogate")
        ax.plot(f, base, lw=0.8, ls=":", c="grey", label="train mean")
        ax.set_xscale("log")
        ax.set(ylabel="amplitude [dB]", title=f"{lab} test design (dB MAE {err[i]:.2f})")
    axes[0].legend(loc="lower left")
    axes[-1].set_xlabel("frequency [Hz]")
    fig.tight_layout()
    fig.savefig(run / "test_examples.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.plot(f, np.abs(yt - yp).mean(0), label="surrogate")
    ax.plot(f, np.abs(yt - base[None]).mean(0), label="mean-curve baseline", alpha=0.7)
    ax.set_xscale("log")
    ax.set(xlabel="frequency [Hz]", ylabel="MAE [dB]", title="Test error by frequency")
    ax.legend()
    fig.tight_layout()
    fig.savefig(run / "error_by_frequency.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    plot_run(sys.argv[1])
