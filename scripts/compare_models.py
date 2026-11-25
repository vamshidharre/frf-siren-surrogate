"""Activation / architecture study over several seeds.

    python scripts/compare_models.py --config configs/default.yaml --variants configs/baselines.yaml

Writes runs/compare/summary.csv with mean +- std of test skill and dB MAE.
"""

import argparse
import copy
from pathlib import Path

import pandas as pd

from _common import load_config
from run_pipeline import run

p = argparse.ArgumentParser()
p.add_argument("--config", default="configs/default.yaml")
p.add_argument("--variants", default="configs/baselines.yaml")
p.add_argument("--data-dir")
p.add_argument("--epochs", type=int)
p.add_argument("--out", default="runs/compare")
a = p.parse_args()

base = load_config(a.config)
spec = load_config(a.variants)
if a.data_dir:
    base["data_dir"] = a.data_dir
if a.epochs:
    base["train"]["epochs"] = a.epochs

rows = []
for name, mcfg in spec["variants"].items():
    for seed in spec.get("seeds", [42]):
        cfg = copy.deepcopy(base)
        cfg["model"] = mcfg
        cfg["train"]["seed"] = seed          # split seed stays fixed; only init/order changes
        cfg["out_dir"] = str(Path(a.out) / f"{name}_s{seed}")
        print(f"\n=== {name}  seed {seed} ===")
        m = run(cfg, verbose=False)
        rows.append({"variant": name, "seed": seed, "params": m["n_params"],
                     "test_skill": m["test"]["skill"], "test_db_mae": m["test"]["db_mae"],
                     "train_r2": m["train"]["r2_per_freq"], "test_r2": m["test"]["r2_per_freq"]})
        print(rows[-1])

df = pd.DataFrame(rows)
df.to_csv(Path(a.out) / "runs.csv", index=False)
summary = df.groupby("variant").agg(params=("params", "first"),
                                    skill_mean=("test_skill", "mean"), skill_std=("test_skill", "std"),
                                    mae_mean=("test_db_mae", "mean"), mae_std=("test_db_mae", "std"))
summary = summary.sort_values("skill_mean", ascending=False)
summary.to_csv(Path(a.out) / "summary.csv")
print("\n", summary.round(3).to_string())
