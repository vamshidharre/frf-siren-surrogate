"""Collect DoE results into the training layout.

Input
  --params   CSV exported from HyperStudy: one row per run, a run-id column plus
             one column per design variable (E_*, RHO_*, GE_*).
  --pch-dir  folder with one punch file per run. The file for run <id> is found
             with --pattern (default "{id}/*.pch", i.e. HyperStudy run folders).

Output (--out, default data/processed)
  params.csv   design_id + parameters (only runs whose punch file parsed)
  frf_amp.npy  (n_designs, n_freq) |H| at the chosen point/component
  freq.npy     (n_freq,) Hz, after interpolation onto a uniform grid

    python scripts/build_dataset.py --params doe.csv --pch-dir runs/ \
        --point 35106 --component T1 --fmin 25 --fmax 10090 --df 5
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import _common  # noqa: F401
from frf_surrogate.punch import extract_frf

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--params", required=True)
p.add_argument("--id-col", default=None, help="run-id column (default: first column)")
p.add_argument("--pch-dir", required=True)
p.add_argument("--pattern", default="{id}/*.pch")
p.add_argument("--point", type=int, default=35106)
p.add_argument("--component", default="T1", help="T1=X, T2=Y, T3=Z")
p.add_argument("--quantity", default="VELOCITY")
p.add_argument("--subcase", type=int, default=None)
p.add_argument("--fmin", type=float, default=25.0)
p.add_argument("--fmax", type=float, default=10090.0)
p.add_argument("--df", type=float, default=5.0)
p.add_argument("--out", default="data/processed")
a = p.parse_args()

params = pd.read_csv(a.params)
id_col = a.id_col or params.columns[0]
params = params.set_index(id_col)
grid = np.arange(a.fmin, a.fmax + a.df / 2, a.df)

rows, amps, failed = [], [], []
for rid in params.index:
    hits = sorted(Path(a.pch_dir).glob(a.pattern.format(id=rid)))
    if not hits:
        failed.append((rid, "no punch file"))
        continue
    try:
        f, h = extract_frf(hits[0], a.point, a.component, a.quantity, a.subcase)
    except Exception as e:  # noqa: BLE001
        failed.append((rid, str(e)))
        continue
    if f.min() > grid[0] + 1e-6 or f.max() < grid[-1] - 1e-6:
        failed.append((rid, f"frequency range {f.min()}-{f.max()} Hz does not cover grid"))
        continue
    amps.append(np.interp(grid, f, np.abs(h)))
    rows.append(rid)

out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
params.loc[rows].rename_axis("design_id").to_csv(out / "params.csv")
np.save(out / "frf_amp.npy", np.stack(amps))
np.save(out / "freq.npy", grid)
print(f"{len(rows)} designs, {len(grid)} frequencies ({grid[0]:g}-{grid[-1]:g} Hz) -> {out}")
if failed:
    print(f"{len(failed)} runs skipped:")
    for rid, why in failed[:20]:
        print(f"  {rid}: {why}")
