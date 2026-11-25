"""Generate a synthetic stand-in dataset for testing the pipeline.

85 components x (E, RHO, GE) = 255 parameters, LHS-sampled at +-10 %, FRFs on
25-10090 Hz (df = 5 Hz, 2014 points). The response is a modal sum in which a
handful of components drive the mode frequencies through sqrt(E/rho) and the
mode damping through GE, so screening has a real signal to find.

It is for testing the code path only. It is not the Mercedes data.

    python scripts/make_synthetic.py --n 682 --out data/synthetic
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

p = argparse.ArgumentParser()
p.add_argument("--n", type=int, default=682)
p.add_argument("--n-comp", type=int, default=85)
p.add_argument("--n-modes", type=int, default=30)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--out", default="data/synthetic")
a = p.parse_args()

rng = np.random.default_rng(a.seed)
d = 3 * a.n_comp

# Latin hypercube in [0,1]^d
u = (rng.permuted(np.tile(np.arange(a.n), (d, 1)), axis=1).T + rng.random((a.n, d))) / a.n
scale = 0.9 + 0.2 * u                     # +-10 % around nominal
E0, R0, G0 = 70e3, 2.7e-9, 0.02
cols = []
for c in range(1, a.n_comp + 1):
    cols += [f"E_{c}", f"RHO_{c}", f"GE_{c}"]
X = np.empty_like(scale)
X[:, 0::3] = E0 * scale[:, 0::3]
X[:, 1::3] = R0 * scale[:, 1::3]
X[:, 2::3] = G0 * scale[:, 2::3]

freq = np.arange(25.0, 10090.0 + 2.5, 5.0)
f0 = np.geomspace(150, 9000, a.n_modes)
active = rng.choice(a.n_comp, 8, replace=False)          # components that matter
W = rng.dirichlet(np.ones(len(active)) * 0.5, size=a.n_modes)   # modal participation
amp0 = 10 ** rng.uniform(-1, 1, a.n_modes)

ratio = np.sqrt(scale[:, 3 * active] / scale[:, 3 * active + 1])      # sqrt(E/rho) per active comp
ge = scale[:, 3 * active + 2]
fn = f0[None, :] * (ratio @ W.T)                       # (n, modes)
zeta = 0.5 * G0 * (ge @ W.T)

w = 2 * np.pi * freq[None, None, :]
wn = 2 * np.pi * fn[:, :, None]
H = (amp0[None, :, None] * 1j * w / (wn ** 2 - w ** 2 + 2j * zeta[:, :, None] * wn * w)).sum(1)
H *= 1 + 0.01 * rng.standard_normal(H.shape)           # 1 % noise

out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
pd.DataFrame(X, columns=cols, index=pd.Index(range(a.n), name="design_id")).to_csv(out / "params.csv")
np.save(out / "frf_amp.npy", np.abs(H))
np.save(out / "freq.npy", freq)
print(f"synthetic: {a.n} designs x {d} params, {len(freq)} freqs; active components "
      f"{sorted(int(i) + 1 for i in active)} -> {out}")
