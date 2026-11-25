# FRF-SIREN Surrogate

**Neural surrogate for NVH frequency response functions.** It predicts the full FRF of a structure from its material parameters, using a frequency-conditioned coordinate network with sine (SIREN) activations, so you don't need a new Nastran SOL 111 run for each design.

Developed during an internship in NVH Powertrain & Integration at Mercedes-Benz. The use case was an electric drive unit (EDU) housing, with the FRF measured at one response node.

> **About this repository.** This is a clean reimplementation of the internship pipeline, written from the documented method. It is **not** the original internship code. The results table below gives the figures reported from the internship work. This code has not reproduced them, because the Mercedes data is not in this repo. To verify, run `scripts/run_pipeline.py` on the original dataset. No proprietary data, FE models or trained weights are committed.

---

## Problem

| | |
|---|---|
| FE model | Nastran SOL 111 (modal frequency response). 7.11 M nodes, 4.33 M elements, ~21.3 M DOF, reduced by AMLS to 8,253 modal DOF |
| Design space | 85 components × (E, ρ, GE) = **255 material parameters**, ±40 % around nominal |
| DoE | **682 runs**, Latin hypercube, parameterised through a HyperStudy Templex template |
| Output | Velocity FRF at one node, X direction, **25 – 10,090 Hz**, Δf = 5 Hz (**2,014 points**) |
| Cost | ~1.75 h per Nastran run |

The goal is to map 255 material parameters to a 2,014-point FRF accurately enough to compare design variants without running the FE solver.

## Method

```
Nastran SOL 111 punch files (682 LHS runs, 255 params)
        │  scripts/build_dataset.py   parse .pch, extract node/DOF, resample to uniform grid
        ▼
1. Split by design variant          70 / 15 / 15  →  477 / 102 / 103 designs (seed 42)
        │                           whole curves go to one split; no frequency-point leakage
        ▼
2. Feature screening 255 → 10       consensus rank of |corr|, mutual information,
        │                           permutation importance, SHAP  (train + val only)
        ▼
3. Target transform                 |H| → 20·log10 |H|  (dB), standardised
        ▼
4. Frequency encoding               f̃ ∈ [0,1],  γ(f) = [sin(2^k π f̃), cos(2^k π f̃)], k = 0…7
        ▼
5. Coordinate network               [x_screened (10), γ(f) (16)]  →  SIREN  →  ŷ_dB(f)
        │                           ω₀ = 2, width 128, 3 residual sine blocks, ~103 k params
        ▼
6. Reconstruct the FRF              query all 2,014 frequencies in one batch (~ms)
        ▼
7. Evaluate on 103 held-out designs skill, dB MAE, R², FRAC, peak-frequency / peak-amplitude error
```

### Why a coordinate network?

An earlier version predicted 20 PCA coefficients of the FRF and reconstructed the curve with inverse PCA. That has a weakness: when a resonance shifts by a few Hz, the PCA basis needs many components to represent it, and the error counts twice, once at the old peak position and once at the new one. Making **frequency an input** turns a peak shift into a smooth change of the learned function `y(x, f)`. The network is evaluated once per frequency, batched.

### Why SIREN?

FRFs are made of oscillatory, resonant structure. Sine activations represent that directly. ReLU-family networks have to approximate it piecewise. At matched accuracy the SIREN needed roughly 4.5× fewer parameters than a deep GELU MLP. ω₀ is much lower than the original paper's 30 because the Fourier encoding already supplies the high-frequency content. Larger ω₀ made results worse.

## Reported internship results

Test set: 103 unseen design variants. **Skill** = 1 − MSE_model / MSE_baseline in dB, where the baseline is the per-frequency mean of the training curves (0 = no better than the mean curve).

| Model | Test skill | Test dB MAE | Params |
|---|---|---|---|
| **SIREN, ω₀ = 2 (final)** | **+0.335** | **2.96 dB** | ~103 k |
| SIREN, ω₀ = 1 | ≈ +0.320 | – | ~103 k |
| Deep GELU MLP (14 layers) | +0.297 | – | ~473 k |
| SiLU MLP | < +0.297 | – | – |
| tanh MLP | ≈ +0.27 | – | – |

- ω₀ sweep {1, 2, 5, 10, 30}: values above 2 were worse.
- Inference: ~3 ms for a full 2,014-point FRF, against ~1.75 h per Nastran run.

### Limitations

- **The accuracy is modest.** 2.96 dB MAE is about 40 % mean amplitude error. The surrogate is suited to *ranking and screening* design variants, not to predicting absolute levels.
- **Resonance peaks are under-resolved.** Lightly damped peaks are smoothed, so peak height and sharp anti-resonances carry most of the error.
- **The model overfits.** A tuning run showed R² of 0.88 on train and 0.38 on test. With 477 training designs, the bottleneck is data more than architecture.
- **The comparison only varies the activation function.** No Gaussian-process or gradient-boosting baselines were run, and the gaps between the top variants are small (≈ 0.04 skill).
- **Only 10 of 255 parameters are used.** Parameters that matter only through interactions may have been screened out.
- **The pipeline was tested on one response node, one DOF and one assembly.**

## Repository layout

```
configs/
  default.yaml           final configuration (SIREN ω₀=2)
  baselines.yaml         activation / ω₀ study
src/frf_surrogate/
  punch.py               Nastran .pch parser (magnitude-phase & real-imag)
  templex.py             MAT1 cards → HyperStudy Templex .tpl (E, RHO, GE per material)
  data.py                dataset loading, dB transform, design-level split
  screening.py           consensus feature screening
  encoding.py            log/linear frequency normalisation + octave Fourier encoding
  models.py              ResidualSIREN, CoordinateMLP (GELU/SiLU/tanh/ReLU)
  surrogate.py           training loop (AdamW, cosine LR, early stopping), save/load, inference
  metrics.py             skill, dB MAE, R², FRAC, peak errors
scripts/
  make_tpl.py            build the .tpl for HyperStudy
  build_dataset.py       punch files + DoE table → data/processed/
  make_synthetic.py      synthetic stand-in dataset with the same shape (for testing)
  run_pipeline.py        screening → training → evaluation → plots
  compare_models.py      multi-seed architecture comparison
  plot_results.py        loss curve, example FRFs, error by frequency
tests/                   unit tests (parser, encoding, model size, metrics, end-to-end smoke)
```

## Usage

```bash
pip install -r requirements.txt        # add `shap` to enable the SHAP screening term

# 0) DoE setup: parameterise the material cards for HyperStudy
python scripts/make_tpl.py mat1_cards.txt MAT1_template.tpl

# 1) After the DoE: collect punch files into the training layout
python scripts/build_dataset.py --params doe_table.csv --pch-dir hst_runs/ \
       --pattern "{id}/*.pch" --point 35106 --component T1 \
       --fmin 25 --fmax 10090 --df 5 --out data/processed

# 2) Screening + training + test evaluation
python scripts/run_pipeline.py --config configs/default.yaml

# 3) Optional: activation / ω₀ study over seeds
python scripts/compare_models.py --variants configs/baselines.yaml
```

Each run writes `model.pt`, `screening.csv`, `split.json`, `metrics.json`, `test_predictions.npz`, `loss.png`, `test_examples.png` and `error_by_frequency.png` to `out_dir`.

**Try it without the real data:**

```bash
python scripts/make_synthetic.py --out data/synthetic
python scripts/run_pipeline.py --data-dir data/synthetic --out-dir runs/demo --epochs 20
pytest -q
```

The synthetic set is a modal sum with 255 parameters, where 8 components drive the mode frequencies through √(E/ρ) and the damping through GE. It only exercises the code path. Its scores say nothing about the real problem.

### Using a trained model

```python
from frf_surrogate.surrogate import Surrogate
sur = Surrogate.load("runs/siren_w2/model.pt")
y_db = sur.predict_db(X_new[sur.features].values, freq)   # (n_designs, n_freq) in dB
```

## Next steps

- Gaussian-process (ARD) and gradient-boosting baselines on the same split
- Peak-aware validation (FRAC, peak-frequency/amplitude error) as headline metrics
- Learning curve (error vs. number of FE runs) to justify the DoE size
- Physics-informed variants: modal / pole-residue targets, reduced-order residual loss near resonances
- Parametric model order reduction as the classical benchmark

## Author

**Vamshidhar Reddy Enugala** · M.Sc. Computational Engineering (FAU Erlangen-Nürnberg)

Private repository. Internship work with Mercedes-Benz NVH Powertrain & Integration; data and FE models remain Mercedes-Benz property.
