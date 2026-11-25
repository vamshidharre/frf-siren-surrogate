# FRF-SIREN Surrogate

**Neural surrogate for NVH frequency response functions.** It predicts the full FRF of a structure from its material parameters, using a frequency-conditioned coordinate network with sine (SIREN) activations, so you don't need a new Nastran SOL 111 run for each design.

Developed during an internship in NVH Powertrain & Integration at Mercedes-Benz AG (05/2026 – 09/2026). The use case was V8 engine components.

> **About this repository.** This is a clean PyTorch reimplementation of the internship pipeline, written from the documented method. It is **not** the original internship code, which was written in TensorFlow/Keras. The Mercedes data, FE models and trained weights are not in this repo and remain Mercedes-Benz property. To check the method, run `scripts/run_pipeline.py` on your own Nastran results, or on the synthetic data set described below.

---

## Problem

| | |
|---|---|
| FE model | Nastran SOL 111 (modal frequency response), V8 engine components |
| Design space | Material parameters (Young's modulus E, density ρ, structural damping GE) of the components, about **±10 %** around nominal |
| DoE | Automated simulation data set, **Latin hypercube sampling**, built with HyperMesh, Nastran and an Altair HyperStudy template |
| Output | Frequency response function at a response node |
| Cost | About **6 h** per Nastran run |

The goal is to map the material parameters to the FRF accurately enough to compare design variants without running the FE solver.

## Method

```
Nastran SOL 111 punch files (LHS runs over the material parameters)
        │  scripts/build_dataset.py   parse .pch, extract node/DOF, resample to uniform grid
        ▼
1. Split by design variant          70 / 15 / 15 train / val / test
        │                           whole curves go to one split; no frequency-point leakage
        ▼
2. Feature screening                consensus rank of |corr|, mutual information,
        │                           permutation importance, SHAP  (train + val only)
        ▼
3. Target transform                 |H| → 20·log10 |H|  (dB), standardised
        ▼
4. Frequency encoding               f̃ ∈ [0,1],  γ(f) = [sin(2^k π f̃), cos(2^k π f̃)], k = 0…7
        ▼
5. Coordinate network               [x_screened, γ(f)]  →  SIREN  →  ŷ_dB(f)
        │                           3 residual sine blocks, width 128, small ω₀
        ▼
6. Reconstruct the FRF              query all frequencies in one batch (seconds instead of ~6 h)
        ▼
7. Evaluate on held-out designs     skill, dB MAE, R², FRAC, peak-frequency / peak-amplitude error
```

## Model study

More than **1,200 model configurations** were trained and compared on GPU (TensorFlow/Keras) during the internship:

- **MLPs** with different activation functions (GELU, SiLU, tanh, ReLU)
- **SIREN** networks with different ω₀ values
- **CNNs** with different layer widths (e.g. 32 – 64 – 32)
- A **PCA approach**: predict the PCA coefficients of the FRF and reconstruct the curve, with different numbers of components
- **Feature-screening variants**: correlation, mutual information, permutation importance and SHAP

The final model is a Fourier-encoded SIREN coordinate network.

### Why a coordinate network instead of PCA?

When a resonance shifts by a few Hz, the PCA basis needs many components to represent it, and the error counts twice, once at the old peak position and once at the new one. Making **frequency an input** turns a peak shift into a smooth change of the learned function `y(x, f)`. The network is evaluated once per frequency, batched.

### Why SIREN?

FRFs are made of oscillatory, resonant structure. Sine activations represent that directly; ReLU-family networks have to approximate it piecewise and tend to smooth high-frequency content. A small ω₀ worked best, because the Fourier encoding already supplies the high-frequency content.

## Result

- FRF prediction in **seconds instead of about 6 hours** per Nastran run (more than 90 % less time per design iteration).
- Within the DoE range (about ±10 % around nominal) the predictions matched Nastran.
- A **Streamlit app with SHAP** let the NVH team study the influence of material changes and see which parameters drive the response.

### Limitations

- The surrogate is valid only inside the DoE range. Outside it, run Nastran.
- Lightly damped resonance peaks are the hardest part: peak height and sharp anti-resonances carry most of the error.
- With a few hundred FE runs, the amount of data limits accuracy more than the architecture does.
- No Gaussian-process or gradient-boosting baselines were part of the study.

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
  make_synthetic.py      synthetic stand-in dataset (for testing)
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
       --pattern "{id}/*.pch" --point <node_id> --component T1 \
       --fmin <f_min> --fmax <f_max> --df <df> --out data/processed

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

The synthetic set is a modal sum over 85 components × (E, ρ, GE), LHS-sampled at ±10 %, where a few components drive the mode frequencies through √(E/ρ) and the damping through GE. It only exercises the code path. Its scores say nothing about the real problem.

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
