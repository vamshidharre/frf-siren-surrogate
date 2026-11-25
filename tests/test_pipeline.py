import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from frf_surrogate.data import from_db, split_designs, to_db  # noqa: E402
from frf_surrogate.encoding import FrequencyNormalizer, fourier_encode  # noqa: E402
from frf_surrogate.metrics import frac, skill  # noqa: E402
from frf_surrogate.models import ResidualSIREN, build_model, n_params  # noqa: E402
from frf_surrogate.punch import extract_frf, parse_punch  # noqa: E402
from frf_surrogate.screening import screen  # noqa: E402
from frf_surrogate.surrogate import Surrogate, train_surrogate  # noqa: E402
from frf_surrogate.templex import mat1_to_templex  # noqa: E402

PCH = """$TITLE   = EATS2.0 D1-MUSTER MIT EM-XL-ANREGNGSDATEN                         1
$SUBTITLE= FREQUENCY RESPONSE SIMULATION                                    2
$LABEL   = ANREGUNG 48MO                                                    3
$VELOCITY                                                                   4
$MAGNITUDE-PHASE OUTPUT                                                     5
$SUBCASE ID =       10048                                                   6
$POINT ID =       35106                                                     7
    1.000000E+01 G      5.850444E+00      4.564463E-01      3.187709E+01       8
-CONT-                  2.234876E-03      1.186267E-01      1.586428E-03       9
-CONT-                  8.482604E+01      8.365439E+01      2.845480E+02      10
-CONT-                  2.886703E+02      2.847940E+02      2.847940E+02      11
    1.500000E+01 G      6.000000E+00      4.000000E-01      3.000000E+01      12
-CONT-                  2.000000E-03      1.000000E-01      1.000000E-03      13
-CONT-                  9.000000E+01      8.000000E+01      2.800000E+02      14
-CONT-                  2.800000E+02      2.800000E+02      2.800000E+02      15
"""


def test_punch_parser(tmp_path):
    p = tmp_path / "a.pch"
    p.write_text(PCH)
    blocks = parse_punch(p)
    assert len(blocks) == 1 and blocks[0].point == 35106 and blocks[0].quantity == "VELOCITY"
    f, h = extract_frf(p, 35106, "T1")
    np.testing.assert_allclose(f, [10.0, 15.0])
    np.testing.assert_allclose(np.abs(h), [5.850444, 6.0], rtol=1e-6)
    np.testing.assert_allclose(np.angle(h, deg=True)[0], 84.82604, rtol=1e-5)


def test_db_roundtrip_and_split():
    a = np.array([1e-3, 1.0, 10.0])
    np.testing.assert_allclose(from_db(to_db(a)), a)
    tr, va, te = split_designs(682, seed=42)
    assert (len(tr), len(va), len(te)) == (477, 102, 103)
    assert not set(tr) & set(te) and not set(va) & set(te)


def test_encoding():
    fn = FrequencyNormalizer(25, 10090, "log")
    assert np.isclose(fn(25), 0) and np.isclose(fn(10090), 1)
    e = fourier_encode(fn(np.array([25.0, 500.0])), 8)
    assert e.shape == (2, 16)
    et = fourier_encode(torch.tensor([0.0, 0.5]), 8)
    assert et.shape == (2, 16)


def test_final_model_size():
    m = ResidualSIREN(10 + 16, width=128, n_blocks=3, omega0=2.0)
    assert 100_000 < n_params(m) < 106_000          # ~103k as reported
    assert m(torch.randn(5, 26)).shape == (5,)
    assert build_model({"type": "gelu", "depth": 14, "width": 192}, 26)(torch.randn(3, 26)).shape == (3,)


def test_metrics():
    y = np.random.default_rng(0).normal(size=(4, 50))
    base = y.mean(0)
    assert skill(y, y, base) == pytest.approx(1.0)
    assert skill(y, np.tile(base, (4, 1)), base) == pytest.approx(0.0)
    assert np.allclose(frac(y, y), 1.0)


def test_templex():
    cards = ["MAT1    1       210000.         .3      7.85E-9         0.02",
             "PSOLID  1       1"]
    out, v = mat1_to_templex(cards)
    assert v == ["E_1", "RHO_1", "GE_1"]
    assert '{E_1,"%8.0f"}' in out[0] and '{GE_1,"%8.3f"}' in out[0] and out[1] == cards[1]


def test_screen_and_train_smoke(tmp_path):
    rng = np.random.default_rng(0)
    n, p, nf = 60, 12, 40
    X = pd.DataFrame(rng.uniform(0.6, 1.4, (n, p)), columns=[f"P{i}" for i in range(p)])
    freq = np.linspace(25, 2000, nf)
    fr = 500 * np.sqrt(X["P3"].values / X["P4"].values)
    Y = -20 * np.log10(np.abs(fr[:, None] ** 2 - freq[None] ** 2) + 1e3)
    feats, table = screen(X, Y, freq, top_k=3, use_shap=False, n_jobs=1)
    assert {"P3", "P4"} <= set(feats)
    tr, va = np.arange(40), np.arange(40, 50)
    sur = train_surrogate(X[feats].values[tr], Y[tr], X[feats].values[va], Y[va], freq, feats,
                          {"type": "siren", "width": 32, "n_blocks": 1, "omega0": 2.0},
                          {"epochs": 3, "batch_size": 256}, {"n_bands": 4}, verbose=False)
    pred = sur.predict_db(X[feats].values[50:], freq)
    assert pred.shape == (10, nf) and np.isfinite(pred).all()
    sur.save(tmp_path / "m.pt")
    np.testing.assert_allclose(Surrogate.load(tmp_path / "m.pt").predict_db(X[feats].values[50:], freq), pred,
                               rtol=1e-5, atol=1e-4)
