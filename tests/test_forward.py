import numpy as np

from nvs.data.clean import extract_forward
from nvs.pricing.bs import bs_price


def test_parity_recovers_known_forward():
    F, DF, T = 412.3, 0.985, 0.4
    K = np.arange(380.0, 445.0, 5.0)
    c = bs_price(F, K, T, 0.2, True, DF)
    p = bs_price(F, K, T, 0.2, False, DF)
    Fh, DFh, method, n = extract_forward(K, c, p, spot=410.0, T=T, r_fallback=0.04, band=0.2)
    assert method == "parity_fit"
    assert abs(Fh - F) < 1e-8 and abs(DFh - DF) < 1e-10


def test_forward_robust_to_noise():
    rng = np.random.default_rng(0)
    F, DF, T = 100.0, 0.99, 0.3
    K = np.arange(95.0, 106.0)
    c = bs_price(F, K, T, 0.2, True, DF) + rng.normal(0, 0.01, K.size)
    p = bs_price(F, K, T, 0.2, False, DF) + rng.normal(0, 0.01, K.size)
    Fh, _, _, _ = extract_forward(K, c, p, spot=100.0, T=T, r_fallback=0.04)
    assert abs(Fh - F) < 0.1


def test_fallback_when_too_few_pairs():
    _, _, method, _ = extract_forward([100.0], [3.0], [2.5], spot=100.0, T=0.5, r_fallback=0.04)
    assert method == "parity_median"
