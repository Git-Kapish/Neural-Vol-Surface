import numpy as np

from nvs.pricing.bs import bs_price, bs_vega


def test_put_call_parity():
    F, K, T, s, df = 100.0, np.array([70.0, 100.0, 140.0]), 0.7, 0.25, 0.97
    c = bs_price(F, K, T, s, True, df)
    p = bs_price(F, K, T, s, False, df)
    np.testing.assert_allclose(c - p, df * (F - K), atol=1e-12)


def test_known_value():
    # ATM, F=K=100, T=1, sigma=0.2, df=1: C = F*(2N(0.1)-1) = 7.9655674554...
    assert abs(bs_price(100, 100, 1.0, 0.2, True) - 7.965567455405804) < 1e-10


def test_vega_matches_finite_difference():
    F, K, T, s = 100.0, 110.0, 0.5, 0.3
    h = 1e-6
    fd = (bs_price(F, K, T, s + h) - bs_price(F, K, T, s - h)) / (2 * h)
    assert abs(fd - bs_vega(F, K, T, s)) < 1e-6
