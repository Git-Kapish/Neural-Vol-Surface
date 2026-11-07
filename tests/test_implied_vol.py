import numpy as np
import pytest

from nvs.pricing.bs import bs_otm_price, bs_price
from nvs.pricing.implied_vol import (
    ABOVE_UPPER,
    BAD_INPUT,
    BELOW_INTRINSIC,
    OK,
    implied_vol,
)

K_GRID = np.exp(np.linspace(-0.6, 0.6, 25))  # k = ln(K/F), F = 1 (deep ITM/OTM included)
T_GRID = np.array([1 / 365, 0.02, 0.1, 0.5, 2.0])
S_GRID = np.array([0.05, 0.1, 0.2, 0.5, 1.0, 2.0])


@pytest.mark.parametrize("is_call", [True, False])
def test_round_trip_grid(is_call):
    K, T, S = np.meshgrid(K_GRID, T_GRID, S_GRID, indexing="ij")
    df = 0.98
    p = bs_price(1.0, K, T, S, is_call, df)
    res = implied_vol(p, 1.0, K, T, is_call, df)
    ok = res.status == OK
    # Price round trip is exact wherever the inversion succeeded.
    p2 = bs_price(1.0, K, T, np.where(ok, res.iv, 0.2), is_call, df)
    assert np.abs(p2 - p)[ok].max() < 1e-8
    # Any price carrying information (OTM leg > 1e-9 F) must be inverted exactly...
    otm = bs_otm_price(1.0, K, T, S, True)
    informative = otm > 1e-9
    assert ok[informative].all()
    assert np.abs(res.iv - S)[informative].max() < 1e-6
    # ...and every failure is flagged (not silent) and only for unresolvable tiny prices.
    assert (otm[~ok] <= 1e-9).all()


def test_bounds_flagged():
    F, K, T = 100.0, 100.0, 1.0
    r = implied_vol(np.array([-1.0, 0.0, 101.0, np.nan]), F, K, T, True)
    assert r.status[0] == BAD_INPUT and r.status[3] == BAD_INPUT
    assert r.status[1] == BELOW_INTRINSIC
    assert r.status[2] == ABOVE_UPPER
    # ITM call below intrinsic: call 85 for K=80 (intrinsic 20)
    assert implied_vol(15.0, 100.0, 80.0, 1.0, True).status == BELOW_INTRINSIC


def test_itm_call_matches_otm_put_vol():
    # Same strike: ITM call (via parity) and OTM put must give the same IV.
    F, K, T, s, df = 100.0, 85.0, 0.5, 0.27, 0.99
    c = bs_price(F, K, T, s, True, df)
    p = bs_price(F, K, T, s, False, df)
    assert abs(implied_vol(c, F, K, T, True, df).iv - s) < 1e-9
    assert abs(implied_vol(p, F, K, T, False, df).iv - s) < 1e-9
