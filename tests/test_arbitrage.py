import numpy as np

from nvs.surface.arbitrage import butterfly_violations, calendar_violations, check_surface
from nvs.surface.svi import fit_svi, svi_derivs, svi_w

K = np.linspace(-0.4, 0.4, 161)
T = np.array([0.1, 0.25, 0.5, 1.0])


def clean_surface():
    # Flat-ish smile whose total variance grows with T: arbitrage-free by construction.
    return np.array([svi_w(K, 0.01 * t, 0.1 * t**0.5 + 0.02, -0.4, 0.0, 0.2) for t in T])


def test_clean_surface_passes():
    r = check_surface(clean_surface(), K)
    assert r["butterfly_violations"] == 0 and r["calendar_violations"] == 0


def test_broken_butterfly_flagged():
    w = clean_surface()
    w[1] = w[1] - 0.05 * np.exp(-(K / 0.05) ** 2)  # narrow dip -> negative density near k = 0
    assert butterfly_violations(w, K)["count"] > 0


def test_broken_calendar_flagged():
    w = clean_surface()
    w[2] = w[1] * 0.9  # variance falls from T=0.25 to T=0.5
    assert calendar_violations(w)["count"] > 0


def test_fd_curvature_matches_analytic_svi():
    p = (0.01, 0.2, -0.4, 0.05, 0.15)
    w, w1, w2 = svi_derivs(K, *p)
    np.testing.assert_allclose(np.gradient(np.gradient(w, K), K)[2:-2], w2[2:-2], rtol=5e-3)


def test_svi_fit_recovers_smile():
    p = (0.01, 0.15, -0.5, 0.02, 0.1)
    k = np.linspace(-0.3, 0.3, 40)
    iv = np.sqrt(svi_w(k, *p) / 0.25)
    _, rmse = fit_svi(k, iv, 0.25, np.full(40, 0.01))
    assert rmse < 0.01  # vol points
