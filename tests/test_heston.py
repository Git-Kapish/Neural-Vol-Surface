import numpy as np
from scipy.integrate import quad

from nvs.pricing.bs import bs_price
from nvs.pricing.heston import heston_call, heston_cf, heston_iv, heston_otm_price


def gil_pelaez_call(theta, F, K, T):
    """Independent reference: P1/P2 Gil-Pelaez integrals with scipy.quad (original CF, no shared
    code with the Lewis/Gauss-Legendre path except the CF itself)."""
    v0, kap, vb, sg, rho = theta
    x = np.log(F / K)

    def p(j):
        def f(u):
            if j == 1:
                phi = heston_cf(u - 1j, T, v0, kap, vb, sg, rho)  # E[e^{x_T} e^{iu x_T}]
            else:
                phi = heston_cf(u, T, v0, kap, vb, sg, rho)
            return (np.exp(1j * u * x) * phi / (1j * u)).real

        return 0.5 + quad(f, 1e-10, 400, limit=800)[0] / np.pi

    return F * p(1) - K * p(2)


def test_matches_independent_quadrature():
    theta = (0.04, 1.5, 0.05, 0.5, -0.7)
    K = np.array([80.0, 95.0, 100.0, 110.0, 130.0])
    mine = heston_call(theta, 100.0, K, 0.75)
    ref = np.array([gil_pelaez_call(theta, 100.0, kk, 0.75) for kk in K])
    assert np.abs(mine - ref).max() < 1e-6


def test_fang_oosterlee_reference_price():
    # Published benchmark (Fang & Oosterlee 2008, COS method): S0=K=100, r=q=0, T=1,
    # kappa=1.5768, vbar=0.0398, sigma=0.5751, rho=-0.5711, v0=0.0175 -> call = 5.785155450.
    theta = (0.0175, 1.5768, 0.0398, 0.5751, -0.5711)
    assert abs(heston_call(theta, 100.0, 100.0, 1.0)[0] - 5.785155450) < 1e-6


def test_zero_volvol_reduces_to_black_scholes():
    v = 0.04
    theta = (v, 1.0, v, 1e-3, 0.0)
    K = np.array([85.0, 100.0, 120.0])
    h = heston_call(theta, 100.0, K, 0.5)
    b = bs_price(100.0, K, 0.5, np.sqrt(v), True)
    assert np.abs(h - b).max() < 1e-5
    iv = heston_iv(np.array(theta), np.log(K / 100.0), [0.5])[0, 0]
    assert np.abs(iv - 0.2).max() < 1e-4


def test_prices_obey_no_arbitrage_and_are_vectorised():
    thetas = np.array([(0.04, 2.0, 0.04, 0.4, -0.6), (0.09, 1.0, 0.06, 0.8, -0.3)])
    k = np.linspace(-0.2, 0.2, 9)
    T = np.array([0.05, 0.5])
    p = heston_otm_price(thetas, k, T)
    assert p.shape == (2, 2, 9) and (p > 0).all()
    one = heston_otm_price(thetas[1], k, T)[0]
    np.testing.assert_allclose(p[1], one, rtol=1e-5, atol=1e-10)  # nodes depend on batch
