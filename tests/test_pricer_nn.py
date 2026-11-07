import numpy as np
import pytest
import torch

from nvs.config import load_config
from nvs.nn import pricer as P
from nvs.pricing.heston import heston_iv, sample_params

pytestmark = pytest.mark.skipif(not P.MODEL_PATH.exists(), reason="run `make train-pricer` first")


def test_heldout_error_below_threshold():
    cfg = load_config()
    net = P.load()
    th = sample_params(150, cfg["heston"]["bounds"], seed=12345)  # fresh seed, never trained on
    k, T = np.array(cfg["grid"]["k"]), np.array(cfg["grid"]["T"])
    exact = heston_iv(th, k, T)
    ok = np.isfinite(exact).all(axis=(1, 2))
    err = np.abs(P.predict(net, th[ok]) - exact[ok]) * 100
    assert err.mean() < cfg["pricer"]["max_mean_err_volpts"]
    assert np.percentile(err, 99) < 0.5  # vol points
    assert err.max() < 2.0  # worst single grid point (observed worst on 5000 test sets: 1.3)


def test_gradients_wrt_theta_finite():
    net = P.load()
    th = torch.tensor([[0.04, 2.0, 0.05, 0.5, -0.6]], dtype=torch.float32, requires_grad=True)
    net(th).sum().backward()
    assert torch.isfinite(th.grad).all() and th.grad.abs().sum() > 0
