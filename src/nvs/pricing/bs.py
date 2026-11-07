"""Black-Scholes in forward form (European options).

Units: F, K in the same currency units; T in years; sigma annualised vol as a decimal
(0.2 = 20%); df is the discount factor to expiry. All prices are `df * undiscounted`.
"""
from __future__ import annotations

import numpy as np
from scipy.special import ndtr


def _prep(F, K, T, sigma):
    return [np.asarray(x, dtype=np.float64) for x in np.broadcast_arrays(F, K, T, sigma)]


def bs_otm_price(F, K, T, sigma, is_call):
    """Undiscounted price of the OTM leg (call if K >= F else put), numerically stable.

    `is_call` is ignored for the leg choice; kept so callers can request a specific leg via
    `bs_price`. Returns array shaped like the broadcast inputs.
    """
    F, K, T, sigma = _prep(F, K, T, sigma)
    sd = sigma * np.sqrt(T)
    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = (np.log(F / K) + 0.5 * sd**2) / sd
        d2 = d1 - sd
        call = F * ndtr(d1) - K * ndtr(d2)
        put = K * ndtr(-d2) - F * ndtr(-d1)
    otm = np.where(K >= F, call, put)
    return np.where(sd > 0, otm, 0.0)


def bs_price(F, K, T, sigma, is_call=True, df=1.0):
    """Discounted European price. ITM leg obtained from the OTM leg by put-call parity."""
    F, K, T, sigma = _prep(F, K, T, sigma)
    otm = bs_otm_price(F, K, T, sigma, True)
    is_call = np.asarray(is_call, dtype=bool)
    # parity: C - P = F - K
    call = np.where(K >= F, otm, otm + (F - K))
    put = np.where(K >= F, otm + (K - F), otm)
    return df * np.where(is_call, call, put)


def bs_vega(F, K, T, sigma, df=1.0):
    """d price / d sigma (per 1.00 = 100 vol points), discounted."""
    F, K, T, sigma = _prep(F, K, T, sigma)
    sd = sigma * np.sqrt(T)
    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = (np.log(F / K) + 0.5 * sd**2) / sd
    return df * F * np.exp(-0.5 * d1**2) / np.sqrt(2 * np.pi) * np.sqrt(T)
