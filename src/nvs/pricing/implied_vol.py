"""Vectorised Black-Scholes implied vol: safeguarded Newton (bisection fallback) + repricing check.

Units: prices in currency units (discounted, as quoted), F/K currency, T years,
returned iv is annualised decimal vol. Failures are never silent: each element gets a status
code and `implied_vol` logs counts per reason.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from nvs.pricing.bs import bs_otm_price, bs_vega

log = logging.getLogger(__name__)

OK, BELOW_INTRINSIC, ABOVE_UPPER, BAD_INPUT, NO_CONVERGE, TOO_SMALL = range(6)
STATUS_NAMES = {
    OK: "ok",
    BELOW_INTRINSIC: "price below intrinsic/OTM floor",
    ABOVE_UPPER: "price above no-arbitrage upper bound",
    BAD_INPUT: "non-finite or non-positive input",
    NO_CONVERGE: "solver/repricing failed",
    TOO_SMALL: "OTM price below resolution (1e-12 F)",
}


@dataclass
class IVResult:
    iv: np.ndarray  # decimal vol, NaN where status != OK
    status: np.ndarray  # int codes, see STATUS_NAMES

    def counts(self) -> dict[str, int]:
        return {STATUS_NAMES[c]: int((self.status == c).sum()) for c in STATUS_NAMES}


def implied_vol(price, F, K, T, is_call=True, df=1.0, *, tol=1e-12, reprice_tol=1e-8,
                max_iter=200, sigma_max=5.0, label: str = "") -> IVResult:
    """Invert BS. `price` is the discounted quote. Returns IVResult (same shape as inputs)."""
    price, F, K, T, is_call, df = np.broadcast_arrays(
        *[np.asarray(x, dtype=np.float64) for x in (price, F, K, T, is_call, df)]
    )
    shape = price.shape
    price, F, K, T, df = (a.ravel().copy() for a in (price, F, K, T, df))
    is_call = is_call.ravel().astype(bool)
    n = price.size
    status = np.full(n, OK, dtype=int)
    iv = np.full(n, np.nan)

    bad = ~(np.isfinite(price) & np.isfinite(F) & np.isfinite(K) & np.isfinite(T) & np.isfinite(df))
    bad |= (F <= 0) | (K <= 0) | (T <= 0) | (df <= 0) | (price < 0)
    status[bad] = BAD_INPUT
    ok = ~bad

    # Undiscounted price of the OTM leg, via parity (C - P = F - K).
    pu = np.where(ok, price / np.where(ok, df, 1.0), 0.0)
    call_leg = K >= F
    # ITM quote (call with K<F, put with K>=F): subtract |F-K| to get the OTM leg by parity
    target = np.where(is_call == call_leg, pu, pu - np.abs(F - K))
    upper = np.where(call_leg, F, K)
    # tiny slack for rounding in the parity conversion
    status[ok & (target < -1e-12 * F)] = BELOW_INTRINSIC
    status[ok & (target <= 0) & (status == OK)] = BELOW_INTRINSIC
    status[(status == OK) & (target >= upper)] = ABOVE_UPPER
    status[(status == OK) & (target < 1e-12 * F)] = TOO_SMALL

    idx = np.flatnonzero(status == OK)
    if idx.size:
        Fi, Ki, Ti, ti = F[idx], K[idx], T[idx], target[idx]
        lo = np.full(idx.size, 1e-8)
        hi = np.full(idx.size, sigma_max)
        s = np.clip(np.sqrt(2 * np.abs(np.log(Fi / Ki)) / Ti), 0.1, 2.0)
        done = np.zeros(idx.size, dtype=bool)
        for _ in range(max_iter):
            f = bs_otm_price(Fi, Ki, Ti, s, True) - ti
            hi = np.where(f > 0, s, hi)
            lo = np.where(f <= 0, s, lo)
            done |= (np.abs(f) <= tol * ti) | ((hi - lo) < 1e-14)
            if done.all():
                break
            vega = bs_vega(Fi, Ki, Ti, s)
            with np.errstate(divide="ignore", invalid="ignore"):
                s_new = s - f / vega
            fallback = ~np.isfinite(s_new) | (s_new <= lo) | (s_new >= hi)
            s_new = np.where(fallback, 0.5 * (lo + hi), s_new)
            s = np.where(done, s, s_new)
        err = np.abs(bs_otm_price(Fi, Ki, Ti, s, True) - ti) / Fi  # reprice check
        good = done & (err <= reprice_tol)
        iv[idx[good]] = s[good]
        status[idx[~good]] = NO_CONVERGE

    res = IVResult(iv.reshape(shape), status.reshape(shape))
    bad_counts = {k: v for k, v in res.counts().items() if v and k != STATUS_NAMES[OK]}
    log.info("IV inversion%s: %d ok / %d total; failures: %s", f" [{label}]" if label else "",
             int((status == OK).sum()), n, bad_counts or "none")
    return res
