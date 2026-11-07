# Neural Volatility Surface

Builds an implied-volatility surface from a real SPY option chain, then uses neural networks to
(A) smooth it with no-arbitrage penalties, (B) calibrate a Heston model with a neural pricer, and
(C) forecast surface evolution. See `PRD.md` for scope and `docs/progress.md` for the per-milestone log.

## Real vs synthetic data (read this first)
| Part | Data |
|---|---|
| Surface construction, SVI, neural smoother, Heston calibration | **Real**: one delayed yfinance SPY snapshot (`data/raw/<date>/`, git-ignored) |
| Neural pricer training and its error table | **Synthetic**: exact Heston prices on sampled parameters |
| Forecaster (Track C) | **Synthetic only**: latent Heston dynamics + quote noise. Free sources give no option history, so **no forecasting result here is a real-data result** |

A daily snapshot collector (`scripts/collect_snapshot.py`) is included. Schedule it if you want real
history to accumulate. Details in `docs/data_sources.md`.

## Run it
Needs `uv` and Python 3.11 (CPU only).
```
make setup          # venv + dependencies
make fetch          # snapshot -> data/raw/YYYY-MM-DD/
make surface        # clean, forwards, IVs, SVI, arbitrage diagnostics, figures
make train-pricer   # Heston training set + neural pricer (about 10 min)
make calibrate      # neural vs exact calibration to the snapshot
make smooth         # neural smoother + stress test (about 10 min)
make forecast       # synthetic-history forecaster
make test && make lint
python scripts/run_all.py [--skip-fetch] [--skip-train]
```
Figures land in `results/figures/` (interactive 3D as `*.html`, loads plotly.js from a CDN),
tables in `results/tables/`, the two small trained models in `results/models/`.
`make fetch` downloads whatever the market looks like today, so a re-run reproduces the pipeline,
not the exact numbers below (those are for the 2026-11-07 snapshot, which was a Saturday: Friday's close).

## Method in brief
- **Cleaning** (every rejection counted in the log): positive bid/ask, spread <= 50%, mid >= $0.05, open interest >= 10, stale-and-no-volume dropped. 6141 quotes -> 4483 -> 2931 OTM quotes, 0 inversion failures.
- **Forward per expiry** from put-call parity on the 12 pairs nearest spot, with DF = exp(-rT) at a 4% rate (a free-slope fit was too noisy; see `extract_forward`). Coordinates are k = ln(K/F) and total variance w = sigma^2 T.
- **IV inversion**: OTM mid, bounds checked first, safeguarded Newton with bisection fallback, repriced to verify.
- **Heston**: characteristic function (little-trap form), Lewis integral, Gauss-Legendre. Checked against the Fang-Oosterlee (2008) benchmark price and an independent quadrature.
- **Neural pricer**: MLP theta -> IV on a fixed 9 (k) x 6 (T) grid (`configs/default.yaml`), validated on held-out parameters against the exact pricer.
- **Smoother**: MLP (k, T) -> IV with autograd butterfly (Gatheral g >= 0) and calendar (dw/dT >= 0) penalties.
- **Forecaster**: persistence, AR on PCA factors, MLP on lagged factors; expanding-window walk-forward, no shuffling.

## Key results
All numbers: SPY snapshot 2026-11-07 (real) unless marked synthetic. Tables in `results/tables/`.

**Neural pricer vs exact Heston, 5,000 held-out parameter sets (synthetic labels), error in vol points**

| region | mean | p95 | max |
|---|---|---|---|
| all | 0.0055 | 0.016 | 1.32 |
| ATM (abs k <= 0.04) | 0.0044 | 0.013 | 0.49 |
| wings (abs k >= 0.14) | 0.0068 | 0.019 | 1.32 |
| short T (<= 0.10y) | 0.0072 | 0.020 | 1.32 |
| long T (>= 0.75y) | 0.0048 | 0.014 | 0.53 |
| wings and short T | 0.0097 | 0.027 | 1.32 |

The mean is far below a vol point; the worst single grid point (1.3) is in the wing/short-maturity corner.

**Calibration to the real surface** (41 of 54 grid cells covered by data; 8 shared restarts)

| | v0 | kappa | vbar | sigma | rho | grid RMSE (vol pts) | time | per evaluation |
|---|---|---|---|---|---|---|---|---|
| neural pricer | 0.0571 | 3.62 | 0.0491 | 1.2 (bound) | -0.818 | 2.505 | 1.3 s | 0.10 ms |
| exact pricer | 0.0571 | 3.61 | 0.0492 | 1.2 (bound) | -0.818 | 2.505 | 7.1 s | 2.9 ms |

Same optimum; 5.6x faster end to end, about 28x per evaluation (CPU). The exact pricer here is already vectorised, so this is a modest speedup, not orders of magnitude. The 2.5-vol-point misfit is Heston's inability to match the short-maturity skew, not an optimiser problem.

**Smoother vs SVI vs raw interpolation** (eval window k in [-0.15, 0.10], T in [0.05, 0.90], 101x40 points)

| method | RMSE (vol pts) | butterfly violations | calendar violations |
|---|---|---|---|
| raw interpolation | 0 (interpolates) | 101 | 2 |
| SVI per slice | 0.42 | 0 | 0 |
| neural, no penalty | 0.34 | 0 | 0 |
| neural, arbitrage penalties | 0.34 | 0 | 0 |

Stress test (same quotes + synthetic 1.0 vol-pt noise): raw 1921 / 237, SVI 0 / 16, both neural variants 0 / 0, RMSE about 1.04-1.06.
Honest reading: the neural smoother fits slightly better than SVI and has fewer violations under noise, but the penalty terms themselves make no measurable difference here.

**Forecast vs persistence: SYNTHETIC history only** (1,000 walk-forward test days, 0.3 vol-pt quote noise, error vs next-day observed surface)

| model | RMSE (vol pts) | ratio vs persistence |
|---|---|---|
| persistence | 0.621 | 1.00 |
| AR on PCA factors | 0.548 | 0.88 |
| MLP on PCA factors | 0.564 | 0.91 |

The gain comes mostly from filtering quote noise and exploiting mean reversion in a process I designed; it says nothing about real markets. All predicted and true surfaces had 0 butterfly/calendar violations on the 9x6 grid, so the calendar projection did nothing.

Figures: `results/figures/` (`surface_svi_3d`, `surface_heston_3d`, `surface_neural_3d`, `smiles_*`, `term_structure_*`, `residuals_*`, `arbitrage_compare*`, `pricer_error_grid`, `forecast_synthetic`).


## Limitations
- One underlying, one delayed snapshot (Saturday, Friday-close quotes); no history, so no real forecast claims.
- SPY options are American; European pricing and parity are used as an approximation.
- DF uses a fixed 4% rate; the forward absorbs dividends/borrow. A free parity-slope fit was too noisy.
- Heston cannot fit the steep short-maturity skew (k < -0.15, T < 0.15y); calibrated vol-of-vol sits on its upper bound (1.2).
- The neural pricer is trained only inside the config's parameter box and fixed grid; its worst single grid point error is larger than its mean (see table).
- The arbitrage penalties add no measurable benefit over the unpenalised network on this data (it is already arbitrage-free); the advantage shown is over SVI-per-slice and raw interpolation.
- Butterfly counts use finite differences on the evaluation grid; on the 9-point forecast grid they are only indicative.
- Rough Bergomi (PRD stretch) not attempted.

## References
Horvath, Muguruza, Tomas (2021) *Deep Learning Volatility*; Ackerer, Tagasovska, Vatter (2020)
*Deep Smoothing of the Implied Volatility Surface*; Gatheral & Jacquier (2014) *Arbitrage-free SVI*;
Heston (1993); Fang & Oosterlee (2008).
