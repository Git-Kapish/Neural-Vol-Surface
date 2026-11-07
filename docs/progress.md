# Progress

## M0 - scaffold and snapshot (2026-11-07)
- Repo layout, Makefile (`uv` + Python 3.11), config, git-ignore.
- `nvs.data.fetch` saves a yfinance chain plus `meta.json` to `data/raw/YYYY-MM-DD/`; `scripts/collect_snapshot.py` is the schedulable collector.
- First snapshot: SPY, 2075 quotes, spot 735.29 (real, delayed; weekend so quotes are Friday's close).
- Track C data plan: collector + synthetic history (see `docs/data_sources.md`).

## M1 - Black-Scholes and IV inversion
- `pricing/bs.py` (forward form, OTM leg computed directly, ITM by parity), `pricing/implied_vol.py` (safeguarded Newton + bisection, bounds check before inversion, reprice check, status code per element, failure counts logged).
- Tests: round trip over k in [-0.6, 0.6], T from 1 day to 2y, vol 5%-200%, calls and puts: reprice error < 1e-8; every price carrying information (OTM leg > 1e-9 F) inverts to < 1e-6 vol; the rest are flagged, not dropped silently.

## M2 - cleaning, forward, SVI baseline, arbitrage diagnostics (real SPY snapshot 2026-11-07)
- Filter log (sequential rejects of 6141 quotes): bid<=0/ask<=bid 384, mid<$0.05 192, spread>50% 43, OI<10 1019, stale+no volume 20, kept 4483. After OTM selection 2931 quotes, 0 IV inversion failures (mid, bid, ask).
- Forward per expiry from put-call parity on the 12 pairs nearest spot. The free regression for DF was noisy (implied DF > 1), so DF = exp(-r T) with r = 4% from config and F = median(K + (C-P)/DF); regression is used only for T > 0.15y with a plausible implied rate. Documented in `extract_forward`.
- SVI per expiry (18 slices), fit RMSE 0.05-0.81 vol points.
- Arbitrage on the eval window (k in [-0.15, 0.10], T in [0.05, 0.90], 101x40): SVI 0 butterfly / 0 calendar; raw interpolation 61 butterfly / 2 calendar (`results/tables/arbitrage_baselines.csv`).
- Figures: `results/figures/surface_svi_3d.{html,png}`, `smiles_svi.png`, `term_structure_svi.png`, `residuals_svi.png`.
- Note: SPY options are American; European approximation used. Per-slice SVI is not guaranteed calendar-free in general; it happens to be here.

## M3 - Heston pricer and training-set generator
- `pricing/heston.py`: little-trap CF, Lewis single-integral price, Gauss-Legendre on [0, U]. U is probed per parameter set (the Heston CF decays like exp(-cu), not Gaussian; my first fixed-U heuristic was off by 2e-5 on the benchmark and was replaced). Parameters are sorted by U and chunked.
- Tests: Fang & Oosterlee (2008) benchmark (S=K=100, T=1, call = 5.785155450) matches to < 1e-6; matches an independent Gil-Pelaez quad to < 1e-6; vol-of-vol -> 0 reduces to BS (< 1e-5 in price, < 0.01 vol pt).
- Speed: about 6.5 ms per parameter set for the 9x6 grid on CPU. About 3% of Sobol samples are dropped because the k=+0.2, T=0.05 call is below the 1e-13 price floor (unresolvable IV); counted and logged, not silent.

## M4 - neural pricer (theta -> 6x9 IV grid), all data synthetic (exact Heston labels)
- 5-256x4-54 SiLU MLP, 47,267 train / 5,000 val / 5,000 held-out test sets (57,267 of 60,000 Sobol draws kept; 2,733 dropped because the far-OTM wing price is below the 1e-12 floor, counted in the log).
- Held-out error vs the exact pricer (vol points): mean 0.0056, p95 0.016, max 1.31. By region (mean): ATM 0.0043, wings 0.0071, short T 0.0075, long T 0.0047, wings & short T 0.0107. The worst single point (1.3 vol pts) is in a corner of the parameter box; the mean is far under a vol point, the max is not "well under", so both are reported. `results/tables/pricer_errors.csv`, `results/figures/pricer_error_grid.png`.
- Model: `results/models/pricer.pt` (0.86 MB, state + config).

## M5 - calibration to the real SPY surface (2026-11-07)
- Target: SVI-smoothed real IVs on the fixed grid, masked to where data exists (41 of 54 cells), weighted by 1/bid-ask half-width. 8 shared Sobol restarts, bounded `least_squares`.
- Result (`results/tables/calibration.csv`): both methods reach the same optimum (v0 0.0571, vbar 0.0492, sigma 1.2, rho -0.82). Grid RMSE 2.51 vs 2.50 vol points (NN vs exact), 1.56 vs 1.55 vol pts on raw quotes inside the grid's (k, T) range.
- Runtime (CPU): whole calibration 1.3 s neural vs 3.7 s exact = 2.9x; single evaluation 0.16 ms vs 2.3 ms = 14x. An earlier run showed 20x, which came from my uncached Gauss-Legendre node computation in the exact pricer; fixed with a cache before reporting. The exact pricer here is already vectorised, so the speedup is modest.
- sigma sits on its upper bound (1.2); Heston cannot reproduce the very steep short-maturity skew (fit is poor at k < -0.15, T < 0.15). Limitations, not tuned away.

## M6 - neural smoother (real SPY 2026-11-07, eval window k in [-0.15, 0.10], T in [0.05, 0.90])
| method | RMSE (vol pts) | butterfly viol. | calendar viol. |
|---|---|---|---|
| raw interpolation | 0.00 (interpolates) | 101 | 2 |
| SVI per slice | 0.42 | 0 | 0 |
| neural, no penalty | 0.34 | 0 | 0 |
| neural, arb. penalties | 0.34 | 0 | 0 |

Stress test (same quotes + synthetic 1.0 vol pt noise): raw 1921 / 237, SVI 0 / 16, neural 0 / 0 (both variants), RMSE about 1.04-1.06 (the noise level).
- The neural smoother matches or beats SVI on fit and has fewer arbitrage violations than SVI/raw under noise. The arbitrage penalties themselves add no measurable benefit over the unpenalised network on this data (the small SiLU network is already smooth enough); I am not claiming otherwise.
