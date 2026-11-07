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
