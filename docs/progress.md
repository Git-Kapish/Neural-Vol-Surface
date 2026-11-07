# Progress

## M0 - scaffold and snapshot (2026-11-07)
- Repo layout, Makefile (`uv` + Python 3.11), config, git-ignore.
- `nvs.data.fetch` saves a yfinance chain plus `meta.json` to `data/raw/YYYY-MM-DD/`; `scripts/collect_snapshot.py` is the schedulable collector.
- First snapshot: SPY, 2075 quotes, spot 735.29 (real, delayed; weekend so quotes are Friday's close).
- Track C data plan: collector + synthetic history (see `docs/data_sources.md`).

## M1 - Black-Scholes and IV inversion
- `pricing/bs.py` (forward form, OTM leg computed directly, ITM by parity), `pricing/implied_vol.py` (safeguarded Newton + bisection, bounds check before inversion, reprice check, status code per element, failure counts logged).
- Tests: round trip over k in [-0.6, 0.6], T from 1 day to 2y, vol 5%-200%, calls and puts: reprice error < 1e-8; every price carrying information (OTM leg > 1e-9 F) inverts to < 1e-6 vol; the rest are flagged, not dropped silently.
