# CLAUDE.md — Neural Volatility Surface Forecaster

## What this project is
Builds implied-volatility surfaces from options chains, then uses neural networks to (A) smooth surfaces with no-arbitrage penalties, (B) calibrate Heston-type models quickly via a neural pricer, and (C) forecast surface evolution. Read `PRD.md` first, especially the track split and data caveats.

## Stack
Python 3.11, numpy, scipy, pandas, PyTorch, yfinance, plotly, matplotlib, pyyaml, pytest, ruff. Ask before adding dependencies.

## Commands
```
make setup
make fetch          # download/cache a chain snapshot
make surface        # clean, invert IV, build surface, SVI baseline
make train-pricer   # generate Heston data + train theta -> IV network
make calibrate      # calibrate to the market snapshot
make smooth         # neural smoother with arbitrage penalties
make forecast       # Track C (synthetic or real history)
make viz
make test
make lint
```

## Layout
`src/nvs/`: `data/` (fetch, clean), `pricing/` (bs, implied_vol, heston), `surface/` (grid, svi, arbitrage), `nn/` (pricer, smoother, forecaster), `calibrate.py`, `viz.py`, `metrics.py`. Parameters in `configs/default.yaml`; data sources recorded in `docs/data_sources.md`.

## Non-negotiable rules
1. **Be honest about data.** Free option data is a current snapshot only. Never present forecasting results as real-data results unless real multi-day history is in `data/raw`. Synthetic-history results must be labelled "synthetic" in tables, figures, and README.
2. Use mid prices of **out-of-the-money** options, convert via put-call parity, and check no-arbitrage price bounds *before* inverting. Never silently drop failed inversions: log counts and reasons.
3. Extract the forward per expiry (put-call parity); don't assume spot × e^{rT} blindly. Work in log-moneyness `k = ln(K/F)` and total variance `w = σ²T` where convenient.
4. Verify every pricer/inverter by round trip (price → IV → price) with tolerance asserted in tests.
5. The neural pricer is validated against the **exact** pricer on held-out parameters, with errors in **vol points**, reported by region (wings, short maturities).
6. Arbitrage checks (butterfly: convexity in strike; calendar: total variance non-decreasing in T) are first-class functions with tests; report violation counts for every surface method.
7. Time series: strictly time-ordered splits, walk-forward evaluation, persistence baseline always included. No shuffling.
8. Keep a fixed training grid of (k, T) for networks; document it in the config.
9. Don't hardcode dates or tickers; use the config. Never commit raw data or trained weights larger than a few MB.
10. Be careful with units: annualised vol vs total variance, years vs days, percent vs decimals. State units in docstrings.

## Conventions
- Type hints; docstrings give shapes and units. Vectorise pricing (numpy); avoid Python loops over strikes.
- Heston pricing via characteristic function (numerically stable "little Heston trap" formulation); document the integration scheme and tolerance.
- Torch: seed everything, `float64` for pricing checks, `float32` OK for training; save model + config together.
- Plots: Plotly HTML for 3D (interactive) plus PNG export; label axes (log-moneyness, maturity in years, IV in %).
- Use `logging`; no stray `print`.

## Testing
- `test_bs.py` and `test_implied_vol.py`: round trips incl. deep ITM/OTM, short maturity.
- `test_forward.py`: parity recovers a known forward.
- `test_heston.py`: reference prices; vol-of-vol → 0 reduces to BS.
- `test_arbitrage.py`: flags a broken surface, passes a clean one.
- `test_pricer_nn.py`: held-out error below configured threshold; finite gradients.
- `test_forecast_split.py`: no leakage between features and targets.
Run `make test` and `make lint` before closing any milestone.

## Workflow
1. Milestones M0 → M8 in order. Tracks A and B come before C.
2. Phase 0: confirm the data plan with the user (collector now? paid source available? synthetic history OK?) before building Track C.
3. Show evidence after each milestone (plots, error tables, test output).
4. Ask before: switching model family (e.g. rough Bergomi), changing the grid, or changing evaluation metrics.
5. Progress notes in `docs/progress.md`.

## Definition of done
Code + tests + lint clean + progress note + the relevant figure/table saved in `results/`.

## Reporting rules
Report errors in vol points with the region breakdown; report arbitrage-violation counts for SVI vs neural; report calibration runtime vs exact calibration; always compare forecasts to persistence. State real vs synthetic data on every result. List limitations (delayed quotes, a single underlying, short history).

## Out of scope
Live trading, American-style exercise adjustments (use European approximations on index/ETF options, and note it), full production calibration engines.
