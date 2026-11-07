# PRD — Neural Volatility Surface Forecaster (ML + Quant Finance, Project 4)

**Owner:** Kapish · **Status:** Draft v1 · **Build tool:** Claude Code

## 1. One-line summary
Build an implied-volatility surface from options-chain data (invert Black-Scholes across strikes and expiries), then use neural networks to (A) calibrate a stochastic-volatility model orders of magnitude faster than classical calibration, and/or (B) produce smooth, arbitrage-aware surfaces and forecast how the surface evolves day to day, with 3D surface, skew, and term-structure visualisations.

## 2. Why this project
- Covers derivatives pricing, numerical methods (IV inversion, characteristic-function pricing), and deep learning (surrogate pricing networks, smoothing with no-arbitrage constraints).
- Produces strong visuals (3D surfaces, skew and term-structure slices).
- Honest scoping matters: **free sources give only the current option chain, not history.** The PRD therefore splits the work into tracks that are achievable without paid data.

## 3. Tracks
| Track | What | Data need | Priority |
|---|---|---|---|
| **A. Surface construction + smoothing** | Clean chain → IV → surface; baseline SVI fit; neural smoother (Ackerer et al.) | One or more snapshots | Must |
| **B. Neural calibration** | Train a network to approximate a model's price→IV map (Heston, optionally rough Bergomi), then calibrate model parameters to a real snapshot | Synthetic training data + a snapshot | Must |
| **C. Day-to-day forecasting** | Predict next-day surface from today's surface | Needs a **history** of surfaces | Should (see §4) |

## 4. Data
- **Snapshots (free):** options chain for a liquid underlying (SPY, optionally QQQ/AAPL) via `yfinance`. Quotes are delayed/noisy; capture bid, ask, last, volume, open interest, timestamp. Save raw snapshots to `data/raw/YYYY-MM-DD/`.
- **Rates / dividends:** risk-free rate from a treasury yield proxy; dividend yield estimate; better, derive the forward from put-call parity on near-the-money pairs for each expiry.
- **History for Track C (choose in Phase 0, document in `docs/data_sources.md`):**
  1. Start a **daily snapshot collector** (scheduled job) now; usable after a few weeks (small sample, say so).
  2. A paid or academic historical source (e.g. a vendor's end-of-day option data), if the user has access.
  3. **Synthetic history**: simulate a Heston / rough-vol path, price full surfaces daily, add quote noise. This is valid for developing and validating the forecaster, but results must be labelled "synthetic".
- Never claim forecasting performance on real data unless real multi-day history exists.

## 5. Methodology

### 5.1 Surface construction (Track A)
1. Filter quotes: positive bid/ask, spread limits, minimum volume/OI, drop stale quotes, use mid prices.
2. Use **out-of-the-money options** (calls above forward, puts below) and convert to a consistent call-equivalent price via put-call parity.
3. Check price bounds (no-arbitrage intrinsic bounds) before inversion.
4. **Implied vol inversion:** vectorised Newton/Brent with a robust fallback; verify by repricing.
5. Coordinates: log-moneyness `k = ln(K/F)` and maturity `T` (years).
6. Baseline fit: per-expiry **SVI** slices; diagnostics for butterfly (convexity in strike) and calendar (total variance non-decreasing in T) arbitrage.

### 5.2 Neural smoother (Track A)
Network maps `(k, T) → total variance` (or IV). Add penalties for butterfly and calendar arbitrage (finite differences / autograd), per Ackerer, Tagasovska & Vatter. Compare against SVI and a raw interpolation on: fit RMSE (in vol points, bid-ask-weighted), and arbitrage-violation counts.

### 5.3 Neural calibration (Track B), two-step approach from *Deep Learning Volatility*
1. **Pricing/IV map:** generate a large training set from the model (Heston first, via a characteristic-function pricer with FFT/COS or quadrature): sample parameters θ, compute implied vols on a fixed grid of (k, T). Train a network `θ → IV grid`. Validate vs the exact pricer on held-out parameters (max/mean abs error in vol points; target well under a vol point).
2. **Calibration:** with the trained network fixed, fit θ to the market surface by optimisation (Levenberg-Marquardt or Adam with bounds, multiple restarts); weight by vega or bid-ask width. Compare speed and error against calibrating with the exact pricer.
3. Stretch: rough Bergomi (needs Monte Carlo for training labels; heavier).

### 5.4 Forecasting (Track C)
- Represent each day's surface on a **fixed grid** (e.g. 9 moneyness × 6 maturities) or via low-dimensional factors (PCA of surface changes, or model parameters θ_t from calibration).
- Targets: next-day grid / factor changes. Models: persistence baseline, AR on PCA factors, and a neural net (MLP/LSTM). Time-ordered splits only; walk-forward evaluation.
- Metrics: RMSE in vol points vs persistence (the hard baseline), plus a check that predicted surfaces remain arbitrage-free or are projected back.
- Results on synthetic history are labelled as such.

## 6. Visualisation requirements
- 3D IV surface (interactive Plotly + static PNG), model vs market overlay.
- Smile/skew slices for several expiries; ATM term structure; residual heatmap (market − model).
- Before/after: SVI vs neural smoother with arbitrage violations highlighted.
- Calibration: parameter table, fit error, runtime comparison.

## 7. Repo structure
```
vol-surface/
├── CLAUDE.md
├── README.md
├── requirements.txt
├── Makefile                      # make fetch | surface | train-pricer | calibrate | smooth | forecast | viz | test
├── configs/default.yaml
├── data/{raw,processed,synthetic}/   # git-ignored
├── docs/data_sources.md
├── src/nvs/
│   ├── data/{fetch.py,clean.py}            # chain download, filtering, forward/rate
│   ├── pricing/{bs.py,implied_vol.py,heston.py}
│   ├── surface/{grid.py,svi.py,arbitrage.py}
│   ├── nn/{pricer.py,smoother.py,forecaster.py}   # PyTorch
│   ├── calibrate.py
│   ├── viz.py
│   └── metrics.py
├── scripts/{collect_snapshot.py,run_all.py}
├── notebooks/
├── results/{figures,tables,models}/
└── tests/
```

## 8. Milestones
| # | Milestone | Done when |
|---|---|---|
| M0 | Scaffold, data-source decision, snapshot collector | one clean chain saved, collector script runs |
| M1 | BS pricing + IV inversion | reprice error < 1e-8 on synthetic; handles bounds/edge cases |
| M2 | Cleaning + surface + SVI baseline + 3D plot | surface plotted, arbitrage diagnostics reported |
| M3 | Heston pricer + training-set generator | exact pricer matches known reference values |
| M4 | Neural pricer (θ → IV grid) | held-out error table in vol points |
| M5 | Calibration to real snapshot | fitted θ, surface overlay, speedup vs exact calibration |
| M6 | Neural smoother with arbitrage penalties | beats/matches SVI on arbitrage counts, comparable RMSE |
| M7 | Forecaster (synthetic or real history) | walk-forward table vs persistence |
| M8 | README, figures, writeup | reproducible from clean clone |

## 9. Testing requirements
- BS price/IV round trip across a grid of (k, T, σ), including deep ITM/OTM and short maturities.
- Put-call parity forward extraction recovers a known forward.
- Heston pricer vs published reference prices; the flat-vol limit (vol-of-vol → 0) reduces to BS.
- Arbitrage checkers flag a deliberately broken surface and pass a known clean one.
- Neural pricer held-out error below threshold; gradients w.r.t. θ finite.
- Forecaster: no leakage (targets are strictly after features; time-ordered splits).

## 10. Deliverables
1. Repo + README (what, data, how to run, key results, limitations).
2. Figures from §6, plus a results table (pricer error, calibration error/speedup, smoother vs SVI, forecaster vs persistence).
3. A clear statement of which parts used real vs synthetic data.

## 11. Risks and pitfalls
| Risk | Mitigation |
|---|---|
| No free historical option data | Tracks A/B need only snapshots; Track C uses collector or synthetic history, clearly labelled |
| Noisy/stale quotes break IV inversion | Filtering, OTM-only, bounds checks, robust solver |
| Wrong forward/dividends distort the skew | Extract forward from put-call parity per expiry |
| Neural net violates no-arbitrage | Penalty terms + arbitrage checkers; report violations |
| Poor accuracy at wings/short maturities | Report error by region; weight by vega |
| Overclaiming forecast skill | Persistence baseline is hard to beat; walk-forward only; report honestly |

## 12. Success criteria
- Cleaned real surface with documented filtering and arbitrage diagnostics.
- Neural pricer accuracy quantified; calibration works and is demonstrably faster than exact calibration.
- Smoother has fewer arbitrage violations than the raw/SVI baseline at similar fit error.
- Forecasting result reported against persistence, labelled real or synthetic.

## 13. References (from the source post)
- Horvath, Muguruza, Tomas (2021), *Deep Learning Volatility*, Quantitative Finance.
- Ackerer, Tagasovska, Vatter (2020), *Deep Smoothing of the Implied Volatility Surface*, NeurIPS.
- Muguruza et al., *NN-StochVol-Calibrations* (GitHub).
- PyQuant News, *Build an Implied Volatility Surface with Python*.
- Also useful: Gatheral & Jacquier (2014), *Arbitrage-free SVI volatility surfaces*; Heston (1993).
