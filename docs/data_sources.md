# Data sources

## Real option chains (Tracks A, B)
- Source: `yfinance` (Yahoo Finance), delayed and noisy quotes. Snapshot only, no history.
- Underlying and filters come from `configs/default.yaml`.
- Snapshots are saved to `data/raw/YYYY-MM-DD/` (git-ignored) with a `meta.json` holding spot and capture time (UTC).
- SPY options are American-style. We use European approximations and note it in results.
- Rates and dividends: the forward per expiry is extracted from put-call parity on near-the-money pairs, not from spot × e^{rT}.

## History for Track C (decision recorded in Phase 0)
- **Chosen:** (1) a daily snapshot collector (`scripts/collect_snapshot.py`), to be scheduled by the owner; real history is usable only after several weeks and will be a small sample. (3) **Synthetic history** from simulated Heston paths, with full surfaces priced daily plus quote noise, for developing and validating the forecaster.
- No paid or academic historical source is available.
- Rule: forecasting results are labelled **synthetic** everywhere unless multi-day real history exists in `data/raw`.
