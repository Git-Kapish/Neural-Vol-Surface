# Progress

## M0 - scaffold and snapshot (2026-11-07)
- Repo layout, Makefile (`uv` + Python 3.11), config, git-ignore.
- `nvs.data.fetch` saves a yfinance chain plus `meta.json` to `data/raw/YYYY-MM-DD/`; `scripts/collect_snapshot.py` is the schedulable collector.
- First snapshot: SPY, 2075 quotes, spot 735.29 (real, delayed; weekend so quotes are Friday's close).
- Track C data plan: collector + synthetic history (see `docs/data_sources.md`).
