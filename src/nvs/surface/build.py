"""`make surface`: clean snapshot -> OTM IVs -> SVI per expiry -> arbitrage diagnostics + figures."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from nvs import viz
from nvs.config import ROOT, load_config, setup_logging
from nvs.data.clean import build_quotes, load_snapshot
from nvs.surface.arbitrage import check_surface
from nvs.surface.grid import eval_grid, raw_surface_w, svi_iv_at_quotes, svi_surface_w
from nvs.surface.svi import fit_slices

log = logging.getLogger(__name__)


def main() -> None:
    setup_logging()
    cfg = load_config()
    chain, meta = load_snapshot(cfg)
    quotes = build_quotes(chain, meta, cfg)
    slices = fit_slices(quotes)
    proc = ROOT / cfg["data"]["processed_dir"]
    proc.mkdir(parents=True, exist_ok=True)
    quotes.to_csv(proc / "quotes.csv", index=False)
    slices.to_csv(proc / "svi_slices.csv", index=False)
    stamp = f"{meta['ticker']} spot={meta['spot']} captured={meta['captured_utc']}\n"
    (proc / "meta.txt").write_text(stamp, encoding="utf-8")

    tag = f"{meta['ticker']} real snapshot {meta['captured_utc'][:10]}"
    tbl, fig = ROOT / "results" / "tables", ROOT / "results" / "figures"
    tbl.mkdir(parents=True, exist_ok=True)
    slices.round(5).to_csv(tbl / "svi_slices.csv", index=False)

    k, T = eval_grid(cfg, float(slices["T"].max()))
    w_svi = svi_surface_w(slices, k, T)
    rows = [{"method": name, "data": "real", **check_surface(w, k)}
            for name, w in (("svi", w_svi), ("raw_interp", raw_surface_w(quotes, k, T)))]
    pd.DataFrame(rows).to_csv(tbl / "arbitrage_baselines.csv", index=False)
    log.info("arbitrage (real, k window %s):\n%s", cfg["smoother"]["eval_k"],
             pd.DataFrame(rows).to_string())

    viz.plot_surface_3d(k, T, np.sqrt(w_svi / T[:, None]), quotes, fig / "surface_svi_3d.html",
                        fig / "surface_svi_3d.png", f"SVI surface vs market quotes - {tag}", "SVI")

    def model_fn(kk, t):
        return np.sqrt(svi_surface_w(slices, kk, [t])[0] / t)

    viz.plot_smiles(quotes, model_fn, fig / "smiles_svi.png", f"SVI smiles - {tag}")
    viz.plot_term_structure(quotes, model_fn, fig / "term_structure_svi.png",
                            f"ATM term structure - {tag}")
    viz.plot_residuals(quotes, svi_iv_at_quotes(slices, quotes), fig / "residuals_svi.png",
                       f"SVI residuals - {tag}")
    log.info("SVI slice rmse (vol pts): %s", slices["rmse_volpts"].round(2).tolist())


if __name__ == "__main__":
    main()
