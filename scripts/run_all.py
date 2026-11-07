"""Run the whole pipeline in order. `python scripts/run_all.py [--skip-fetch] [--skip-train]`.

--skip-fetch reuses the latest snapshot in data/raw; --skip-train reuses results/models/pricer.pt.
"""
import argparse
import subprocess
import sys

STAGES = [
    ("fetch", ["scripts/collect_snapshot.py"]),
    ("surface", ["-m", "nvs.surface.build"]),
    ("train-pricer", ["-m", "nvs.nn.pricer"]),
    ("calibrate", ["-m", "nvs.calibrate"]),
    ("smooth", ["-m", "nvs.nn.smoother"]),
    ("forecast", ["-m", "nvs.nn.forecaster"]),
]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-fetch", action="store_true")
    ap.add_argument("--skip-train", action="store_true")
    a = ap.parse_args()
    skip = {"fetch"} if a.skip_fetch else set()
    skip |= {"train-pricer"} if a.skip_train else set()
    for name, cmd in STAGES:
        if name in skip:
            continue
        print(f"=== {name} ===", flush=True)
        subprocess.run([sys.executable, *cmd], check=True)
