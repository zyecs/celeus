"""Build every paper figure and table from the merged experiment cells.

Run after the experiment stages in ``scripts/run_paper_experiment.py`` have
finished. The build has two parts:

* ``main`` (default): main-text figures and the appendix tables that do not
  need the acquisition-strategy sweep.
* ``acquisition``: the acquisition-strategy appendix (tables and the
  per-strategy width figures). Run it after the ``paper-acq-sweep`` stage
  has been merged.

All outputs are written under ``results/paper_outputs/``.

Usage::

    python scripts/paper_experiment/build_paper_outputs.py
    python scripts/paper_experiment/build_paper_outputs.py --part acquisition
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
PY = sys.executable
OUT_ROOT = _REPO / "results" / "paper_outputs"
SCRIPTS = "scripts/paper_experiment"


def _run(script: str, *args: str) -> None:
    cmd = [PY, f"{SCRIPTS}/{script}", *args]
    print(f"\n>>> {' '.join(cmd)}")
    env = os.environ.copy()
    parts = [p for p in env.get("PYTHONPATH", "").split(":") if p]
    for extra in (str(_REPO), str(_REPO / "src")):
        if extra not in parts:
            parts.insert(0, extra)
    env["PYTHONPATH"] = ":".join(parts)
    if subprocess.run(cmd, env=env).returncode != 0:
        sys.exit(f"step failed: {' '.join(cmd)}")


def build_main() -> None:
    out = str(OUT_ROOT)

    # Collect the merged cells of the paper pairs and summarise them.
    _run("build_snapshot.py", "--out-root", out)
    _run("build_summary.py", "--out-root", out, "--include-oracle-accuracy")

    # Bias, signal MSE and variance (RQ4-RQ6), per metric and combined.
    _run("regen_plots_only.py", "--out-root", out, "--paper-pairs")
    _run("plot_rq4_rq5_rq6_combined.py", "--out-root", out)

    # Width versus number of evaluated samples, 0-1 loss and cross-entropy.
    rq1_out = str(OUT_ROOT / "rq1-efficiency")
    _run("run_rq1_efficiency.py", "--input-root", out, "--output-root", rq1_out)
    _run("run_rq1_efficiency.py", "--input-root", out, "--output-root", rq1_out,
         "--loss", "cross_entropy")
    _run("plot_per_pair_efficiency.py", "--input-root", out,
         "--output-root", str(OUT_ROOT / "per_pair_efficiency"))

    # Complementarity of surrogate and acquisition (RQ7).
    _run("compute_rq7_predictors.py", "--out-root", out)
    _run("compute_rq7_outcomes.py", "--out-root", out)
    _run("plot_rq7_complementarity.py", "--out-root", out)

    # Acquisition appendix on the original six-cell sweep.
    _run("render_acquisition_appendix.py", "--out-root", out,
         "--cells", "legacy", "--strategies", "all")

    for renderer in ("render_surrogate_onoff.py", "render_ce_appendix.py",
                     "render_per_cell_efficiency.py", "render_per_cell_mechanism.py"):
        _run(renderer, "--out-root", out, "--paper-pairs")
    _run("render_hparam_appendix.py", "--out-root", out)
    _run("render_wallclock_appendix.py", "--out-root", out)

    print(f"\nMain outputs written to {OUT_ROOT}.")
    print("After the paper-acq-sweep stage is merged, run again with --part acquisition.")


def build_acquisition() -> None:
    out = str(OUT_ROOT)
    _run("render_acquisition_appendix.py", "--out-root", out,
         "--cells", "paper", "--strategies", "top3")
    _run("compute_mae_axis.py", "--out-root", out, "--paper-pairs", "--axis", "mae_acq")
    _run("render_mae_axis.py", "--out-root", out, "--paper-pairs")
    _run("plot_acq_strategies_width.py", "--input-root", out,
         "--output-root", str(OUT_ROOT / "acquisition_appendix"))
    print(f"\nAcquisition appendix written to {OUT_ROOT}.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--part", choices=("main", "acquisition"), default="main")
    args = ap.parse_args()
    build_main() if args.part == "main" else build_acquisition()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
