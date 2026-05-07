"""Orchestrate the full v0502 rebuild: snapshot → summary → RQ1/4/5/6
plotters → all renders. Spec §7.1 (Approach 3 thin wrapper).

Idempotent: each step is independently restartable. Manifest preserves
fields populated by Phase E (`acq_sweep_v0502_run`, `row_count_checks`).
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
PY = sys.executable
OUT_ROOT = _REPO / "results" / "paper_experiments_v0502"


def _run(cmd: list[str], **kwargs) -> None:
    print(f"\n>>> {' '.join(str(c) for c in cmd)}")
    # Ensure PYTHONPATH includes:
    #   - repo root (for `from scripts.paper_experiment.X import ...`)
    #   - repo/src (for `from save.paper_experiment.X import ...`)
    import os
    env = kwargs.pop("env", None) or os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    extra = [str(_REPO), str(_REPO / "src")]
    parts = existing.split(":") if existing else []
    for e in extra:
        if e not in parts:
            parts.insert(0, e)
    env["PYTHONPATH"] = ":".join(p for p in parts if p)
    result = subprocess.run(cmd, env=env, **kwargs)
    if result.returncode != 0:
        sys.exit(f"step failed: {' '.join(str(c) for c in cmd)}")


def main() -> int:
    # Step 1: snapshot symlinks + per_seed filter + manifest
    _run([PY, "scripts/paper_experiment/build_v0502_snapshot.py"])

    # Step 2: build v0502 summary.csv
    _run([PY, "scripts/paper_experiment/build_summary.py",
          "--out-root", str(OUT_ROOT), "--include-oracle-accuracy"])

    # Step 3: regenerate RQ1/4/5/6 figures from filtered per_seed CSVs
    _run([PY, "scripts/paper_experiment/regen_plots_only.py",
          "--out-root", str(OUT_ROOT), "--paper-pairs"])

    # Step 3b: regenerate the rq4-rq5 combined figure (consumes per_dataset_curves.csv
    # produced in step 3; not part of regen_plots_only).
    _run([PY, "scripts/paper_experiment/plot_rq4_rq5_combined.py",
          "--rq4", str(OUT_ROOT / "rq4-unbiasedness/accuracy/per_dataset_curves.csv"),
          "--rq5", str(OUT_ROOT / "rq5-signal-mse/accuracy/per_dataset_curves.csv"),
          "--out", str(OUT_ROOT / "rq4-rq5-combined/combined.pdf")])

    # Step 3c: regenerate the v0502 RQ1 efficiency figure (separate from regen_plots_only).
    _run([PY, "scripts/paper_experiment/run_rq1_efficiency.py",
          "--paper-pairs",
          "--output-root", str(OUT_ROOT / "rq1-efficiency"),
          "--skip-cereval", "--plot-only"])

    # Step 3d: CELEUS-anchored RQ1 variant (sibling of step 3c). Adds the M3
    # ablation curve and pins x_end to where CELEUS reaches epsilon. Output
    # lives next to step 3c's folder; paper/main.tex is not switched here.
    _run([PY, "scripts/paper_experiment/run_rq1_efficiency_new.py",
          "--input-root", str(OUT_ROOT),
          "--output-root", str(OUT_ROOT / "rq1-efficiency-new")])

    # Step 3e: RQ7 complementarity analysis (pre-registered predictors,
    # variance-reduction outcomes, clustered bootstrap, and figures).
    _run([PY, "scripts/paper_experiment/compute_rq7_predictors.py",
          "--out-root", str(OUT_ROOT)])
    _run([PY, "scripts/paper_experiment/compute_rq7_outcomes.py",
          "--out-root", str(OUT_ROOT)])
    _run([PY, "scripts/paper_experiment/plot_rq7_complementarity.py",
          "--out-root", str(OUT_ROOT)])

    # Step 4: render appendix tables (primary v0502 + supplementary legacy)
    _run([PY, "scripts/paper_experiment/render_acquisition_appendix.py",
          "--out-root", str(OUT_ROOT), "--cells", "v0502", "--strategies", "top3"])
    _run([PY, "scripts/paper_experiment/render_acquisition_appendix.py",
          "--out-root", str(OUT_ROOT), "--cells", "legacy", "--strategies", "all"])

    # Step 5: render the various tables. compute_mae_axis must run BEFORE
    # render_mae_axis (the latter consumes mae_axis.csv produced by the former).
    for renderer in [
        "render_surrogate_onoff.py",
        "render_ce_appendix.py",
        "render_per_cell_efficiency.py",
        "render_per_cell_mechanism.py",
        "compute_mae_axis.py",
        "render_mae_axis.py",
    ]:
        cmd = [PY, f"scripts/paper_experiment/{renderer}",
               "--out-root", str(OUT_ROOT), "--paper-pairs"]
        if renderer == "compute_mae_axis.py":
            cmd.extend(["--axis", "mae_acq"])
        _run(cmd)

    print(f"\n=== v0502 rebuild complete: {OUT_ROOT} ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
