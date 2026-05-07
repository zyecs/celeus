"""Orchestrate the v0504 rebuild (6-pair scope extending v0502).

Two-phase build:

* Phase B (default): main figures only (rq1/3/4/5/6/7 + tables that don't
  depend on acquisition-sweep results). Skips compute_mae_axis,
  render_mae_axis, and render_acquisition_appendix because the new
  deepseek_67b-target cells haven't been swept yet.

* Phase C: post-SLURM completion. Re-runs only the three steps skipped in
  Phase B once the v0504 acquisition-sweep cells have been merged.

Routes ``SAVE_PE_CONFIG`` so all subprocess steps load the v0504 yaml
(6 paper_pairs, 18 cells_v0502). v0502 outputs are untouched.

Usage::
    python scripts/paper_experiment/build_v0504_outputs.py             # Phase B (default)
    python scripts/paper_experiment/build_v0504_outputs.py --phase C   # post-SLURM completion
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
PY = sys.executable
OUT_ROOT = _REPO / "results" / "paper_experiments_v0504"
V0504_CONFIG = _REPO / "configs" / "paper_experiment_v0504.yaml"

# Route every subprocess child's load_config() to the v0504 yaml. This is
# the env hook added to save.paper_experiment.config.default_config_path().
os.environ["SAVE_PE_CONFIG"] = str(V0504_CONFIG)


def _run(cmd: list[str], **kwargs) -> None:
    print(f"\n>>> {' '.join(str(c) for c in cmd)}")
    env = kwargs.pop("env", None) or os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    extra = [str(_REPO), str(_REPO / "src")]
    parts = existing.split(":") if existing else []
    for e in extra:
        if e not in parts:
            parts.insert(0, e)
    env["PYTHONPATH"] = ":".join(p for p in parts if p)
    # Make sure the v0504 yaml route reaches the subprocess too.
    env.setdefault("SAVE_PE_CONFIG", str(V0504_CONFIG))
    result = subprocess.run(cmd, env=env, **kwargs)
    if result.returncode != 0:
        sys.exit(f"step failed: {' '.join(str(c) for c in cmd)}")


def main_phase_b() -> None:
    """Main figures + tables that don't require new acquisition-sweep data."""
    # Step 1: snapshot symlinks + per_seed filter + manifest
    _run([PY, "scripts/paper_experiment/build_v0502_snapshot.py",
          "--out-root", str(OUT_ROOT)])

    # Step 2: build v0504 summary.csv
    _run([PY, "scripts/paper_experiment/build_summary.py",
          "--out-root", str(OUT_ROOT), "--include-oracle-accuracy"])

    # Step 3: regenerate RQ4/5/6 figures from filtered per_seed CSVs
    _run([PY, "scripts/paper_experiment/regen_plots_only.py",
          "--out-root", str(OUT_ROOT), "--paper-pairs"])

    # Step 3b: rq4-rq5 combined figure
    _run([PY, "scripts/paper_experiment/plot_rq4_rq5_combined.py",
          "--rq4", str(OUT_ROOT / "rq4-unbiasedness/accuracy/per_dataset_curves.csv"),
          "--rq5", str(OUT_ROOT / "rq5-signal-mse/accuracy/per_dataset_curves.csv"),
          "--out", str(OUT_ROOT / "rq4-rq5-combined/combined.pdf")])

    # Step 3b': rq4-rq5-rq6 combined figure (adds conditional-variance column,
    # mean ± 1·SD-between bands).
    _run([PY, "scripts/paper_experiment/plot_rq4_rq5_rq6_combined.py",
          "--out-root", str(OUT_ROOT)])

    # Step 3c: legacy RQ1 figure (uses cereval campaign; --skip-cereval keeps
    # the M5 columns out for the v0504 scope, matching the v0502 main pipeline).
    _run([PY, "scripts/paper_experiment/run_rq1_efficiency.py",
          "--paper-pairs",
          "--output-root", str(OUT_ROOT / "rq1-efficiency"),
          "--skip-cereval", "--plot-only"])

    # Step 3d: CELEUS-anchored RQ1 variant.
    _run([PY, "scripts/paper_experiment/run_rq1_efficiency_new.py",
          "--input-root", str(OUT_ROOT),
          "--output-root", str(OUT_ROOT / "rq1-efficiency-new")])

    # Step 3d': appendix-1 — CE width main figure (mirrors accuracy main).
    _run([PY, "scripts/paper_experiment/run_rq1_efficiency_new.py",
          "--input-root", str(OUT_ROOT),
          "--output-root", str(OUT_ROOT / "rq1-efficiency-new"),
          "--loss", "cross_entropy"])

    # Step 3d'': appendix-4 — per-(model-pair) width grid (accuracy only).
    _run([PY, "scripts/paper_experiment/plot_per_pair_efficiency.py",
          "--input-root", str(OUT_ROOT),
          "--output-root", str(OUT_ROOT / "per_pair_efficiency")])

    # Step 3e: RQ7 complementarity.
    _run([PY, "scripts/paper_experiment/compute_rq7_predictors.py",
          "--out-root", str(OUT_ROOT)])
    _run([PY, "scripts/paper_experiment/compute_rq7_outcomes.py",
          "--out-root", str(OUT_ROOT)])
    _run([PY, "scripts/paper_experiment/plot_rq7_complementarity.py",
          "--out-root", str(OUT_ROOT)])

    # Step 4: legacy acquisition appendix (cells_legacy unchanged from v0502).
    # The v0502-scope appendix is skipped in Phase B because new deepseek
    # cells haven't been swept yet; re-runs in Phase C.
    _run([PY, "scripts/paper_experiment/render_acquisition_appendix.py",
          "--out-root", str(OUT_ROOT), "--cells", "legacy", "--strategies", "all"])

    # Step 5: tables that don't depend on the v0504 acq sweep.
    for renderer in [
        "render_surrogate_onoff.py",
        "render_ce_appendix.py",
        "render_per_cell_efficiency.py",
        "render_per_cell_mechanism.py",
    ]:
        _run([PY, f"scripts/paper_experiment/{renderer}",
              "--out-root", str(OUT_ROOT), "--paper-pairs"])

    # Step 5b: appendix-3 — hparam pooled-per-dataset table (cross-snapshot
    # read from results/paper_experiment/summary.csv).
    _run([PY, "scripts/paper_experiment/render_hparam_appendix.py",
          "--out-root", str(OUT_ROOT)])

    # Step 5c: appendix-5 — per-round wallclock table (M5 dropped).
    _run([PY, "scripts/paper_experiment/render_wallclock_appendix.py",
          "--out-root", str(OUT_ROOT)])

    print(f"\n=== v0504 Phase B complete: {OUT_ROOT} ===")
    print("Next: submit stage6c_acquisition_sweep_v0504_{acc,ce}.slurm,")
    print("      then re-run with --phase C to populate the remaining appendix.")


def main_phase_c() -> None:
    """Post-SLURM completion: render the v0502-scope appendix + MAE axis
    using the freshly merged v0504 acquisition-sweep cells."""
    _run([PY, "scripts/paper_experiment/render_acquisition_appendix.py",
          "--out-root", str(OUT_ROOT), "--cells", "v0502", "--strategies", "top3"])
    _run([PY, "scripts/paper_experiment/compute_mae_axis.py",
          "--out-root", str(OUT_ROOT), "--paper-pairs", "--axis", "mae_acq"])
    _run([PY, "scripts/paper_experiment/render_mae_axis.py",
          "--out-root", str(OUT_ROOT), "--paper-pairs"])
    # Appendix-2 — per-strategy width trajectory figures (top-3 per loss).
    # Reads sweep cells produced by the SLURM run + merged via merge-cells.
    _run([PY, "scripts/paper_experiment/plot_acq_strategies_width.py",
          "--input-root", str(OUT_ROOT),
          "--output-root", str(OUT_ROOT / "acquisition_appendix")])
    print(f"\n=== v0504 Phase C complete: {OUT_ROOT} ===")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phase", choices=("B", "C"), default="B",
                    help="B = main figures (default); C = post-SLURM appendix completion")
    args = ap.parse_args()
    if args.phase == "B":
        main_phase_b()
    else:
        main_phase_c()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
