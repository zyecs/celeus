"""Shared loader for paper-experiment cell files -> tidy DataFrame.

Dispatches per-subdir parsers so main / ce_sweep / beta_sweep cells all
flow through the same row model in ``summary.csv``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from save.paper_experiment.cell_paths import (  # noqa: E402
    parse_acquisition_sweep_cell_filename,
    parse_beta_sweep_cell_filename,
    parse_ce_sweep_cell_filename,
    parse_hparam_sweep_cell_filename,
    parse_main_cell_filename,
    parse_oracle_accuracy_cell_filename,
    parse_wallclock_cell_filename,
)
from save.paper_experiment.cell_schema import load_cell  # noqa: E402


_SUBDIR_PARSERS = {
    "trajectories/main": parse_main_cell_filename,
    "trajectories/ce_sweep": parse_ce_sweep_cell_filename,
    "trajectories/beta_sweep": parse_beta_sweep_cell_filename,
    "trajectories/oracle_accuracy": parse_oracle_accuracy_cell_filename,
    "trajectories/acquisition_sweep": parse_acquisition_sweep_cell_filename,
    "trajectories/hparam_sweep": parse_hparam_sweep_cell_filename,
    "trajectories/wallclock": parse_wallclock_cell_filename,
}


def load_summary(
    out_base: Path | None = None,
    subdir: str = "trajectories/main",
) -> pd.DataFrame:
    base = Path(out_base) if out_base else (_REPO / "results" / "paper_experiment")
    traj_dir = base / subdir
    parser = _SUBDIR_PARSERS.get(subdir)
    if parser is None:
        raise ValueError(f"no parser registered for subdir {subdir!r}")
    rows = []
    for path in sorted(traj_dir.glob("cell__*.npz")):
        try:
            keys = parser(path.name)
        except ValueError:
            continue
        meta, results = load_cell(path)
        for seed, r in results.items():
            row = {
                **keys,
                # Method defaults to the cell's metadata - main files carry
                # it explicitly; sweep files fall back to meta.method_id
                # which for ce_sweep/beta_sweep is always "M1".
                "method": keys.get("method", meta.method_id),
                "seed": int(seed),
                "did_stop": r.did_stop,
                "labels_to_stop": r.labels_to_stop,
                "width_at_stop": r.width_at_stop,
                "coverage_at_stop": r.coverage_at_stop,
                "ever_miss": r.ever_miss,
                "pop_inverted_count": r.pop_inverted_count,
                "final_width": r.final_width,
                "true_R": r.true_R,
                "rho": r.rho,
                "elapsed_seconds": r.elapsed_seconds,
                "git_commit": r.git_commit,
                "hostname": r.hostname,
                "T_max": meta.T_max,
                "epsilon": meta.epsilon,
                "beta_min": meta.beta_min,
                "surrogate_type": meta.surrogate_type,
                "adaptive_bounds": meta.adaptive_bounds,
                "pool_sha256": meta.pool_sha256,
                "config_json": meta.config_json,
            }
            rows.append(row)
    return pd.DataFrame(rows)


def load_trajectories(
    out_base: Path | None = None,
    subdir: str = "trajectories/main",
) -> dict[Path, tuple]:
    """Return {path: (metadata, {seed: PerSeedResult})}."""
    base = Path(out_base) if out_base else (_REPO / "results" / "paper_experiment")
    traj_dir = base / subdir
    return {p: load_cell(p) for p in sorted(traj_dir.glob("cell__*.npz"))}
