#!/usr/bin/env python3
# scripts/paper_experiment/check_missing_cells.py
"""Write results/paper_experiment/missing_cells.csv (spec §9)."""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))

from save.paper_experiment.cell_paths import (  # noqa: E402
    acquisition_sweep_cell_path, beta_sweep_cell_path, ce_sweep_cell_path,
    hparam_sweep_cell_path, main_cell_path, oracle_accuracy_cell_path,
    wallclock_cell_path,
)
from save.paper_experiment.config import load_config  # noqa: E402
from save.paper_experiment.index_maps import (  # noqa: E402
    acquisition_sweep_cells, beta_sweep_cells, ce_sweep_cells,
    hparam_sweep_cells, main_cells_for_loss, oracle_accuracy_cells,
    wallclock_cells,
)


def collect_missing(out_base: Path, cells: list, kind: str) -> list[tuple[str, object]]:
    missing = []
    for cell in cells:
        if kind == "ce_sweep":
            path = ce_sweep_cell_path(
                out_base, dataset=cell.dataset, surrogate=cell.surrogate,
                target=cell.target, surrogate_type=cell.surrogate_type,
                beta_min=cell.beta_min,
            )
        elif kind == "beta_sweep":
            path = beta_sweep_cell_path(
                out_base, dataset=cell.dataset, surrogate=cell.surrogate,
                target=cell.target, loss=cell.loss, beta_min=cell.beta_min,
            )
        elif kind == "main":
            path = main_cell_path(
                out_base, method=cell.method_id, dataset=cell.dataset,
                surrogate=cell.surrogate, target=cell.target, loss=cell.loss,
            )
        elif kind == "oracle_accuracy":
            path = oracle_accuracy_cell_path(
                out_base, dataset=cell.dataset, surrogate=cell.surrogate,
                target=cell.target, surrogate_type="remark2_oracle_strategy4",
            )
        elif kind == "acquisition_sweep":
            path = acquisition_sweep_cell_path(
                out_base, dataset=cell.dataset, surrogate=cell.surrogate,
                target=cell.target, loss=cell.loss,
                surrogate_type=cell.surrogate_type,
            )
        elif kind == "hparam_sweep":
            path = hparam_sweep_cell_path(
                out_base, dataset=cell.dataset, surrogate=cell.surrogate,
                target=cell.target, loss=cell.loss,
                config_name=cell.config_name,
            )
        elif kind == "wallclock":
            path = wallclock_cell_path(
                out_base, dataset=cell.dataset, surrogate=cell.surrogate,
                target=cell.target, loss=cell.loss,
            )
        else:
            raise ValueError(f"unknown kind: {kind!r}")
        if not path.exists():
            missing.append((kind, cell))
    return missing


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Write missing_cells.csv")
    p.add_argument(
        "--include-oracle-accuracy",
        action="store_true",
        help="Include trajectories/oracle_accuracy in the missing-cell scan.",
    )
    p.add_argument(
        "--paper-pairs", action="store_true",
        help="Restrict cell audit to cfg.paper_pair_keys (v0502 scope).",
    )
    return p.parse_args(argv)


def _cell_in_paper_pairs(cell, paper_pair_keys: set) -> bool:
    """Return True if (cell.surrogate, cell.target) is in paper_pair_keys."""
    return (getattr(cell, "surrogate", None), getattr(cell, "target", None)) in paper_pair_keys


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = load_config(_REPO / "configs" / "paper_experiment.yaml")
    out_base = Path(cfg.paths["out_root"])

    # When --paper-pairs is active, restrict all cell lists to paper_pair_keys.
    paper_pair_keys = cfg.paper_pair_keys if args.paper_pairs else None

    def _maybe_filter(cells):
        if paper_pair_keys is None:
            return cells
        return [c for c in cells if _cell_in_paper_pairs(c, paper_pair_keys)]

    all_missing = []
    for loss in cfg.losses:
        all_missing += collect_missing(
            out_base,
            _maybe_filter(main_cells_for_loss(cfg, loss=loss, include_cereval=True)),
            "main",
        )
    all_missing += collect_missing(out_base, _maybe_filter(ce_sweep_cells(cfg)), "ce_sweep")
    all_missing += collect_missing(out_base, _maybe_filter(beta_sweep_cells(cfg)), "beta_sweep")
    if args.include_oracle_accuracy:
        all_missing += collect_missing(
            out_base, _maybe_filter(oracle_accuracy_cells(cfg)), "oracle_accuracy"
        )
    # §6.5 acquisition-sweep audit (only emit rows when block populated).
    if cfg.acquisition_sweep.get("seeds"):
        for loss in cfg.losses:
            all_missing += collect_missing(
                out_base, _maybe_filter(acquisition_sweep_cells(cfg, loss=loss)),
                "acquisition_sweep",
            )
    # §6.5 hparam-sweep audit (only emit rows when block populated).
    if cfg.hparam_sweep.get("seeds"):
        all_missing += collect_missing(
            out_base, _maybe_filter(hparam_sweep_cells(cfg)), "hparam_sweep",
        )
    # §6.5 wallclock audit (only emit rows when block populated).
    if cfg.wallclock.get("seeds"):
        all_missing += collect_missing(
            out_base, _maybe_filter(wallclock_cells(cfg)), "wallclock",
        )

    csv_path = out_base / "missing_cells.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["kind", "method", "dataset", "surrogate", "target", "loss",
                    "surrogate_type", "beta_min"])
        for kind, c in all_missing:
            w.writerow([
                kind,
                getattr(c, "method_id", ""),
                c.dataset, c.surrogate, c.target,
                getattr(c, "loss", ""),
                getattr(c, "surrogate_type", ""),
                getattr(c, "beta_min", ""),
            ])
    print(f"missing: {len(all_missing)} → {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
