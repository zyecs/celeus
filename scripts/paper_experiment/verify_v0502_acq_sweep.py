"""Verify v0502 acquisition_sweep completeness (G3 gate, spec §11.1).

Two modes:
  --mode subcells     : every (cell, strategy, chunk) sub-cell exists. Run post-SLURM,
                        pre-merge.
  --mode merged       : every (cell, strategy) merged cell exists with valid metadata.
                        Run post-merge.
  --mode auto (default): infer from disk state. If any sub-cell exists, run subcells
                        check. Else run merged check.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))

from save.paper_experiment.config import load_config
from save.paper_experiment.cell_paths import (
    acquisition_sweep_cell_path,
    acquisition_sweep_subcell_path,
)


def _validate_merged_cell(cell_path: Path) -> list[str]:
    """Return list of validation errors for a merged cell .npz, or [] if OK."""
    errors = []
    try:
        arr = np.load(cell_path, allow_pickle=True)
    except Exception as e:
        return [f"failed to load: {e}"]
    if "did_stop" not in arr:
        errors.append("missing did_stop")
    else:
        ds = np.asarray(arr["did_stop"])
        if ds.dtype == bool:
            pass  # bool arrays are valid
        elif not np.all(np.isin(ds, [0, 1])):
            errors.append(f"did_stop has out-of-domain values {set(ds.tolist())}")
    if "ever_miss" not in arr:
        errors.append("missing ever_miss")
    if "meta__seeds" in arr:
        seeds = np.asarray(arr["meta__seeds"]).tolist()
        if len(seeds) != 50:
            errors.append(f"expected 50 seeds, got {len(seeds)}")
    return errors


def _check_subcells_mode(cfg, out_root: Path) -> list:
    """Verify every (cell, strategy, chunk) sub-cell exists (pre-merge mode)."""
    seeds = list(cfg.seeds_main)
    chunk_size = 5
    n_chunks = (len(seeds) + chunk_size - 1) // chunk_size
    missing_subcells = []
    for loss in cfg.losses:
        strategies = cfg.acquisition_sweep["strategies_v0502"][loss]
        for cell in cfg.acquisition_sweep["cells_v0502"]:
            for strat in strategies:
                for ci in range(n_chunks):
                    sp = acquisition_sweep_subcell_path(
                        out_root,
                        dataset=cell["dataset"],
                        surrogate=cell["surrogate"],
                        target=cell["target"],
                        loss=loss,
                        surrogate_type=strat,
                        chunk=ci,
                    )
                    if not sp.exists():
                        missing_subcells.append((loss, cell, strat, ci))
    return missing_subcells


def _check_merged_mode(cfg, out_root: Path) -> tuple[list, list]:
    """Verify every (cell, strategy) merged cell exists with valid metadata (post-merge)."""
    missing_cells = []
    invalid_cells = []
    for loss in cfg.losses:
        strategies = cfg.acquisition_sweep["strategies_v0502"][loss]
        for cell in cfg.acquisition_sweep["cells_v0502"]:
            for strat in strategies:
                cp = acquisition_sweep_cell_path(
                    out_root,
                    dataset=cell["dataset"],
                    surrogate=cell["surrogate"],
                    target=cell["target"],
                    loss=loss,
                    surrogate_type=strat,
                )
                if not cp.exists():
                    missing_cells.append((loss, cell, strat))
                else:
                    errs = _validate_merged_cell(cp)
                    if errs:
                        invalid_cells.append((loss, cell, strat, errs))
    return missing_cells, invalid_cells


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out-root", type=Path,
        default=_REPO / "results" / "paper_experiments_v0502",
    )
    ap.add_argument("--mode", choices=("auto", "subcells", "merged"), default="auto")
    args = ap.parse_args()

    cfg = load_config("configs/paper_experiment.yaml")
    out_root: Path = args.out_root

    if args.mode == "auto":
        v0502_subcells = out_root / "_subcells" / "acquisition_sweep"
        if v0502_subcells.exists() and any(v0502_subcells.iterdir()):
            mode = "subcells"
        else:
            mode = "merged"
        print(f"[auto] selected mode={mode}")
    else:
        mode = args.mode

    if mode == "subcells":
        missing_subcells = _check_subcells_mode(cfg, out_root)
        print(f"missing subcells: {len(missing_subcells)}")
        for m in missing_subcells[:20]:
            print(f"  SUBCELL: {m}")
        if missing_subcells:
            return 1
        print("OK: all v0502 acquisition_sweep sub-cells present (pre-merge)")
        return 0

    missing_cells, invalid_cells = _check_merged_mode(cfg, out_root)
    print(f"missing merged cells: {len(missing_cells)}")
    print(f"invalid merged cells: {len(invalid_cells)}")
    for m in missing_cells[:20]:
        print(f"  CELL MISSING: {m}")
    for m in invalid_cells[:10]:
        print(f"  CELL INVALID: {m}")
    if missing_cells or invalid_cells:
        return 1
    print("OK: all v0502 acquisition_sweep merged cells present and valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
