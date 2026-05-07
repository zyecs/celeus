#!/usr/bin/env python
"""Compute pre-registered RQ7 CELEUS variance-reduction outcomes."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from save.paper_experiment.config import default_config_path, load_config  # noqa: E402
from scripts.paper_experiment.aggregation import parse_cell_filename  # noqa: E402
from scripts.paper_experiment.compute_rq7_predictors import rq7_cells  # noqa: E402

DEFAULT_OUT_ROOT = _REPO / "results" / "paper_experiments_v0502"
DEFAULT_CONFIG = default_config_path()
ANCHORS = (500, 1000, 1500, 2000, 2500, 3000)


def _annotate_cell_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    parsed = out["cell"].map(parse_cell_filename).apply(pd.Series)
    for col in ("dataset", "surrogate", "target", "loss"):
        out[f"_cell_{col}"] = parsed[col]
    return out


def _eta_from_values(v_uniform: float, v_celeus: float) -> float:
    if not np.isfinite(v_uniform) or v_uniform == 0.0:
        return float("nan")
    return float((v_uniform - v_celeus) / v_uniform)


def build_outcomes(
    *,
    out_root: Path,
    config_path: Path,
    anchors: tuple[int, ...] = ANCHORS,
) -> pd.DataFrame:
    cfg = load_config(config_path)
    rq6_csv = out_root / "rq6-variance" / "accuracy" / "per_cell_curves.csv"
    curves = _annotate_cell_columns(pd.read_csv(rq6_csv))
    curves = curves[curves["t"].isin(anchors)].copy()

    rows = []
    for cell in rq7_cells(cfg):
        dataset = cell["dataset"]
        surrogate = cell["surrogate"]
        target = cell["target"]
        sub = curves[
            (curves["_cell_dataset"] == dataset)
            & (curves["_cell_surrogate"] == surrogate)
            & (curves["_cell_target"] == target)
            & (curves["_cell_loss"] == "accuracy")
            & (curves["acquisition"].isin(["ada", "uniform"]))
        ]
        if sub.empty:
            continue
        pivot = sub.pivot_table(
            index="t",
            columns="acquisition",
            values="cond_var_S_mean",
            aggfunc="mean",
        )
        anchor_etas: dict[int, float] = {}
        for t in anchors:
            if t not in pivot.index or {"ada", "uniform"} - set(pivot.columns):
                anchor_etas[t] = float("nan")
                continue
            anchor_etas[t] = _eta_from_values(
                float(pivot.loc[t, "uniform"]),
                float(pivot.loc[t, "ada"]),
            )
        rows.append(
            {
                "dataset": dataset,
                "surrogate": surrogate,
                "target": target,
                "eta_bar_full": float(np.nanmean(list(anchor_etas.values()))),
                **{f"eta_{t}": anchor_etas[t] for t in anchors},
                # Stored trajectories lack item identities; B-split variance
                # recomputation requires deterministic acquisition replay and
                # is intentionally flagged in manifest.json by the plot script.
                "eta_bar_B": float("nan"),
            }
        )
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args(argv)

    out_dir = args.out_root / "rq7-complementarity"
    out_dir.mkdir(parents=True, exist_ok=True)
    df = build_outcomes(out_root=args.out_root, config_path=args.config)
    df.to_csv(out_dir / "outcomes.csv", index=False, float_format="%.18g")
    print(f"wrote {out_dir / 'outcomes.csv'} ({len(df)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
