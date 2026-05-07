#!/usr/bin/env python
"""rq6 — Variance Comparison (§6.4) plotter, Nature-style rewrite.

DOUBLE-column exception: 6.75 × 4.8 in figure with 3 dataset rows × 2 metric
columns (cond_var_S | emp_var_R). 3 acquisition curves per panel
(ada / oracle / uniform).

Bands:
  - cond_var_S column: ±2·MC-SE bands.
  - emp_var_R  column: ±1·SD-between bands (sample-variance MC-SE misleading).

Per spec §5.5, §5.6 (commit 4c5596c).
"""
from __future__ import annotations

import argparse
import json
import logging
import socket
import subprocess
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.paper_experiment.aggregation import (
    per_dataset_pool, per_pair_filter, filter_per_seed_by_paper_pairs,
)
from scripts.paper_experiment.plot_style import (
    ACQUISITION_STYLE, BAND_ALPHA, DATASET_ORDER,
    apply_rc_helvetica, bands_enabled, finalize_panel, get_runtime_pairs,
)

logger = logging.getLogger("rq6")
T_GRID = np.array([500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000],
                  dtype=np.int64)


def _git_hash():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _legacy_pool_global(per_seed_df: pd.DataFrame) -> pd.DataFrame:
    """LEGACY aggregated_curves.csv (no dataset column).

    Header (verified live): acquisition, t, n_cells, mean_cond_var_S,
    se_cond_var_S, sd_between_cond_var_S, mean_emp_var_R, sd_between_emp_var_R.
    """
    per_cell = (
        per_seed_df.groupby(["acquisition", "cell", "t"])
        .agg(
            cond_var_mean=("cond_var_S", "mean"),
            cond_var_sd=("cond_var_S", lambda v: v.std(ddof=1)),
            emp_var=("rhat", lambda v: v.var(ddof=1)),
            n_seeds=("rhat", "count"),
        )
        .reset_index()
    )
    per_cell["cond_var_se"] = per_cell["cond_var_sd"] / np.sqrt(per_cell["n_seeds"])

    rows = []
    for (acq, t), grp in per_cell.groupby(["acquisition", "t"]):
        n_c = len(grp)
        rows.append({
            "acquisition": acq, "t": int(t), "n_cells": n_c,
            "mean_cond_var_S":       float(grp["cond_var_mean"].mean()),
            "se_cond_var_S":         float(np.sqrt((grp["cond_var_se"] ** 2).sum() / n_c ** 2)),
            "sd_between_cond_var_S": float(grp["cond_var_mean"].std(ddof=1)) if n_c > 1 else 0.0,
            "mean_emp_var_R":        float(grp["emp_var"].mean()),
            "sd_between_emp_var_R":  float(grp["emp_var"].std(ddof=1)) if n_c > 1 else 0.0,
        })
    return pd.DataFrame(rows)


def _render_pooled(per_dataset_df: pd.DataFrame, out_path: Path) -> None:
    # 2 rows (metrics) × 3 cols (datasets). sharey=False so each panel adapts
    # to its own range (per-dataset cond_var_S magnitudes differ ~3×).
    fig, axes = plt.subplots(2, 3, figsize=(6.75, 3.1),
                             sharex=True, sharey=False)
    metrics = [
        {"row": 0, "key": "mean_cond_var_S", "se": "se_cond_var_S",
         "title": r"$\mathrm{Var}(\hat{S}_t \mid \mathcal{F}_{t-1})$",
         "use_se_band": True, "yscale": "linear"},
        {"row": 1, "key": "mean_emp_var_R", "se": "sd_between_emp_var_R",
         "title": r"$\mathrm{Var}(\hat R_t)$",
         "use_se_band": False, "yscale": "log"},
    ]
    for col_idx, dataset in enumerate(DATASET_ORDER):
        sub = per_dataset_df[per_dataset_df["dataset"] == dataset]
        for metric in metrics:
            ax = axes[metric["row"]][col_idx]
            panel_lo = float("inf")
            panel_hi = -float("inf")
            for acq in ("ada", "oracle_accuracy", "uniform"):
                acq_rows = sub[sub["acquisition"] == acq].sort_values("t")
                if len(acq_rows) == 0:
                    continue
                color = ACQUISITION_STYLE[acq]["color"]
                label = ACQUISITION_STYLE[acq]["label"]
                ax.plot(acq_rows["t"], acq_rows[metric["key"]],
                        color=color, lw=1.4, label=label)
                if metric["use_se_band"]:
                    band = 2 * acq_rows[metric["se"]]
                else:
                    band = acq_rows[metric["se"]]  # ±1·SD-between
                if bands_enabled():
                    ax.fill_between(
                        acq_rows["t"],
                        acq_rows[metric["key"]] - band,
                        acq_rows[metric["key"]] + band,
                        color=color, alpha=BAND_ALPHA, linewidth=0,
                    )
                panel_lo = min(panel_lo, float((acq_rows[metric["key"]] - band).min()))
                panel_hi = max(panel_hi, float((acq_rows[metric["key"]] + band).max()))
            ax.set_yscale(metric["yscale"])
            if metric["yscale"] == "linear" and panel_hi > panel_lo:
                pad = (panel_hi - panel_lo) * 0.12
                ax.set_ylim(max(0.0, panel_lo - pad), panel_hi + pad)
            finalize_panel(
                ax,
                dataset="",  # column titles label datasets in this layout
                xlabel=("labels $t$" if metric["row"] == 1 else None),
                ylabel=(metric["title"] if col_idx == 0 else None),
            )
            if col_idx == 0:
                ax.yaxis.labelpad = 2  # tight gap between ylabel and axis
            # Top row: dataset name as column title.
            if metric["row"] == 0:
                ax.set_title(dataset, fontsize=8, pad=2)
    # Single shared legend below the figure (close to x-axis labels).
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center",
               bbox_to_anchor=(0.5, 0.02), ncol=3, fontsize=6)
    fig.tight_layout(rect=(0, 0.05, 1, 0.99), h_pad=0.1, w_pad=0.2)
    fig.savefig(out_path)
    plt.close(fig)


def _render_showcase(per_seed_df: pd.DataFrame, out_path: Path,
                     acquisition: str = "ada",
                     pair_defs: list[dict] | None = None) -> None:
    pairs = pair_defs if pair_defs is not None else get_runtime_pairs()
    pair_filtered = per_pair_filter(per_seed_df, pairs)
    pair_filtered = pair_filtered[pair_filtered["acquisition"] == acquisition]
    per_cell = (
        pair_filtered.groupby(
            ["dataset", "pair_slot", "pair_label", "pair_color", "t"]
        )
        .agg(
            cond_var_mean=("cond_var_S", "mean"),
            emp_var=("rhat", lambda v: v.var(ddof=1)),
        )
        .reset_index()
    )

    fig, axes = plt.subplots(2, 3, figsize=(6.75, 3.1),
                             sharex=True, sharey=False)
    row_specs = [
        ("cond_var_mean", r"$\mathrm{Var}(\hat{S}_t \mid \mathcal{F}_{t-1})$", "linear"),
        ("emp_var",       r"$\mathrm{Var}(\hat R_t)$", "log"),
    ]
    for col_idx, dataset in enumerate(DATASET_ORDER):
        sub = per_cell[per_cell["dataset"] == dataset]
        for row_idx, (key, title, yscale) in enumerate(row_specs):
            ax = axes[row_idx][col_idx]
            panel_lo = float("inf")
            panel_hi = -float("inf")
            for pair_def in pairs:
                rows = sub[sub["pair_slot"] == pair_def["slot"]].sort_values("t")
                if len(rows) == 0:
                    continue
                ax.plot(rows["t"], rows[key],
                        color=pair_def["color"], lw=1.4, ls="-",
                        label=pair_def["label"])
                panel_lo = min(panel_lo, float(rows[key].min()))
                panel_hi = max(panel_hi, float(rows[key].max()))
            ax.set_yscale(yscale)
            if yscale == "linear" and panel_hi > panel_lo:
                pad = (panel_hi - panel_lo) * 0.12
                ax.set_ylim(max(0.0, panel_lo - pad), panel_hi + pad)
            finalize_panel(
                ax,
                dataset="",
                xlabel=("labels $t$" if row_idx == 1 else None),
                ylabel=(title if col_idx == 0 else None),
            )
            if col_idx == 0:
                ax.yaxis.labelpad = 2
            if row_idx == 0:
                ax.set_title(dataset, fontsize=8, pad=2)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center",
               bbox_to_anchor=(0.5, 0.02), ncol=3, fontsize=6)
    fig.tight_layout(rect=(0, 0.05, 1, 0.99), h_pad=0.1, w_pad=0.2)
    fig.savefig(out_path)
    plt.close(fig)


def _rename_stale(out_dir: Path) -> None:
    stale = out_dir / "per_cell.pdf"
    if stale.exists():
        stale.rename(out_dir / "per_cell.pdf.deprecated")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--cells-root", type=Path, required=True)
    parser.add_argument("--oracle-cells", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--acquisition-showcase", choices=("ada", "oracle_accuracy", "uniform"),
                        default="ada")
    parser.add_argument("--paper-pairs", action="store_true",
                        help="Filter cells to cfg.paper_pair_keys (v0502 scope).")
    parser.add_argument("--out-root", type=Path, default=None,
                        help="Override results root for input/output paths.")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    apply_rc_helvetica()

    out_root = args.out_root if args.out_root else args.out
    args.out.mkdir(parents=True, exist_ok=True)
    _rename_stale(args.out)

    per_seed_path = args.out / "per_seed_curves.csv"
    if not per_seed_path.exists():
        sys.exit(f"per_seed_curves.csv missing at {per_seed_path}")

    per_seed_df = pd.read_csv(per_seed_path)
    if args.paper_pairs:
        from save.paper_experiment.config import (
            default_config_path as _default_config_path,
            load_config as _load_config,
        )
        _cfg = _load_config(_default_config_path())
        per_seed_df = filter_per_seed_by_paper_pairs(per_seed_df, _cfg.paper_pair_keys)
        logger.info("paper_pairs filter: %d rows remaining", len(per_seed_df))
    t0 = time.time()
    per_dataset_df = per_dataset_pool(
        per_seed_df, section="rq6",
        value_spec={"cond_var_col": "cond_var_S", "rhat_col": "rhat"},
    )
    legacy_df = _legacy_pool_global(per_seed_df)
    aggregate_s = time.time() - t0

    per_dataset_df.to_csv(args.out / "per_dataset_curves.csv",
                          index=False, float_format="%.18g")
    legacy_df.to_csv(args.out / "aggregated_curves.csv",
                     index=False, float_format="%.18g")
    _render_pooled(per_dataset_df, args.out / "pooled.pdf")
    showcase_pair_defs = None
    if args.paper_pairs:
        from scripts.paper_experiment.plot_style import paper_pair_defs as _pp_defs
        showcase_pair_defs = _pp_defs(_cfg.paper_pairs)
    _render_showcase(per_seed_df, args.out / "showcase.pdf",
                     acquisition=args.acquisition_showcase,
                     pair_defs=showcase_pair_defs)

    meta = {
        "git_commit": _git_hash(),
        "section": "rq6",
        "n_cells": int(per_seed_df["cell"].nunique()),
        "datasets": list(DATASET_ORDER),
        "t_grid": T_GRID.tolist(),
        "showcase_acquisition": args.acquisition_showcase,
        "timing_s": {"aggregate": aggregate_s},
        "hostname": socket.gethostname(),
    }
    (args.out / "metadata.json").write_text(json.dumps(meta, indent=2, default=str))
    logger.info("wrote %s", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
