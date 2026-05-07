#!/usr/bin/env python
"""rq5 — Signal Error (§6.2.4) plotter, Nature-style rewrite.

3-dataset-row pooled per loss; 3-pair showcase per loss. Two CSVs per loss
(legacy + per_dataset). Loops over args.losses for accuracy + cross_entropy.

Per spec §5.3, §5.4 (commit 4c5596c).
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
    BAND_ALPHA, DATASET_ORDER, KIND_LINESTYLE, TOL_MUTED,
    apply_rc_helvetica, bands_enabled, finalize_panel, get_runtime_pairs,
)

logger = logging.getLogger("rq5")
T_GRID = np.arange(100, 5001, 100, dtype=np.int64)


def _git_hash():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _legacy_pool_global(per_seed_df: pd.DataFrame) -> pd.DataFrame:
    """LEGACY aggregated_curves.csv schema (no dataset column).

    Header (verified live): t, is_mse_pooled, naive_mse_pooled, is_se_pooled,
    naive_se_pooled, is_sd_between, naive_sd_between, n_cells.
    """
    per_cell = (
        per_seed_df.groupby(["cell", "t"])
        .agg(
            is_mean=("is_sq", "mean"),
            is_sd=("is_sq", lambda v: v.std(ddof=1)),
            naive_mean=("naive_sq", "mean"),
            naive_sd=("naive_sq", lambda v: v.std(ddof=1)),
            n_seeds=("is_sq", "count"),
        )
        .reset_index()
    )
    per_cell["is_se"] = per_cell["is_sd"] / np.sqrt(per_cell["n_seeds"])
    per_cell["naive_se"] = per_cell["naive_sd"] / np.sqrt(per_cell["n_seeds"])

    rows = []
    for t, grp in per_cell.groupby("t"):
        n_c = len(grp)
        rows.append({
            "t": int(t),
            "is_mse_pooled":     float(grp["is_mean"].mean()),
            "naive_mse_pooled":  float(grp["naive_mean"].mean()),
            "is_se_pooled":      float(np.sqrt((grp["is_se"] ** 2).sum() / n_c ** 2)),
            "naive_se_pooled":   float(np.sqrt((grp["naive_se"] ** 2).sum() / n_c ** 2)),
            "is_sd_between":     float(grp["is_mean"].std(ddof=1)) if n_c > 1 else 0.0,
            "naive_sd_between":  float(grp["naive_mean"].std(ddof=1)) if n_c > 1 else 0.0,
            "n_cells": n_c,
        })
    return pd.DataFrame(rows)


def _render_pooled(per_dataset_df: pd.DataFrame, out_path: Path, loss: str) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(3.25, 4.8), sharex=True, sharey=True)
    for ax, dataset in zip(axes, DATASET_ORDER):
        sub = per_dataset_df[per_dataset_df["dataset"] == dataset].sort_values("t")
        is_color = TOL_MUTED["indigo"]
        naive_color = TOL_MUTED["wine"]
        ax.plot(sub["t"], sub["is_mse_pooled"],
                color=is_color, ls=KIND_LINESTYLE["is_corrected"]["linestyle"],
                lw=KIND_LINESTYLE["is_corrected"]["linewidth"],
                label=KIND_LINESTYLE["is_corrected"]["label"])
        if bands_enabled():
            ax.fill_between(sub["t"],
                            sub["is_mse_pooled"] - 2 * sub["is_se_pooled"],
                            sub["is_mse_pooled"] + 2 * sub["is_se_pooled"],
                            color=is_color, alpha=BAND_ALPHA, linewidth=0)
        ax.plot(sub["t"], sub["naive_mse_pooled"],
                color=naive_color, ls=KIND_LINESTYLE["naive"]["linestyle"],
                lw=KIND_LINESTYLE["naive"]["linewidth"],
                label=KIND_LINESTYLE["naive"]["label"])
        if bands_enabled():
            ax.fill_between(sub["t"],
                            sub["naive_mse_pooled"] - 2 * sub["naive_se_pooled"],
                            sub["naive_mse_pooled"] + 2 * sub["naive_se_pooled"],
                            color=naive_color, alpha=BAND_ALPHA, linewidth=0)
        ax.set_yscale("log")
        finalize_panel(
            ax, dataset=dataset, ylabel=r"MSE vs $R_N$",
            xlabel=("labels $t$" if dataset == DATASET_ORDER[-1] else None),
        )
    axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, 1.40),
                   ncol=2, fontsize=6)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _render_showcase(per_seed_df: pd.DataFrame, out_path: Path,
                     pair_defs: list[dict] | None = None) -> None:
    pairs = pair_defs if pair_defs is not None else get_runtime_pairs()
    pair_filtered = per_pair_filter(per_seed_df, pairs)
    per_cell = (
        pair_filtered.groupby(
            ["dataset", "pair_slot", "pair_label", "pair_color", "t"]
        )
        .agg(is_mean=("is_sq", "mean"),
             is_sd=("is_sq", lambda v: v.std(ddof=1)),
             n=("is_sq", "count"))
        .reset_index()
    )
    per_cell["se"] = per_cell["is_sd"] / np.sqrt(per_cell["n"])

    fig, axes = plt.subplots(3, 1, figsize=(3.25, 4.8), sharex=True, sharey=True)
    for ax, dataset in zip(axes, DATASET_ORDER):
        sub = per_cell[per_cell["dataset"] == dataset]
        for pair_def in pairs:
            rows = sub[sub["pair_slot"] == pair_def["slot"]].sort_values("t")
            if len(rows) == 0:
                continue
            ax.plot(rows["t"], rows["is_mean"],
                    color=pair_def["color"], lw=1.4, ls="-",
                    label=pair_def["label"])
            if bands_enabled():
                ax.fill_between(rows["t"],
                                rows["is_mean"] - 2 * rows["se"],
                                rows["is_mean"] + 2 * rows["se"],
                                color=pair_def["color"], alpha=BAND_ALPHA, linewidth=0)
        ax.set_yscale("log")
        finalize_panel(
            ax, dataset=dataset, ylabel=r"$\hat S_t$ MSE",
            xlabel=("labels $t$" if dataset == DATASET_ORDER[-1] else None),
        )
    axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, 1.45),
                   ncol=1, fontsize=6)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _rename_stale(out_dir: Path) -> None:
    stale = out_dir / "per_cell.pdf"
    if stale.exists():
        stale.rename(out_dir / "per_cell.pdf.deprecated")


def _run_for_loss(args, loss: str) -> None:
    out_dir = args.out / loss
    out_dir.mkdir(parents=True, exist_ok=True)
    _rename_stale(out_dir)

    per_seed_path = out_dir / "per_seed_curves.csv"
    if not per_seed_path.exists():
        sys.exit(f"per_seed_curves.csv missing at {per_seed_path}")

    per_seed_df = pd.read_csv(per_seed_path)
    if getattr(args, "paper_pairs", False):
        from save.paper_experiment.config import (
            default_config_path as _default_config_path,
            load_config as _load_config,
        )
        _cfg = _load_config(_default_config_path())
        per_seed_df = filter_per_seed_by_paper_pairs(per_seed_df, _cfg.paper_pair_keys)
        logger.info("paper_pairs filter: %d rows remaining", len(per_seed_df))
    t0 = time.time()
    per_dataset_df = per_dataset_pool(
        per_seed_df, section="rq5",
        value_spec={"sq_err_cols": ["is_sq", "naive_sq"]},
    )
    legacy_df = _legacy_pool_global(per_seed_df)
    aggregate_s = time.time() - t0

    per_dataset_df.to_csv(out_dir / "per_dataset_curves.csv",
                          index=False, float_format="%.18g")
    legacy_df.to_csv(out_dir / "aggregated_curves.csv",
                     index=False, float_format="%.18g")
    _render_pooled(per_dataset_df, out_dir / "pooled.pdf", loss)
    showcase_pair_defs = None
    if getattr(args, "paper_pairs", False):
        from scripts.paper_experiment.plot_style import paper_pair_defs as _pp_defs
        showcase_pair_defs = _pp_defs(_cfg.paper_pairs)
    _render_showcase(per_seed_df, out_dir / "showcase.pdf",
                     pair_defs=showcase_pair_defs)

    meta = {
        "git_commit": _git_hash(),
        "section": "rq5", "loss": loss,
        "n_cells": int(per_seed_df["cell"].nunique()),
        "datasets": list(DATASET_ORDER),
        "t_grid": T_GRID.tolist(),
        "timing_s": {"aggregate": aggregate_s},
        "hostname": socket.gethostname(),
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2, default=str))
    logger.info("wrote %s", out_dir)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--cells-root", type=Path, required=True)  # back-compat
    parser.add_argument("--losses", type=str, default="accuracy,cross_entropy")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)  # back-compat, ignored
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--verbose", action="store_true")
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

    for loss in args.losses.split(","):
        _run_for_loss(args, loss.strip())
    return 0


if __name__ == "__main__":
    sys.exit(main())
