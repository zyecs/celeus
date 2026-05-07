#!/usr/bin/env python
"""rq4 — Unbiased Estimation (§6.2.3) plotter (Nature-style rewrite).

Emits TWO PDFs:
  - pooled.pdf:   3 dataset panels stacked vertically (sst2/mmlu/agnews);
                  4 curves per panel (ada-LURE, ada-unweighted,
                  oracle-LURE, oracle-unweighted) with ±2·SE on LURE only.
  - showcase.pdf: 3 dataset panels; 3 demo pairs overlaid (cross_arch/weak/strong)
                  showing only LURE curves under default acquisition (ada).

And TWO CSVs:
  - aggregated_curves.csv:    LEGACY schema preserved verbatim for the §12
                              spot-check at run_task18.slurm. No `dataset` column.
  - per_dataset_curves.csv:   NEW schema with `dataset` column; consumed by
                              the plot scripts and regen_plots_only.

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
    ACQUISITION_STYLE, BAND_ALPHA, DATASET_ORDER, KIND_LINESTYLE,
    apply_rc_helvetica, bands_enabled, finalize_panel, get_runtime_pairs,
)

logger = logging.getLogger("rq4")

T_GRID = np.arange(100, 5001, 100, dtype=np.int64)


def _git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _legacy_pool_global(per_seed_df: pd.DataFrame) -> pd.DataFrame:
    """Compute the LEGACY aggregated_curves.csv (no dataset column).

    Schema: acquisition, kind, t, bias_pooled, MC_SE_pooled, SD_between, n_cells.
    Pools across all 30 cells globally — preserves §12 spot-check audit chain.
    """
    rows = []
    for kind, rhat_col in (
        ("lure", "R_hat_lure"), ("unweighted", "R_hat_unweighted"),
    ):
        df_k = per_seed_df.assign(_err=per_seed_df[rhat_col] - per_seed_df["R_N"])
        per_cell = (
            df_k.groupby(["acquisition", "cell", "t"])["_err"]
            .agg(_mean="mean", _sd=lambda v: v.std(ddof=1), _n="count")
            .reset_index()
        )
        per_cell["_se"] = per_cell["_sd"] / np.sqrt(per_cell["_n"])
        for (acq, t), grp in per_cell.groupby(["acquisition", "t"]):
            n_c = len(grp)
            rows.append({
                "acquisition": acq, "kind": kind, "t": int(t),
                "bias_pooled":  float(grp["_mean"].mean()),
                "MC_SE_pooled": float(np.sqrt((grp["_se"] ** 2).sum() / n_c ** 2)),
                "SD_between":   float(grp["_mean"].std(ddof=1)) if n_c > 1 else 0.0,
                "n_cells": n_c,
            })
    return pd.DataFrame(rows)


def _render_pooled(per_dataset_df: pd.DataFrame, out_path: Path) -> None:
    """Render rq4 pooled.pdf — 3 dataset rows, 4 curves per panel."""
    fig, axes = plt.subplots(
        3, 1, figsize=(3.25, 3.8), sharex=True, sharey=False,
    )
    for ax, dataset in zip(axes, DATASET_ORDER):
        sub = per_dataset_df[per_dataset_df["dataset"] == dataset]
        panel_lo = float("inf")
        panel_hi = -float("inf")
        for acq in ("ada", "oracle_accuracy"):
            for kind in ("lure", "unweighted"):
                row = sub[(sub["acquisition"] == acq) & (sub["kind"] == kind)]
                if len(row) == 0:
                    continue
                row = row.sort_values("t")
                color = ACQUISITION_STYLE[acq]["color"]
                ls = KIND_LINESTYLE[kind]["linestyle"]
                lw = KIND_LINESTYLE[kind]["linewidth"]
                label = f"{ACQUISITION_STYLE[acq]['label']} {KIND_LINESTYLE[kind]['label']}"
                ax.plot(row["t"], row["bias_pooled"],
                        color=color, ls=ls, lw=lw, label=label)
                if kind == "lure" and bands_enabled():
                    ax.fill_between(
                        row["t"],
                        row["bias_pooled"] - 2 * row["MC_SE_pooled"],
                        row["bias_pooled"] + 2 * row["MC_SE_pooled"],
                        color=color, alpha=BAND_ALPHA, linewidth=0,
                    )
                # Track the panel envelope for adaptive ylim.
                if kind == "lure":
                    band = 2 * row["MC_SE_pooled"]
                    panel_lo = min(panel_lo, float((row["bias_pooled"] - band).min()))
                    panel_hi = max(panel_hi, float((row["bias_pooled"] + band).max()))
                else:
                    panel_lo = min(panel_lo, float(row["bias_pooled"].min()))
                    panel_hi = max(panel_hi, float(row["bias_pooled"].max()))
        # Per-dataset adaptive ylim: include 0 (zero line) + 12% padding.
        if panel_hi > panel_lo:
            panel_lo = min(panel_lo, 0.0)
            panel_hi = max(panel_hi, 0.0)
            pad = (panel_hi - panel_lo) * 0.12
            ax.set_ylim(panel_lo - pad, panel_hi + pad)
        finalize_panel(
            ax, dataset=dataset, ylabel=r"bias  $\hat R_t - R_N$",
            show_zero_line=True,
            xlabel=("labels $t$" if dataset == DATASET_ORDER[-1] else None),
        )
        ax.yaxis.labelpad = 2
    axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, 1.30),
                   ncol=2, fontsize=6)
    fig.tight_layout(rect=(0, 0, 1, 0.99), h_pad=0.2)
    fig.savefig(out_path)
    plt.close(fig)


def _render_showcase(per_seed_df: pd.DataFrame, out_path: Path,
                     acquisition: str = "ada",
                     pair_defs: list[dict] | None = None) -> None:
    """Render rq4 showcase.pdf — 3 dataset rows × N demo pairs overlaid (LURE only).

    Honors `SAVE_PLOT_PAIRS` env var via `get_runtime_pairs()` to allow
    `regen_plots_only.py --pairs` to restrict the rendered slots.

    `pair_defs` overrides the showcase pair set (used by --paper-pairs to swap
    the legacy 3-pair `PAIR_DEFS` for the 4 v0502 paper_pairs).
    """
    pairs = pair_defs if pair_defs is not None else get_runtime_pairs()
    pair_filtered = per_pair_filter(per_seed_df, pairs)
    pair_filtered = pair_filtered[pair_filtered["acquisition"] == acquisition]

    # Per-cell bias for LURE (one row per cell × t)
    df_k = pair_filtered.assign(
        _err=pair_filtered["R_hat_lure"] - pair_filtered["R_N"]
    )
    per_cell = (
        df_k.groupby(["dataset", "pair_slot", "pair_label", "pair_color", "t"])["_err"]
        .agg(bias="mean", sd=lambda v: v.std(ddof=1), n="count")
        .reset_index()
    )
    per_cell["se"] = per_cell["sd"] / np.sqrt(per_cell["n"])

    fig, axes = plt.subplots(
        3, 1, figsize=(3.25, 3.8), sharex=True, sharey=False,
    )
    for ax, dataset in zip(axes, DATASET_ORDER):
        sub = per_cell[per_cell["dataset"] == dataset]
        panel_lo = float("inf")
        panel_hi = -float("inf")
        for pair_def in pairs:
            pair_rows = sub[sub["pair_slot"] == pair_def["slot"]].sort_values("t")
            if len(pair_rows) == 0:
                continue
            ax.plot(pair_rows["t"], pair_rows["bias"],
                    color=pair_def["color"], lw=1.4, ls="-",
                    label=pair_def["label"])
            band = 2 * pair_rows["se"]
            if bands_enabled():
                ax.fill_between(
                    pair_rows["t"],
                    pair_rows["bias"] - band,
                    pair_rows["bias"] + band,
                    color=pair_def["color"], alpha=BAND_ALPHA, linewidth=0,
                )
            panel_lo = min(panel_lo, float((pair_rows["bias"] - band).min()))
            panel_hi = max(panel_hi, float((pair_rows["bias"] + band).max()))
        if panel_hi > panel_lo:
            panel_lo = min(panel_lo, 0.0)
            panel_hi = max(panel_hi, 0.0)
            pad = (panel_hi - panel_lo) * 0.12
            ax.set_ylim(panel_lo - pad, panel_hi + pad)
        finalize_panel(
            ax, dataset=dataset, ylabel=r"bias  $\hat R_t - R_N$",
            show_zero_line=True,
            xlabel=("labels $t$" if dataset == DATASET_ORDER[-1] else None),
        )
        ax.yaxis.labelpad = 2
    axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, 1.30),
                   ncol=1, fontsize=6)
    fig.tight_layout(rect=(0, 0, 1, 0.99), h_pad=0.2)
    fig.savefig(out_path)
    plt.close(fig)


def _rename_stale_per_cell(out_dir: Path) -> None:
    """G5/H6 fix: rename stale per_cell.pdf → per_cell.pdf.deprecated."""
    stale = out_dir / "per_cell.pdf"
    if stale.exists():
        target = out_dir / "per_cell.pdf.deprecated"
        stale.rename(target)
        logger.info("renamed stale %s -> %s", stale, target)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--cells-root", type=Path, required=True)
    parser.add_argument("--oracle-cells", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)  # back-compat, ignored
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--acquisition-showcase", choices=("ada", "oracle_accuracy"),
                        default="ada", help="Which acquisition to render in showcase.pdf")
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
    _rename_stale_per_cell(args.out)

    # Read existing per_seed_curves.csv (this script no longer recomputes it).
    per_seed_path = args.out / "per_seed_curves.csv"
    if not per_seed_path.exists():
        sys.exit(f"per_seed_curves.csv missing at {per_seed_path}; run the data-layer SLURM job first")

    per_seed_df = pd.read_csv(per_seed_path)
    logger.info("loaded %d rows from %s", len(per_seed_df), per_seed_path)

    if args.paper_pairs:
        from save.paper_experiment.config import (
            default_config_path as _default_config_path,
            load_config as _load_config,
        )
        _cfg = _load_config(_default_config_path())
        per_seed_df = filter_per_seed_by_paper_pairs(per_seed_df, _cfg.paper_pair_keys)
        logger.info("paper_pairs filter: %d rows remaining", len(per_seed_df))

    # Aggregate
    t0 = time.time()
    per_dataset_df = per_dataset_pool(
        per_seed_df,
        section="rq4",
        value_spec={
            "rhat_cols": {"lure": "R_hat_lure", "unweighted": "R_hat_unweighted"},
            "rn_col": "R_N",
        },
    )
    legacy_df = _legacy_pool_global(per_seed_df)
    aggregate_s = time.time() - t0

    # CSVs (H3: float_format='%.18g' for byte-identity with legacy)
    per_dataset_df.to_csv(args.out / "per_dataset_curves.csv",
                          index=False, float_format="%.18g")
    legacy_df.to_csv(args.out / "aggregated_curves.csv",
                     index=False, float_format="%.18g")

    # PDFs
    _render_pooled(per_dataset_df, args.out / "pooled.pdf")
    showcase_pair_defs = None
    if args.paper_pairs:
        from scripts.paper_experiment.plot_style import paper_pair_defs as _pp_defs
        showcase_pair_defs = _pp_defs(_cfg.paper_pairs)
    _render_showcase(per_seed_df, args.out / "showcase.pdf",
                     acquisition=args.acquisition_showcase,
                     pair_defs=showcase_pair_defs)

    # metadata.json
    meta = {
        "git_commit": _git_hash(),
        "section": "rq4",
        "n_seeds_per_cell": int(per_seed_df.groupby("cell")["seed"].nunique().mode().iloc[0]),
        "n_cells": int(per_seed_df["cell"].nunique()),
        "datasets": list(DATASET_ORDER),
        "t_grid": T_GRID.tolist(),
        "showcase_acquisition": args.acquisition_showcase,
        "timing_s": {"aggregate": aggregate_s},
        "hostname": socket.gethostname(),
    }
    (args.out / "metadata.json").write_text(json.dumps(meta, indent=2, default=str))
    logger.info("wrote outputs to %s", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
