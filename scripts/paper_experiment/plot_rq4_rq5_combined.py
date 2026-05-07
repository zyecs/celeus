#!/usr/bin/env python
"""Combine rq4 (bias) and rq5 (signal MSE) pooled curves into one figure.

Layout: 3 dataset rows × 2 metric cols.
  - Left column:  rq4 bias  (R_hat_t - R_N) with ±2·MC_SE bands on LURE.
                  Curves: ada-LURE, ada-unweighted, oracle-LURE, oracle-unweighted.
  - Right column: rq5 signal MSE on log y-axis with ±2·MC_SE bands.
                  Curves: IS-corrected, naive.

Reads existing per_dataset_curves.csv produced by plot_rq4_unbiasedness.py
and plot_rq5_signal_mse.py — no SLURM trajectory replay needed.

Usage from project root:
  PYTHONPATH=. python scripts/paper_experiment/plot_rq4_rq5_combined.py \
    [--rq4 PATH] [--rq5 PATH] [--out PATH] [--loss accuracy|cross_entropy]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from scripts.paper_experiment.plot_style import (
    ACQUISITION_STYLE, BAND_ALPHA, DATASET_ORDER, KIND_LINESTYLE, TOL_MUTED,
    apply_rc_helvetica, bands_enabled, finalize_panel,
)

# Display names for dataset-corner annotations (consistent with the RQ1 plot).
_DATASET_DISPLAY = {"sst2": "SST-2", "mmlu": "MMLU", "agnews": "AG News"}

# Local override: this figure presents the method as ``CELEUS`` (matching the
# RQ1 figure / paper text). The shared ``ACQUISITION_STYLE`` keeps "ADA (ours)"
# for legacy plots, so we override only on the path that builds these labels.
_ACQ_LABEL = {"ada": "CELEUS", "oracle_accuracy": "Oracle"}

# Per-(acq, kind) legend label override. Concise: "LURE-weighted" /
# "Unweighted" for CELEUS variants, "Oracle" for the oracle reference.
_CURVE_LABEL = {
    ("ada", "lure"):                  "LURE-weighted",
    ("ada", "unweighted"):            "Unweighted",
    ("oracle_accuracy", "lure"):      "Oracle",
}

# Local override: bump the Oracle color from the low-saturation Tol "sand" to
# a deeper, more visible orange for contrast against CELEUS's indigo.
_ACQ_COLOR = {
    "ada":             ACQUISITION_STYLE["ada"]["color"],
    "oracle_accuracy": "#D17B16",
}

# Font sizes — matched to the RQ1 new variant for visual consistency on the
# same paper page. Tuned slightly smaller than RQ1 (3×2 grid vs 1×3 grid).
_FONT_SIZES = {
    "font.size": 12,
    "axes.titlesize": 16,
    "axes.labelsize": 13,
    "legend.fontsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
}


def _render_rq4_panel(ax, sub: pd.DataFrame) -> tuple[float, float]:
    """Plot the rq4 curves on `ax`. Returns (lo, hi) of the rendered envelope.

    Curves: ada-LURE, ada-Unweighted, oracle-LURE. Oracle-Unweighted is
    intentionally omitted — it duplicates the visual of ada-Unweighted and
    crowds the bias panel without adding inferential value (the oracle's
    contribution is fully captured by oracle-LURE)."""
    panel_lo = float("inf")
    panel_hi = -float("inf")
    _CURVES = (("ada", "lure"), ("ada", "unweighted"), ("oracle_accuracy", "lure"))
    for acq, kind in _CURVES:
        row = sub[(sub["acquisition"] == acq) & (sub["kind"] == kind)]
        if len(row) == 0:
            continue
        row = row.sort_values("t")
        color = _ACQ_COLOR[acq]
        ls = KIND_LINESTYLE[kind]["linestyle"]
        # Thickened relative to the shared KIND_LINESTYLE defaults (1.2-1.4)
        # for readability at the new figure size.
        lw = 2.4 if kind == "lure" else 2.0
        label = _CURVE_LABEL[(acq, kind)]
        ax.plot(row["t"], row["bias_pooled"],
                color=color, ls=ls, lw=lw, label=label)
        if kind == "lure":
            band = 2 * row["MC_SE_pooled"]
            if bands_enabled():
                ax.fill_between(
                    row["t"],
                    row["bias_pooled"] - band,
                    row["bias_pooled"] + band,
                    color=color, alpha=BAND_ALPHA, linewidth=0,
                )
            panel_lo = min(panel_lo, float((row["bias_pooled"] - band).min()))
            panel_hi = max(panel_hi, float((row["bias_pooled"] + band).max()))
        else:
            panel_lo = min(panel_lo, float(row["bias_pooled"].min()))
            panel_hi = max(panel_hi, float(row["bias_pooled"].max()))
    return panel_lo, panel_hi


def _render_rq5_panel(ax, sub: pd.DataFrame) -> None:
    """Plot the 2 rq5 curves on `ax` (log y; no envelope tracking needed)."""
    sub = sub.sort_values("t")
    is_color = TOL_MUTED["indigo"]
    naive_color = TOL_MUTED["wine"]
    ax.plot(sub["t"], sub["is_mse_pooled"],
            color=is_color,
            ls=KIND_LINESTYLE["is_corrected"]["linestyle"],
            lw=2.4,
            label=KIND_LINESTYLE["is_corrected"]["label"])
    if bands_enabled():
        ax.fill_between(sub["t"],
                        sub["is_mse_pooled"] - 2 * sub["is_se_pooled"],
                        sub["is_mse_pooled"] + 2 * sub["is_se_pooled"],
                        color=is_color, alpha=BAND_ALPHA, linewidth=0)
    ax.plot(sub["t"], sub["naive_mse_pooled"],
            color=naive_color,
            ls=KIND_LINESTYLE["naive"]["linestyle"],
            lw=2.0,
            label=KIND_LINESTYLE["naive"]["label"])
    if bands_enabled():
        ax.fill_between(sub["t"],
                        sub["naive_mse_pooled"] - 2 * sub["naive_se_pooled"],
                        sub["naive_mse_pooled"] + 2 * sub["naive_se_pooled"],
                        color=naive_color, alpha=BAND_ALPHA, linewidth=0)
    ax.set_yscale("log")


def render_combined(rq4_df: pd.DataFrame, rq5_df: pd.DataFrame, out_path: Path) -> None:
    apply_rc_helvetica()
    plt.rcParams.update(_FONT_SIZES)
    # Override the shared "no legend frame" default so this figure's per-panel
    # legends actually render their boxes.
    plt.rcParams["legend.frameon"] = True
    # 3 rows × 2 cols, matched to the RQ1 figure's full-page width (13.2 in).
    fig, axes = plt.subplots(
        3, 2, figsize=(13.2, 7.2),
        sharex="col", sharey=False,
    )
    for row_idx, dataset in enumerate(DATASET_ORDER):
        # Left column — rq4 bias.
        ax_l = axes[row_idx][0]
        sub_l = rq4_df[rq4_df["dataset"] == dataset]
        panel_lo, panel_hi = _render_rq4_panel(ax_l, sub_l)
        if panel_hi > panel_lo:
            panel_lo = min(panel_lo, 0.0)
            panel_hi = max(panel_hi, 0.0)
            pad = (panel_hi - panel_lo) * 0.12
            ax_l.set_ylim(panel_lo - pad, panel_hi + pad)
        finalize_panel(
            ax_l,
            dataset="",  # corner annotation done manually below at larger size
            ylabel=(r"bias  $\hat R_t - R_N$" if row_idx == 1 else None),
            show_zero_line=True,
            xlabel=("labels $t$" if row_idx == 2 else None),
            dataset_pos="topleft",
        )
        ax_l.yaxis.labelpad = 4
        ax_l.text(
            0.02, 0.93, _DATASET_DISPLAY[dataset],
            transform=ax_l.transAxes,
            fontsize=13, fontweight="bold", va="top", ha="left",
        )

        # Right column — rq5 signal MSE.
        ax_r = axes[row_idx][1]
        sub_r = rq5_df[rq5_df["dataset"] == dataset]
        _render_rq5_panel(ax_r, sub_r)
        finalize_panel(
            ax_r,
            dataset="",  # right column relies on left-column annotation per row
            ylabel=(r"$\hat S_t$ MSE" if row_idx == 1 else None),
            xlabel=("labels $t$" if row_idx == 2 else None),
        )
        ax_r.yaxis.labelpad = 4

        # Cap x-axis at 3500 on every panel; sharex='col' propagates to row 0.
        ax_l.set_xlim(0, 3500)
        ax_r.set_xlim(0, 3500)

        if row_idx == 0:
            ax_l.set_title("(a) Bias", pad=4)
            ax_r.set_title("(b) Inferential Signal MSE", pad=4)

    # Two separate legends, one per column, in a soft-edged white box that
    # sits flush against the column title (smaller gap than the prior layout).
    # Frameless legends — no border, no shadow. Just text on the figure
    # background. Right-column legend is bumped a bit larger so the only-two
    # entries don't read as visually under-weighted relative to the left.
    legend_common = dict(frameon=False, columnspacing=1.4)
    handles_l, labels_l = axes[0][0].get_legend_handles_labels()
    handles_r, labels_r = axes[0][1].get_legend_handles_labels()
    # ``loc="center"`` anchors at vertical centre, so legend boxes with
    # different content heights (the right legend has math hats / subscripts
    # that extend further than the left's plain text) sit at the same visual
    # midline rather than top-aligned with mismatched baselines.
    fig.legend(
        handles_l, labels_l,
        loc="center", bbox_to_anchor=(0.28, 0.94),
        ncol=len(labels_l), fontsize=15, **legend_common,
    )
    fig.legend(
        handles_r, labels_r,
        loc="center", bbox_to_anchor=(0.78, 0.94),
        ncol=len(labels_r), fontsize=15, **legend_common,
    )
    # Tighter inter-panel margins; legend now hugs the column titles.
    fig.subplots_adjust(
        top=0.86, bottom=0.08, left=0.06, right=0.995, hspace=0.16, wspace=0.13,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rq4", type=Path,
        default=_REPO / "results" / "paper_experiment" / "rq4-unbiasedness" / "accuracy" / "per_dataset_curves.csv",
        help="rq4 per_dataset_curves.csv (defaults to accuracy results)",
    )
    parser.add_argument(
        "--rq5", type=Path,
        default=None,
        help="rq5 per_dataset_curves.csv (defaults to results/paper_experiment/rq5-signal-mse/<loss>/per_dataset_curves.csv)",
    )
    parser.add_argument(
        "--loss", choices=("accuracy", "cross_entropy"), default="accuracy",
        help="rq5 loss to plot (default: accuracy, matches rq4 scope)",
    )
    parser.add_argument(
        "--out", type=Path,
        default=_REPO / "results" / "paper_experiment" / "rq4-rq5-combined" / "combined.pdf",
        help="output PDF path",
    )
    parser.add_argument("--paper-pairs", action="store_true",
                        help="Filter cells to cfg.paper_pair_keys (v0502 scope). "
                             "Note: this script reads pre-aggregated per_dataset_curves.csv; "
                             "the filter applies to the upstream rq4/rq5 plotters.")
    parser.add_argument("--out-root", type=Path, default=None,
                        help="Override results root for input/output paths.")
    args = parser.parse_args(argv)
    out_root = args.out_root if args.out_root else _REPO / "results" / "paper_experiment"
    rq5_path = args.rq5 or (
        out_root / "rq5-signal-mse" / args.loss / "per_dataset_curves.csv"
    )
    # When --out-root is supplied, rewrite the rq4 default path too.
    rq4_path = args.rq4
    if args.out_root and rq4_path == _REPO / "results" / "paper_experiment" / "rq4-unbiasedness" / "accuracy" / "per_dataset_curves.csv":
        rq4_path = out_root / "rq4-unbiasedness" / "accuracy" / "per_dataset_curves.csv"
    out_path = args.out
    if args.out_root and args.out == _REPO / "results" / "paper_experiment" / "rq4-rq5-combined" / "combined.pdf":
        out_path = out_root / "rq4-rq5-combined" / "combined.pdf"
    rq4_df = pd.read_csv(rq4_path)
    rq5_df = pd.read_csv(rq5_path)
    render_combined(rq4_df, rq5_df, out_path)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
