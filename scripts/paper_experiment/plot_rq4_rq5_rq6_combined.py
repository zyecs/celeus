#!/usr/bin/env python
"""Combine rq4 (bias), rq5 (signal MSE), and rq6 (conditional variance only)
pooled curves into a single 3-row × 3-col figure.

Layout: 3 dataset rows × 3 metric cols.
  - Col (a): rq4 bias  (R_hat_t - R_N).
              Curves: ada-LURE, ada-unweighted, oracle-LURE.
  - Col (b): rq5 signal MSE (log y).
              Curves: IS-corrected, naive.
  - Col (c): rq6 conditional variance Var(S_hat_t | F_{t-1}).
              Curves: ada, oracle, uniform.
              (Empirical Var(R_hat_t) is intentionally excluded.)

Bands show **mean ± 1·SD-between** (standard deviation across cells), matching
the user's request for mean ± std rather than the ±2·MC-SE bands used by the
2-column rq4-rq5 figure. SD columns:
  rq4: SD_between
  rq5: SD_between_is, SD_between_naive
  rq6: sd_between_cond_var_S

Reads existing per_dataset_curves.csv produced by the rq4/rq5/rq6 plotters;
no SLURM trajectory replay needed.

Usage from project root:
  PYTHONPATH=. python scripts/paper_experiment/plot_rq4_rq5_rq6_combined.py \
    --out-root results/paper_experiments_v0504
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

_DATASET_DISPLAY = {"sst2": "SST-2", "mmlu": "MMLU", "agnews": "AG News"}

_ACQ_LABEL = {"ada": "CELEUS", "oracle_accuracy": "Oracle", "uniform": "Uniform"}

_CURVE_LABEL = {
    ("ada", "lure"):                  "LURE-weighted",
    ("ada", "unweighted"):            "Unweighted",
    ("oracle_accuracy", "lure"):      "Oracle",
}

_ACQ_COLOR = {
    "ada":             ACQUISITION_STYLE["ada"]["color"],
    "oracle_accuracy": "#D17B16",
    "uniform":         "#4C566A",
}
_ACQ_LS = {"ada": "-", "oracle_accuracy": "-", "uniform": "--"}
_ACQ_LW = {"ada": 2.4, "oracle_accuracy": 2.4, "uniform": 2.0}

_FONT_SIZES = {
    "font.size": 12,
    "axes.titlesize": 16,
    "axes.labelsize": 13,
    "legend.fontsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
}


def _render_rq4_panel(ax, sub: pd.DataFrame) -> tuple[float, float]:
    """rq4 bias panel with mean ± 1·SD_between bands on LURE curves."""
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
        lw = 2.4 if kind == "lure" else 2.0
        label = _CURVE_LABEL[(acq, kind)]
        ax.plot(row["t"], row["bias_pooled"],
                color=color, ls=ls, lw=lw, label=label)
        if kind == "lure":
            band = row["SD_between"]
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
    """rq5 signal-MSE panel with mean ± 1·SD_between bands (log y)."""
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
                        sub["is_mse_pooled"] - sub["SD_between_is"],
                        sub["is_mse_pooled"] + sub["SD_between_is"],
                        color=is_color, alpha=BAND_ALPHA, linewidth=0)
    ax.plot(sub["t"], sub["naive_mse_pooled"],
            color=naive_color,
            ls=KIND_LINESTYLE["naive"]["linestyle"],
            lw=2.0,
            label=KIND_LINESTYLE["naive"]["label"])
    if bands_enabled():
        ax.fill_between(sub["t"],
                        sub["naive_mse_pooled"] - sub["SD_between_naive"],
                        sub["naive_mse_pooled"] + sub["SD_between_naive"],
                        color=naive_color, alpha=BAND_ALPHA, linewidth=0)
    ax.set_yscale("log")


def _render_rq6_panel(ax, sub: pd.DataFrame, x_lo: float, x_hi: float) -> None:
    """rq6 conditional-variance panel: mean curves only (no SD band).

    Plots the three acquisitions (ada / oracle / uniform). Uses the same
    x-clip as the other columns when possible; rq6 starts at t=500 so the
    visible range is [max(x_lo, 500), x_hi]."""
    panel_lo, panel_hi = float("inf"), -float("inf")
    for acq in ("ada", "oracle_accuracy", "uniform"):
        rows = sub[sub["acquisition"] == acq].sort_values("t")
        if len(rows) == 0:
            continue
        ax.plot(
            rows["t"], rows["mean_cond_var_S"],
            color=_ACQ_COLOR[acq],
            ls=_ACQ_LS[acq],
            lw=_ACQ_LW[acq],
            label=_ACQ_LABEL[acq],
        )
        panel_lo = min(panel_lo, float(rows["mean_cond_var_S"].min()))
        panel_hi = max(panel_hi, float(rows["mean_cond_var_S"].max()))
    if panel_hi > panel_lo:
        rng = panel_hi - panel_lo
        ax.set_ylim(max(0.0, panel_lo - rng * 0.08), panel_hi + rng * 0.10)
    ax.set_xlim(max(x_lo, 500), x_hi)


def render_combined(
    rq4_df: pd.DataFrame,
    rq5_df: pd.DataFrame,
    rq6_df: pd.DataFrame,
    out_path: Path,
) -> None:
    apply_rc_helvetica()
    plt.rcParams.update(_FONT_SIZES)
    plt.rcParams["legend.frameon"] = True

    x_lo, x_hi = 0, 4000
    fig, axes = plt.subplots(
        3, 3, figsize=(16.0, 7.6),
        sharex=False, sharey=False,
    )
    for row_idx, dataset in enumerate(DATASET_ORDER):
        # Col 0 — rq4 bias.
        ax_a = axes[row_idx][0]
        sub_a = rq4_df[rq4_df["dataset"] == dataset]
        panel_lo, panel_hi = _render_rq4_panel(ax_a, sub_a)
        if panel_hi > panel_lo:
            panel_lo = min(panel_lo, 0.0)
            panel_hi = max(panel_hi, 0.0)
            pad = (panel_hi - panel_lo) * 0.12
            ax_a.set_ylim(panel_lo - pad, panel_hi + pad)
        finalize_panel(
            ax_a,
            dataset="",
            ylabel=(r"bias  $\hat R_t - R_N$" if row_idx == 1 else None),
            show_zero_line=True,
            xlabel=("Evaluated Samples" if row_idx == 2 else None),
            dataset_pos="topleft",
        )
        ax_a.set_xlim(x_lo, x_hi)
        ax_a.yaxis.labelpad = 4
        ax_a.text(
            0.02, 0.93, _DATASET_DISPLAY[dataset],
            transform=ax_a.transAxes,
            fontsize=13, fontweight="bold", va="top", ha="left",
        )

        # Col 1 — rq5 signal MSE.
        ax_b = axes[row_idx][1]
        sub_b = rq5_df[rq5_df["dataset"] == dataset]
        _render_rq5_panel(ax_b, sub_b)
        finalize_panel(
            ax_b,
            dataset="",
            ylabel=(r"$\hat S_t$ MSE" if row_idx == 1 else None),
            xlabel=("Evaluated Samples" if row_idx == 2 else None),
        )
        ax_b.set_xlim(x_lo, x_hi)
        ax_b.yaxis.labelpad = 4

        # Col 2 — rq6 conditional variance.
        ax_c = axes[row_idx][2]
        sub_c = rq6_df[rq6_df["dataset"] == dataset]
        _render_rq6_panel(ax_c, sub_c, x_lo=x_lo, x_hi=x_hi)
        finalize_panel(
            ax_c,
            dataset="",
            ylabel=None,
            xlabel=("Evaluated Samples" if row_idx == 2 else None),
        )
        ax_c.yaxis.labelpad = 4

        if row_idx == 0:
            ax_a.set_title("(a) Bias", pad=4)
            ax_b.set_title("(b) Inferential Signal MSE", pad=4)
            ax_c.set_title(
                r"(c) Conditional Variance  "
                r"$\mathrm{Var}(\hat S_t \mid \mathcal{F}_{t-1})$",
                pad=4,
            )

    legend_common = dict(frameon=False, columnspacing=1.4)
    handles_a, labels_a = axes[0][0].get_legend_handles_labels()
    handles_b, labels_b = axes[0][1].get_legend_handles_labels()
    handles_c, labels_c = axes[0][2].get_legend_handles_labels()
    fig.legend(
        handles_a, labels_a,
        loc="center", bbox_to_anchor=(0.19, 0.94),
        ncol=len(labels_a), fontsize=14, **legend_common,
    )
    fig.legend(
        handles_b, labels_b,
        loc="center", bbox_to_anchor=(0.52, 0.94),
        ncol=len(labels_b), fontsize=14, **legend_common,
    )
    fig.legend(
        handles_c, labels_c,
        loc="center", bbox_to_anchor=(0.85, 0.94),
        ncol=len(labels_c), fontsize=14, **legend_common,
    )
    fig.subplots_adjust(
        top=0.86, bottom=0.08, left=0.05, right=0.995, hspace=0.16, wspace=0.18,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rq4", type=Path, default=None,
        help="rq4 per_dataset_curves.csv (defaults to <out-root>/rq4-unbiasedness/<loss>/per_dataset_curves.csv)",
    )
    parser.add_argument(
        "--rq5", type=Path, default=None,
        help="rq5 per_dataset_curves.csv (defaults to <out-root>/rq5-signal-mse/<loss>/per_dataset_curves.csv)",
    )
    parser.add_argument(
        "--rq6", type=Path, default=None,
        help="rq6 per_dataset_curves.csv (defaults to <out-root>/rq6-variance/<loss>/per_dataset_curves.csv)",
    )
    parser.add_argument(
        "--loss", choices=("accuracy", "cross_entropy"), default="accuracy",
        help="loss subdirectory to read (default: accuracy)",
    )
    parser.add_argument(
        "--out", type=Path, default=None,
        help="output PDF path (defaults to <out-root>/rq4-rq5-rq6-combined/combined.pdf)",
    )
    parser.add_argument(
        "--out-root", type=Path,
        default=_REPO / "results" / "paper_experiments_v0504",
        help="results root for input/output paths.",
    )
    args = parser.parse_args(argv)
    out_root = args.out_root
    rq4_path = args.rq4 or out_root / "rq4-unbiasedness" / args.loss / "per_dataset_curves.csv"
    rq5_path = args.rq5 or out_root / "rq5-signal-mse" / args.loss / "per_dataset_curves.csv"
    rq6_path = args.rq6 or out_root / "rq6-variance" / args.loss / "per_dataset_curves.csv"
    out_path = args.out or out_root / "rq4-rq5-rq6-combined" / "combined.pdf"

    rq4_df = pd.read_csv(rq4_path)
    rq5_df = pd.read_csv(rq5_path)
    rq6_df = pd.read_csv(rq6_path)
    render_combined(rq4_df, rq5_df, rq6_df, out_path)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
