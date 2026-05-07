#!/usr/bin/env python
"""rq6 — Variance figure restricted to SST-2 + MMLU, restyled to match the
visual conventions of plot_rq4_rq5_combined.py (CELEUS naming; thicker lines;
full-page fonts) but laid out as 2 metric-rows × 2 dataset-cols with tight
margins for a concise figure.

Reads the existing per_dataset_curves.csv produced by plot_rq6_variance.py;
no SLURM trajectory replay needed. Output goes to a fresh directory so the
original rq6-variance/ artefacts are preserved.

Default I/O (paper_experiments_v0502 scope):
  in:  results/paper_experiments_v0502/rq6-variance/accuracy/per_dataset_curves.csv
  out: results/paper_experiments_v0502/rq6-variance-new/accuracy/pooled.pdf
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
    ACQUISITION_STYLE, BAND_ALPHA,
    apply_rc_helvetica, bands_enabled, finalize_panel,
)

_DATASETS = ("sst2", "mmlu")
_DATASET_DISPLAY = {"sst2": "SST-2", "mmlu": "MMLU"}
_X_CAP = 4000

# CELEUS naming + deeper Oracle orange, mirroring plot_rq4_rq5_combined.py.
_ACQ_LABEL = {
    "ada":             "CELEUS",
    "oracle_accuracy": "Oracle",
    "uniform":         "Uniform",
}
_ACQ_COLOR = {
    "ada":             ACQUISITION_STYLE["ada"]["color"],   # indigo
    "oracle_accuracy": "#D17B16",                            # deeper orange
    "uniform":         "#4C566A",                            # slate
}
_ACQ_LS = {"ada": "-", "oracle_accuracy": "-", "uniform": "--"}
_ACQ_LW = {"ada": 2.4, "oracle_accuracy": 2.4, "uniform": 2.0}

# Match _FONT_SIZES in plot_rq4_rq5_combined.py for cross-figure parity.
_FONT_SIZES = {
    "font.size": 12,
    "axes.titlesize": 16,
    "axes.labelsize": 13,
    "legend.fontsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
}

# Metric per row: row 0 = (a) cond_var_S; row 1 = (b) emp_var_R.
_METRICS = (
    {
        "key": "mean_cond_var_S",
        "se_key": "se_cond_var_S",
        "use_se_band": True,   # ±2·MC-SE
        "row_label": r"(a) $\mathrm{Var}(\hat{S}_t \mid \mathcal{F}_{t-1})$",
        "yscale": "linear",
    },
    {
        "key": "mean_emp_var_R",
        "se_key": "sd_between_emp_var_R",
        "use_se_band": False,  # ±1·SD-between (sample-variance MC-SE misleading)
        "row_label": r"(b) $\mathrm{Var}(\hat R_t)$",
        "yscale": "log",
    },
)


def render_combined(per_dataset_df: pd.DataFrame, out_path: Path) -> None:
    apply_rc_helvetica()
    plt.rcParams.update(_FONT_SIZES)
    plt.rcParams["legend.frameon"] = False

    fig, axes = plt.subplots(
        len(_METRICS), len(_DATASETS),
        figsize=(7.8, 4.6),
        sharex="col", sharey=False,
    )
    for row_idx, metric in enumerate(_METRICS):
        for col_idx, dataset in enumerate(_DATASETS):
            ax = axes[row_idx][col_idx]
            sub = per_dataset_df[per_dataset_df["dataset"] == dataset]
            sub = sub[sub["t"] <= _X_CAP]
            panel_lo, panel_hi = float("inf"), -float("inf")
            for acq in ("ada", "oracle_accuracy", "uniform"):
                rows = sub[sub["acquisition"] == acq].sort_values("t")
                if len(rows) == 0:
                    continue
                ax.plot(
                    rows["t"], rows[metric["key"]],
                    color=_ACQ_COLOR[acq],
                    ls=_ACQ_LS[acq],
                    lw=_ACQ_LW[acq],
                    label=_ACQ_LABEL[acq],
                )
                band = (2 * rows[metric["se_key"]] if metric["use_se_band"]
                        else rows[metric["se_key"]])
                if bands_enabled():
                    ax.fill_between(
                        rows["t"],
                        rows[metric["key"]] - band,
                        rows[metric["key"]] + band,
                        color=_ACQ_COLOR[acq],
                        alpha=BAND_ALPHA, linewidth=0,
                    )
                panel_lo = min(panel_lo, float((rows[metric["key"]] - band).min()))
                panel_hi = max(panel_hi, float((rows[metric["key"]] + band).max()))
            ax.set_yscale(metric["yscale"])
            if metric["yscale"] == "linear" and panel_hi > panel_lo:
                rng = panel_hi - panel_lo
                ax.set_ylim(max(0.0, panel_lo - rng * 0.08), panel_hi + rng * 0.10)
            ax.set_xlim(500, _X_CAP)
            finalize_panel(
                ax,
                dataset="",
                xlabel=("labels $t$" if row_idx == len(_METRICS) - 1 else None),
                ylabel=(metric["row_label"] if col_idx == 0 else None),
            )
            ax.yaxis.labelpad = 3
            if row_idx == 0:
                ax.set_title(_DATASET_DISPLAY[dataset], pad=3)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="center", bbox_to_anchor=(0.5, 0.965),
        ncol=len(labels), fontsize=12,
        frameon=False, columnspacing=1.6, handlelength=2.0, handletextpad=0.6,
    )
    # Tight margins: ylabels carry the row identity, column titles carry the
    # dataset, the shared legend hugs the column titles, x-labels only on
    # bottom row, x-ticks shared across rows.
    fig.subplots_adjust(
        top=0.88, bottom=0.10, left=0.10, right=0.995,
        hspace=0.10, wspace=0.20,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=Path,
        default=_REPO / "results" / "paper_experiments_v0502" /
                "rq6-variance" / "accuracy" / "per_dataset_curves.csv",
        help="rq6 per_dataset_curves.csv (defaults to v0502 accuracy results)",
    )
    parser.add_argument(
        "--out", type=Path,
        default=_REPO / "results" / "paper_experiments_v0502" /
                "rq6-variance-new" / "accuracy" / "pooled.pdf",
        help="output PDF path",
    )
    args = parser.parse_args(argv)
    df = pd.read_csv(args.input)
    df = df[df["dataset"].isin(_DATASETS)].copy()
    if df.empty:
        sys.exit(f"no rows for datasets {_DATASETS} in {args.input}")
    render_combined(df, args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
