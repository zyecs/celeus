#!/usr/bin/env python3
"""Per-(model-pair) confidence-interval-width trajectories, accuracy-only.

Mirrors the visual style of plot_rq1_efficiency_new.py but disaggregates the
per-dataset pool into a (paper_pair × dataset) grid so reviewers can verify
that CELEUS beats the baseline at every individual cell, not just on the
dataset mean.

Layout: 6 rows (paper_pairs in cfg.paper_pairs order) × 3 columns (datasets).
Each panel overlays {CELEUS / CELEUS w/o surr / Oracle / Baseline e-value}
mean width with ±std bands. Per-panel x_end = first label where CELEUS hits
ε, with the same fallback rule as the main figure.

Outputs:
    <output-root>/per_pair_efficiency/fig_per_pair_width.{pdf,png}
    <output-root>/per_pair_efficiency/per_pair_manifest.json
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

_TMP_CACHE = Path("/tmp/save-rq1-mpl")
os.environ.setdefault("MPLCONFIGDIR", str(_TMP_CACHE / "mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(_TMP_CACHE / "xdg-cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[2]
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from save.paper_experiment.config import default_config_path, load_config  # noqa: E402

import rq1_efficiency_common as common  # noqa: E402
import plot_rq1_efficiency as base  # noqa: E402
import plot_rq1_efficiency_new as new_mod  # noqa: E402


_DATASET_LABELS = new_mod.DATASET_LABELS
_METHOD_ORDER = ("M1", "M3", "ORACLE_ACC", "M4")
_METHOD_STYLE = new_mod.METHOD_STYLE
_EPSILON = new_mod.EPSILON
_Y_LIM = new_mod.Y_LIM
# Methods to mark with median-labels-to-ε markers on the ε hairline.
_MEDIAN_MARKER_METHODS = ("M1", "ORACLE_ACC")
_MEDIAN_MARKER_STYLE = new_mod._MEDIAN_MARKER_STYLE
# Pairs to skip in the per-pair grid (presentation knob; doesn't affect
# the v0504 paper_pair scope or other figures). Empty by default.
_EXCLUDED_PAIRS_FOR_GRID: set[tuple[str, str]] = set()
_FONT_SIZES = {
    "font.size": 13,
    "axes.titlesize": 16,
    "axes.labelsize": 13,
    "legend.fontsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
}


def _aggregate_per_pair(
    trajectories: list[common.SeedWidthTrajectory],
    *,
    methods: tuple[str, ...],
    pairs: tuple[common.SelectedPair, ...],
    datasets: tuple[str, ...],
) -> pd.DataFrame:
    """Per-(pair × dataset × method × labels_used) mean and std across seeds."""
    grouped: dict[tuple, list[common.SeedWidthTrajectory]] = {}
    for traj in trajectories:
        grouped.setdefault(
            (traj.dataset, traj.surrogate, traj.target, traj.method), []
        ).append(traj)

    rows: list[dict[str, object]] = []
    for (dataset, surr, tgt, method), items in grouped.items():
        if dataset not in datasets:
            continue
        if (surr, tgt) not in {(p.surrogate, p.target) for p in pairs}:
            continue
        if method not in methods:
            continue
        grid = np.array(
            sorted({int(label) for item in items for label in item.labels}),
            dtype=np.int64,
        )
        matrix = np.vstack(
            [
                base.right_continuous_resample(item.labels, item.widths, grid)
                for item in items
            ]
        )
        mean = matrix.mean(axis=0)
        std = matrix.std(axis=0, ddof=0)
        for label, m, s in zip(grid, mean, std):
            rows.append({
                "dataset": dataset,
                "surrogate": surr,
                "target": tgt,
                "method": method,
                "labels_used": int(label),
                "mean_width": float(m),
                "std_width": float(s),
                "n_units": int(matrix.shape[0]),
            })
    return pd.DataFrame(rows)


def _per_pair_median_labels_to_eps(
    trajectories: list[common.SeedWidthTrajectory],
    *,
    methods: tuple[str, ...] = _MEDIAN_MARKER_METHODS,
    pairs: tuple[common.SelectedPair, ...],
    datasets: tuple[str, ...],
) -> dict[tuple[str, str, str, str], int]:
    """Median labels-to-ε keyed by (dataset, surrogate, target, method).

    A trajectory's last label equals labels_to_stop (per ``common._truncate_to_stop``).
    Non-stoppers are excluded by ``load_selected_seed_trajectories``. Median is
    over the seed dimension only, separately per (pair, dataset, method).
    """
    pair_keys = {(p.surrogate, p.target) for p in pairs}
    by_key: dict[tuple[str, str, str, str], list[int]] = {}
    for traj in trajectories:
        if traj.method not in methods or traj.dataset not in datasets:
            continue
        if (traj.surrogate, traj.target) not in pair_keys:
            continue
        if traj.labels.size == 0:
            continue
        key = (traj.dataset, traj.surrogate, traj.target, traj.method)
        by_key.setdefault(key, []).append(int(traj.labels[-1]))
    return {
        k: int(np.median(np.asarray(stops, dtype=np.int64)))
        for k, stops in by_key.items() if stops
    }


def _panel_x_end(df: pd.DataFrame, dataset: str, surr: str, tgt: str,
                 *, epsilon: float = _EPSILON) -> tuple[int, bool]:
    """First labels-used where CELEUS (M1) mean drops to <=epsilon. Falls
    back to CELEUS's rightmost label if it never crosses."""
    rows = df.loc[
        (df["dataset"] == dataset) & (df["surrogate"] == surr)
        & (df["target"] == tgt) & (df["method"] == "M1"),
        ["labels_used", "mean_width"],
    ].sort_values("labels_used")
    if rows.empty:
        return 5000, False
    labels = rows["labels_used"].to_numpy(dtype=np.int64)
    means = rows["mean_width"].to_numpy(dtype=np.float64)
    mask = means <= epsilon
    if mask.any():
        return int(labels[int(np.argmax(mask))]), True
    return int(labels[-1]), False


def plot_per_pair_grid(
    df: pd.DataFrame,
    out_pdf: Path,
    out_png: Path,
    *,
    pairs: tuple[common.SelectedPair, ...],
    datasets: tuple[str, ...],
    methods: tuple[str, ...] = _METHOD_ORDER,
    median_labels: dict[tuple[str, str, str, str], int] | None = None,
) -> None:
    base.apply_rq1_nature_style()
    plt.rcParams.update(_FONT_SIZES)
    n_rows, n_cols = len(pairs), len(datasets)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4.0 * n_cols, 2.4 * n_rows),
        sharey=True, sharex=False,
    )
    if n_rows == 1:
        axes = np.atleast_2d(axes)

    for r, pair in enumerate(pairs):
        for c, dataset in enumerate(datasets):
            ax = axes[r][c]
            x_end, _crossed = _panel_x_end(df, dataset, pair.surrogate, pair.target)
            for method in methods:
                rows = df[
                    (df["dataset"] == dataset)
                    & (df["surrogate"] == pair.surrogate)
                    & (df["target"] == pair.target)
                    & (df["method"] == method)
                ].sort_values("labels_used")
                if rows.empty:
                    continue
                x = rows["labels_used"].to_numpy(dtype=np.int64)
                mean = rows["mean_width"].to_numpy(dtype=np.float64)
                std = rows["std_width"].to_numpy(dtype=np.float64)
                x, mean, std = base._insert_boundary_anchors(x, mean, std, x_end=x_end)
                style = _METHOD_STYLE[method]
                ax.plot(x, mean,
                        color=style["color"],
                        linewidth=1.7,
                        linestyle=style["linestyle"],
                        drawstyle="steps-post",
                        zorder=3)
                ax.fill_between(
                    x,
                    np.maximum(0.0, mean - std),
                    mean + std,
                    color=style["color"],
                    alpha=style["band_alpha"],
                    linewidth=0.0,
                    step="post",
                    zorder=2,
                )

            ax.axhline(_EPSILON, color="#8A8A8A", linewidth=0.9, linestyle="--", zorder=1)

            # Median labels-to-ε markers on the ε hairline (Oracle ★ + CELEUS ▲).
            if median_labels:
                for method in _MEDIAN_MARKER_METHODS:
                    if method not in _MEDIAN_MARKER_STYLE:
                        continue
                    m = median_labels.get(
                        (dataset, pair.surrogate, pair.target, method)
                    )
                    if m is None or not (500 <= m <= x_end):
                        continue
                    marker_style = _MEDIAN_MARKER_STYLE[method]
                    ax.scatter(
                        [m], [_EPSILON],
                        marker=marker_style["marker"],
                        s=marker_style["size"] * 0.55,  # smaller per-panel grid
                        color=_METHOD_STYLE[method]["color"],
                        edgecolor="white",
                        linewidths=marker_style["edge_width"],
                        zorder=6,
                        clip_on=False,
                    )

            ax.set_xlim(500, x_end)
            ax.set_ylim(*_Y_LIM)
            ax.grid(axis="y")
            ax.spines["left"].set_color("#7A7A7A")
            ax.spines["bottom"].set_color("#7A7A7A")
            ax.xaxis.set_major_formatter(FuncFormatter(base._format_labels))
            if r == 0:
                ax.set_title(_DATASET_LABELS.get(dataset, dataset))
            if r == n_rows - 1:
                ax.set_xlabel("Evaluated Samples")
            if c == 0:
                pair_label = f"{pair.surrogate}\n→ {pair.target}".replace("_", " ")
                ax.set_ylabel(pair_label, fontsize=14)

    # Shared legend at the top.
    legend_handles = []
    legend_labels = []
    for method in methods:
        legend_handles.append(
            mlines.Line2D(
                [], [],
                color=_METHOD_STYLE[method]["color"],
                linewidth=2.2,
                linestyle=_METHOD_STYLE[method]["linestyle"],
            )
        )
        legend_labels.append(_METHOD_STYLE[method]["label"])
    if median_labels:
        marker_handles = []
        for method in _MEDIAN_MARKER_METHODS:
            if method not in _MEDIAN_MARKER_STYLE:
                continue
            ms = _MEDIAN_MARKER_STYLE[method]
            marker_handles.append(
                mlines.Line2D(
                    [], [],
                    marker=ms["marker"],
                    markersize=12 if method == "ORACLE_ACC" else 9,
                    color=_METHOD_STYLE[method]["color"],
                    linestyle="None",
                    markeredgecolor="white",
                    markeredgewidth=ms["edge_width"],
                )
            )
        if marker_handles:
            legend_handles.append(
                tuple(marker_handles) if len(marker_handles) > 1 else marker_handles[0]
            )
            legend_labels.append(r"median labels to $\epsilon$")
    legend_handles.append(
        mlines.Line2D([], [], color="#8A8A8A", linewidth=1.2, linestyle="--")
    )
    legend_labels.append(rf"$\epsilon = {_EPSILON}$")

    fig.legend(
        legend_handles, legend_labels,
        ncol=len(legend_labels),
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        frameon=False,
        handler_map={tuple: HandlerTuple(ndivide=None, pad=0.6)},
        columnspacing=2.0,
    )
    fig.subplots_adjust(top=0.945, bottom=0.05, left=0.10, right=0.99,
                        hspace=0.40, wspace=0.10)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf)
    fig.savefig(out_png)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=default_config_path())
    ap.add_argument("--input-root", type=Path,
                    default=_REPO / "results" / "paper_experiments_v0504")
    ap.add_argument("--output-root", type=Path,
                    default=_REPO / "results" / "paper_experiments_v0504" / "per_pair_efficiency")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    pairs = tuple(
        common.SelectedPair(p["surrogate"], p["target"])
        for p in cfg.paper_pairs
        if (p["surrogate"], p["target"]) not in _EXCLUDED_PAIRS_FOR_GRID
    )
    datasets = common.selected_dataset_names(cfg)

    trajectories, diag = common.load_selected_seed_trajectories(
        args.input_root, cfg=cfg, methods=_METHOD_ORDER,
        datasets=datasets, pairs=pairs, loss="accuracy",
    )
    df = _aggregate_per_pair(
        trajectories, methods=_METHOD_ORDER, pairs=pairs, datasets=datasets,
    )
    median_labels = _per_pair_median_labels_to_eps(
        trajectories, methods=_MEDIAN_MARKER_METHODS,
        pairs=pairs, datasets=datasets,
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_root / "per_pair_curves.csv", index=False)

    out_pdf = args.output_root / "fig_per_pair_width.pdf"
    out_png = args.output_root / "fig_per_pair_width.png"
    plot_per_pair_grid(
        df, out_pdf, out_png,
        pairs=pairs, datasets=datasets, methods=_METHOD_ORDER,
        median_labels=median_labels,
    )
    manifest = {
        "variant": "per-pair-efficiency",
        "loss": "accuracy",
        "pairs": [{"surrogate": p.surrogate, "target": p.target} for p in pairs],
        "datasets": list(datasets),
        "methods": list(_METHOD_ORDER),
        "n_panels": len(pairs) * len(datasets),
        "input_root": str(args.input_root),
        "output_root": str(args.output_root),
        "included_seed_count": int(diag["included_seed_count"]),
        "excluded_non_stopping_count": int(diag["excluded_non_stopping_count"]),
        "epsilon": _EPSILON,
        "figure_files": [str(out_pdf), str(out_png)],
    }
    (args.output_root / "per_pair_manifest.json").write_text(
        json.dumps(manifest, indent=2)
    )
    print(f"wrote per_pair_efficiency artifacts -> {args.output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
