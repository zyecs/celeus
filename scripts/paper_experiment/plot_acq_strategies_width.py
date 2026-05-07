#!/usr/bin/env python3
"""Acquisition-strategy width-vs-labels figures (top-3 per loss).

Produces two 1×3-dataset figures, one per loss:

    <output-root>/acquisition_appendix/fig_strategies_width_acc.{pdf,png}
    <output-root>/acquisition_appendix/fig_strategies_width_ce.{pdf,png}

For each loss, shows the top-3 surrogate-construction strategies (R1-S* /
R2-S* + the loss-matched Oracle) ranked by median labels-to-ε from the
existing acquisition_appendix/tab_strategies_v0502_top3.tex. Curves are
the mean width across (paper_pair × seed) within each (strategy, dataset)
slice.

Mirrors the visual conventions of plot_rq1_efficiency_new.py (Helvetica,
steps-post draw, ε hairline, large fonts).
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
from save.paper_experiment.cell_paths import (  # noqa: E402
    acquisition_sweep_cell_path, main_cell_path, oracle_accuracy_cell_path,
)
from save.paper_experiment.cell_schema import load_cell  # noqa: E402

import plot_rq1_efficiency as base  # noqa: E402


_DATASET_LABELS = {"sst2": "SST-2", "mmlu": "MMLU", "agnews": "AG News"}
_EPSILON = 0.05
_Y_LIM_BY_LOSS = {"accuracy": (0.04, 0.15), "cross_entropy": (0.0, 0.4)}
_FONT_SIZES = {
    "font.size": 13,
    "axes.titlesize": 18,
    "axes.labelsize": 15,
    "legend.fontsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
}

# Non-oracle strategies ranked by median labels-to-ε (lower = better),
# pulled from acquisition_appendix/tab_strategies_v0502_top3.tex (v0504 rerun).
# Oracle is excluded by design — it's a per-loss reference ceiling, not a
# deployable acquisition strategy, so a "top-3" comparison should be among
# the practical (non-oracle) candidates.
_RANKING_BY_LOSS = {
    "accuracy": [
        # (display_label, source_kind, surrogate_type)
        # Strategy A = R2-S4, Strategy B = R2-S2, Strategy C = R2-S3 (top-3 non-Oracle)
        ("Strategy A", "main", None),
        ("Strategy B", "sweep", "remark2_strategy2"),
        ("Strategy C", "sweep", "remark2_strategy3"),
    ],
    "cross_entropy": [
        # Strategy A = R1-S2, Strategy B = R1-S3, Strategy C = R1-S1 (top-3 non-Oracle)
        ("Strategy A", "main", None),
        ("Strategy B", "sweep", "remark1_strategy3"),
        ("Strategy C", "sweep", "remark1_strategy1"),
    ],
}

# Color palette (mirrors rq1-efficiency-new tones).
_COLORS = ["#B64A3B", "#2B7A78", "#4C566A", "#D89488"]
_LINESTYLES = ["-", "-", "-", "--"]


def _find_cell_path(input_root: Path, *, source_kind: str, dataset: str,
                    surrogate: str, target: str, loss: str,
                    surrogate_type: str | None) -> Path:
    if source_kind == "main":
        return main_cell_path(
            input_root, method="M1", dataset=dataset,
            surrogate=surrogate, target=target, loss=loss,
        )
    if source_kind == "sweep":
        return acquisition_sweep_cell_path(
            input_root, dataset=dataset, surrogate=surrogate,
            target=target, loss=loss, surrogate_type=surrogate_type,
        )
    if source_kind == "oracle":
        return oracle_accuracy_cell_path(
            input_root, dataset=dataset, surrogate=surrogate,
            target=target, surrogate_type=surrogate_type,
        )
    raise ValueError(f"unknown source_kind {source_kind!r}")


def _load_widths_for_strategy(
    input_root: Path, *, source_kind: str, surrogate_type: str | None,
    loss: str, datasets: tuple[str, ...], pairs: list[dict],
) -> pd.DataFrame:
    """Return long-form rows: (strategy_proxy, dataset, labels_used, mean_width, std_width, n_units).

    Aggregates across (pair × seed) within (dataset). Per-seed truncation
    to labels_to_stop is honored so curves end at the stopping point.
    """
    rows: list[dict] = []
    for ds in datasets:
        per_seed_curves: list[tuple[np.ndarray, np.ndarray]] = []
        for pair in pairs:
            path = _find_cell_path(
                input_root, source_kind=source_kind, dataset=ds,
                surrogate=pair["surrogate"], target=pair["target"], loss=loss,
                surrogate_type=surrogate_type,
            )
            if not path.exists():
                continue
            try:
                _meta, results = load_cell(path)
            except Exception:
                continue
            for _seed, r in results.items():
                if not r.did_stop:
                    continue
                # M1/M3/oracle/sweep cells store save_labels/save_lo/save_hi.
                lo = getattr(r, "save_lo", None)
                hi = getattr(r, "save_hi", None)
                labels = getattr(r, "save_labels", None)
                if labels is None or lo is None or hi is None:
                    continue
                labels = np.asarray(labels, dtype=np.int64)
                lo_arr = np.asarray(lo, dtype=np.float64)
                hi_arr = np.asarray(hi, dtype=np.float64)
                stop = int(r.labels_to_stop)
                stop_idx = int(np.searchsorted(labels, stop, side="right"))
                if stop_idx <= 0:
                    continue
                t_labels = labels[:stop_idx]
                widths = hi_arr[:stop_idx] - lo_arr[:stop_idx]
                per_seed_curves.append((t_labels, widths))
        if not per_seed_curves:
            continue
        grid = np.array(
            sorted({int(label) for labs, _ in per_seed_curves for label in labs}),
            dtype=np.int64,
        )
        mat = np.vstack([
            base.right_continuous_resample(labs, w, grid)
            for labs, w in per_seed_curves
        ])
        mean = mat.mean(axis=0)
        std = mat.std(axis=0, ddof=0)
        for label, m, s in zip(grid, mean, std):
            rows.append({
                "dataset": ds, "labels_used": int(label),
                "mean_width": float(m), "std_width": float(s),
                "n_units": int(mat.shape[0]),
            })
    return pd.DataFrame(rows)


def _panel_x_end(df: pd.DataFrame, dataset: str, *,
                 epsilon: float = _EPSILON) -> tuple[int, bool]:
    if df.empty or "dataset" not in df.columns:
        return 5000, False
    rows = df[df["dataset"] == dataset].sort_values("labels_used")
    if rows.empty:
        return 5000, False
    labels = rows["labels_used"].to_numpy(dtype=np.int64)
    means = rows["mean_width"].to_numpy(dtype=np.float64)
    mask = means <= epsilon
    if mask.any():
        return int(labels[int(np.argmax(mask))]), True
    return int(labels[-1]), False


def _plot_loss_figure(
    strategies: list[tuple[str, pd.DataFrame]],
    *,
    datasets: tuple[str, ...],
    out_pdf: Path,
    out_png: Path,
    loss: str,
) -> None:
    """One row × N-dataset figure. ``strategies`` is a list of
    (display_label, dataframe). The first strategy's CELEUS-equivalent
    curve sets the panel x_end (rightmost label where its mean hits ε)."""
    base.apply_rq1_nature_style()
    plt.rcParams.update(_FONT_SIZES)
    y_lim = _Y_LIM_BY_LOSS[loss]
    n = len(datasets)
    fig, axes = plt.subplots(1, n, figsize=(13.2 if n == 3 else 5.6, 4.1), sharey=True)
    if n == 1:
        axes = [axes]

    # Panel x_end: max(labels) across plotted strategies (the slowest strategy
    # to hit ε sets the panel width). Each strategy's curve still ends at its
    # own natural max — we do NOT boundary-anchor here, so a strategy with
    # shorter coverage (e.g. only 2 of 6 paper_pairs swept) renders as a
    # truncated curve in its actual range rather than getting extended flat
    # to the panel edge (which would hide it under longer-running strategies).
    x_ends: dict[str, int] = {}
    for ds in datasets:
        ends: list[int] = []
        for _, df in strategies:
            xend, _crossed = _panel_x_end(df, ds)
            ends.append(xend)
        x_ends[ds] = max(ends) if ends else 5000

    for ax, ds in zip(axes, datasets):
        x_end = x_ends[ds]
        for i, (label, df) in enumerate(strategies):
            rows = df[df["dataset"] == ds].sort_values("labels_used")
            if rows.empty:
                continue
            x = rows["labels_used"].to_numpy(dtype=np.int64)
            mean = rows["mean_width"].to_numpy(dtype=np.float64)
            std = rows["std_width"].to_numpy(dtype=np.float64)
            color = _COLORS[i % len(_COLORS)]
            ls = _LINESTYLES[i % len(_LINESTYLES)]
            ax.plot(x, mean, color=color, linewidth=2.0, linestyle=ls,
                    drawstyle="steps-post", zorder=3, label=label)
            ax.fill_between(x, np.maximum(0.0, mean - std), mean + std,
                            color=color, alpha=0.13, linewidth=0.0,
                            step="post", zorder=2)
        ax.axhline(_EPSILON, color="#8A8A8A", linewidth=1.2,
                   linestyle="--", zorder=1)
        ax.set_title(_DATASET_LABELS.get(ds, ds))
        ax.set_xlabel("Evaluated Samples")
        ax.xaxis.set_major_formatter(FuncFormatter(base._format_labels))
        ax.set_xlim(500, x_end)
        ax.set_ylim(*y_lim)
        ax.grid(axis="y")
        ax.spines["left"].set_color("#7A7A7A")
        ax.spines["bottom"].set_color("#7A7A7A")

    axes[0].set_ylabel("Confidence Interval Width")
    handles = []
    labels: list[str] = []
    for i, (label, _) in enumerate(strategies):
        handles.append(mlines.Line2D(
            [], [], color=_COLORS[i % len(_COLORS)],
            linewidth=2.2, linestyle=_LINESTYLES[i % len(_LINESTYLES)],
        ))
        labels.append(label)
    handles.append(mlines.Line2D(
        [], [], color="#8A8A8A", linewidth=1.2, linestyle="--"
    ))
    labels.append(rf"$\epsilon = {_EPSILON}$")
    fig.legend(
        handles, labels,
        ncol=len(labels),
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        frameon=False,
        columnspacing=2.0,
    )
    fig.subplots_adjust(top=0.78, bottom=0.18, left=0.07, right=0.995, wspace=0.10)
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
                    default=_REPO / "results" / "paper_experiments_v0504" / "acquisition_appendix")
    ap.add_argument("--top-k", type=int, default=3,
                    help="Number of strategies per loss (default 3).")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    pairs = list(cfg.paper_pairs)
    datasets = tuple(ds for ds in ("sst2", "mmlu", "agnews") if ds in cfg.datasets)

    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest_strats: dict[str, list[dict]] = {}

    for loss in ("accuracy", "cross_entropy"):
        ranking = _RANKING_BY_LOSS[loss][: args.top_k]
        loaded: list[tuple[str, pd.DataFrame]] = []
        for label, source_kind, surrogate_type in ranking:
            df = _load_widths_for_strategy(
                args.input_root, source_kind=source_kind,
                surrogate_type=surrogate_type, loss=loss,
                datasets=datasets, pairs=pairs,
            )
            loaded.append((label, df))
        suffix = "acc" if loss == "accuracy" else "ce"
        out_pdf = args.output_root / f"fig_strategies_width_{suffix}.pdf"
        out_png = args.output_root / f"fig_strategies_width_{suffix}.png"
        _plot_loss_figure(
            loaded, datasets=datasets, out_pdf=out_pdf, out_png=out_png, loss=loss,
        )
        manifest_strats[loss] = [
            {"label": label, "source_kind": sk, "surrogate_type": st}
            for label, sk, st in ranking
        ]
        print(f"wrote {out_pdf.name} (top-{args.top_k} {loss} strategies)")

    (args.output_root / "fig_strategies_width_manifest.json").write_text(
        json.dumps({
            "datasets": list(datasets),
            "pairs": pairs,
            "top_k": args.top_k,
            "strategies_by_loss": manifest_strats,
            "input_root": str(args.input_root),
        }, indent=2)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
