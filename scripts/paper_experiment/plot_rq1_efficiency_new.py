#!/usr/bin/env python3
"""Render the RQ1 label-efficiency figure with the CELEUS-anchored variant.

Differences from ``plot_rq1_efficiency.py``:

- Adds the ``M3`` ablation curve (CELEUS w/o surrogate).
- ``M1`` is labelled ``CELEUS`` (not ``SAVE-ADA``) in the legend, CSV, and
  manifest.
- Per-panel ``x_end`` is the first label where the CELEUS dataset-level mean
  curve drops to ``<= EPSILON``. If CELEUS never crosses, ``x_end`` falls
  back to CELEUS's rightmost label and ``epsilon_crossed`` is recorded as
  ``False`` in the manifest.
- ``y_lim`` tightened to (0.04, 0.15).
- Cer-Eval scope is intentionally not produced.
- The renderer is a sibling of ``plot_rq1_efficiency.py``; it reuses
  ``aggregate_pair_then_dataset`` / ``right_continuous_resample`` /
  ``_insert_boundary_anchors`` / ``apply_rq1_nature_style`` / ``_format_labels``
  via import, but defines its own method table and ``_panel_x_end``.
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


DATASET_LABELS = {"sst2": "SST-2", "mmlu": "MMLU", "agnews": "AG News"}
EPSILON = 0.05
Y_LIM = (0.04, 0.15)
METHOD_ORDER = ("M1", "M3", "ORACLE_ACC", "M4")

# Per-loss method coverage: CE drops ORACLE_ACC because no
# trajectories/oracle_cross_entropy stage exists. EPSILON is shared
# (both losses gate on width <= 0.05) but Y_LIM differs (CE width is
# on log-prob scale, much larger absolute range).
METHOD_ORDER_BY_LOSS = {
    "accuracy":      ("M1", "M3", "ORACLE_ACC", "M4"),
    "cross_entropy": ("M1", "M3", "M4"),
}
Y_LIM_BY_LOSS: dict[str, tuple[float, float] | None] = {
    "accuracy":      (0.04, 0.15),
    "cross_entropy": (0.0, 0.4),
}
# Median markers by loss: accuracy keeps both Oracle + CELEUS markers,
# CE just CELEUS (no Oracle method).
MEDIAN_MARKER_METHODS_BY_LOSS = {
    "accuracy":      ("M1", "ORACLE_ACC"),
    "cross_entropy": ("M1",),
}
# Font sizes for the new variant. Larger than the base variant so the figure
# reads well at NeurIPS column width. Defined as a module-level dict so tests
# can introspect it.
FONT_SIZES = {
    "font.size": 13,
    "axes.titlesize": 18,
    "axes.labelsize": 15,
    "legend.fontsize": 15,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
}
# Tighter horizontal margin between the three panels. Kept wide enough that
# the leftmost x-tick label of each panel (e.g. "500", "1,500") doesn't get
# clipped against the previous panel's right edge under sharey=True.
PANEL_WSPACE = 0.10
METHOD_STYLE = {
    "M1": {
        "label": "CELEUS",
        "color": "#B64A3B",
        "linestyle": "-",
        "band_alpha": 0.16,
    },
    "M3": {
        "label": "CELEUS (w/o surr)",
        "color": "#D89488",
        "linestyle": "--",
        "band_alpha": 0.10,
    },
    "ORACLE_ACC": {
        "label": "Oracle",
        "color": "#2B7A78",
        "linestyle": "-",
        "band_alpha": 0.16,
    },
    "M4": {
        "label": "Baseline e-value",
        "color": "#4C566A",
        "linestyle": "-",
        "band_alpha": 0.16,
    },
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path,
        default=default_config_path(),
        help="Main paper_experiment config (must contain paper_pairs).",
    )
    parser.add_argument(
        "--input-root", type=Path,
        default=_REPO / "results" / "paper_experiments_v0502",
        help="Root holding merged main paper_experiment trajectories.",
    )
    parser.add_argument(
        "--output-root", type=Path,
        default=_REPO / "results" / "paper_experiments_v0502" / "rq1-efficiency-new",
        help="Output directory for the new RQ1 artifacts.",
    )
    parser.add_argument(
        "--validate-only", action="store_true",
        help="Aggregate inputs and write CSV+manifest, but do not render the figure.",
    )
    return parser.parse_args(argv)


def aggregate_pair_then_dataset(
    trajectories: list[common.SeedWidthTrajectory],
    *,
    methods: tuple[str, ...],
    pairs: tuple[common.SelectedPair, ...],
    datasets: tuple[str, ...],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Same shape as ``base.aggregate_pair_then_dataset`` but writes
    ``method_label`` from this module's ``METHOD_STYLE`` (CELEUS naming)."""
    pair_rows: list[dict[str, object]] = []
    dataset_rows: list[dict[str, object]] = []
    pair_curves: dict[tuple[str, str], list[tuple[str, str, np.ndarray, np.ndarray]]] = {}

    grouped: dict[tuple[str, str, str, str], list[common.SeedWidthTrajectory]] = {}
    for traj in trajectories:
        grouped.setdefault(
            (traj.dataset, traj.surrogate, traj.target, traj.method), []
        ).append(traj)

    pair_rank = {(p.surrogate, p.target): idx for idx, p in enumerate(pairs)}
    method_rank = {m: idx for idx, m in enumerate(methods)}

    def _key_order(item):
        dataset, surrogate, target, method = item
        return (
            dataset,
            pair_rank.get((surrogate, target), len(pair_rank)),
            method_rank.get(method, len(method_rank)),
        )

    for key in sorted(grouped, key=_key_order):
        dataset, surrogate, target, method = key
        if dataset not in datasets:
            continue
        if (surrogate, target) not in pair_rank:
            continue
        if method not in method_rank:
            continue
        items = grouped[key]
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
        mean_width = matrix.mean(axis=0)
        std_width = matrix.std(axis=0, ddof=0)
        pair_curves.setdefault((dataset, method), []).append(
            (surrogate, target, grid, mean_width)
        )
        for label, mean, std in zip(grid, mean_width, std_width):
            pair_rows.append(
                {
                    "scope_type": "pair",
                    "dataset": dataset,
                    "surrogate": surrogate,
                    "target": target,
                    "method": method,
                    "method_label": METHOD_STYLE[method]["label"],
                    "labels_used": int(label),
                    "mean_width": float(mean),
                    "std_width": float(std),
                    "n_units": int(matrix.shape[0]),
                    "epsilon": EPSILON,
                }
            )

    for dataset in datasets:
        for method in methods:
            curves = pair_curves.get((dataset, method), [])
            if not curves:
                continue
            grid = np.array(
                sorted({int(label) for _, _, labels, _ in curves for label in labels}),
                dtype=np.int64,
            )
            matrix = np.vstack(
                [
                    base.right_continuous_resample(labels, mean_width, grid)
                    for _, _, labels, mean_width in curves
                ]
            )
            mean_width = matrix.mean(axis=0)
            std_width = matrix.std(axis=0, ddof=0)
            for label, mean, std in zip(grid, mean_width, std_width):
                dataset_rows.append(
                    {
                        "scope_type": "dataset",
                        "dataset": dataset,
                        "surrogate": "",
                        "target": "",
                        "method": method,
                        "method_label": METHOD_STYLE[method]["label"],
                        "labels_used": int(label),
                        "mean_width": float(mean),
                        "std_width": float(std),
                        "n_units": int(matrix.shape[0]),
                        "epsilon": EPSILON,
                    }
                )

    return pd.DataFrame(pair_rows), pd.DataFrame(dataset_rows)


def _panel_x_end_celeus(
    dataset_df: pd.DataFrame, panel: str, *, epsilon: float = EPSILON,
) -> tuple[int, str, bool]:
    """Return ``(x_end, rule, epsilon_crossed)`` for ``panel``.

    Rule 1 (``first_label_<=_epsilon``): smallest ``labels_used`` in the
    dataset-level mean of CELEUS (M1) where ``mean_width <= epsilon``.

    Rule 2 (``celeus_rightmost_fallback``): if CELEUS's mean never crosses
    epsilon, return CELEUS's rightmost ``labels_used``; ``epsilon_crossed`` is
    ``False``.
    """
    rows = dataset_df.loc[
        (dataset_df["dataset"] == panel) & (dataset_df["method"] == "M1"),
        ["labels_used", "mean_width"],
    ].sort_values("labels_used")
    if rows.empty:
        raise ValueError(f"no M1 (CELEUS) dataset rows for panel={panel!r}")
    labels = rows["labels_used"].to_numpy(dtype=np.int64)
    means = rows["mean_width"].to_numpy(dtype=np.float64)
    mask = means <= epsilon
    if mask.any():
        first_idx = int(np.argmax(mask))  # first True
        return int(labels[first_idx]), "first_label_<=_epsilon", True
    return int(labels[-1]), "celeus_rightmost_fallback", False


_MEDIAN_MARKER_METHODS = ("M1", "ORACLE_ACC")
# Per-method marker shapes for the median labels-to-ε annotation.
# Star (Oracle) and triangle (CELEUS), colored by method, white edge for the
# Nature-style cleanly separated look on top of the ε hairline.
_MEDIAN_MARKER_STYLE = {
    "ORACLE_ACC": {"marker": "*", "size": 320, "edge_width": 1.4},
    "M1":         {"marker": "^", "size": 180, "edge_width": 1.4},
}


def compute_median_labels_to_eps(
    trajectories: list[common.SeedWidthTrajectory],
    *,
    methods: tuple[str, ...] | None = None,
    datasets: tuple[str, ...],
) -> dict[tuple[str, str], int]:
    """Return ``(dataset, method) -> median labels-to-ε`` across stoppers.

    A trajectory's last label equals ``labels_to_stop`` (per
    ``common._truncate_to_stop``). Non-stoppers are already excluded by
    ``load_selected_seed_trajectories``. Median is taken across all
    pair × seed stoppers.
    """
    methods = methods if methods is not None else _MEDIAN_MARKER_METHODS
    by_key: dict[tuple[str, str], list[int]] = {}
    for traj in trajectories:
        if traj.method not in methods or traj.dataset not in datasets:
            continue
        if traj.labels.size == 0:
            continue
        by_key.setdefault((traj.dataset, traj.method), []).append(int(traj.labels[-1]))
    return {
        key: int(np.median(np.asarray(stops, dtype=np.int64)))
        for key, stops in by_key.items() if stops
    }


def _format_kbudget(n: int) -> str:
    """Render a label budget as ``2.5\\,k`` style for compact annotations."""
    if n >= 1000:
        return f"{n / 1000:.1f}\\,k".replace(".0\\,k", "\\,k")
    return f"{n}"


def _legend_handles(
    methods: tuple[str, ...],
    *,
    median_marker_methods: tuple[str, ...] = _MEDIAN_MARKER_METHODS,
) -> tuple[list, list[str]]:
    """Return (handles, labels). One combined tuple-handle is used for the
    median markers so the legend has a single ``★▲ median labels to ε`` entry
    rather than two separate marker rows."""
    handles: list = []
    labels: list[str] = []
    for method in methods:
        handles.append(
            mlines.Line2D(
                [], [],
                color=METHOD_STYLE[method]["color"],
                linewidth=2.2,
                linestyle=METHOD_STYLE[method]["linestyle"],
            )
        )
        labels.append(METHOD_STYLE[method]["label"])

    marker_handles = []
    for m in median_marker_methods:
        if m not in _MEDIAN_MARKER_STYLE:
            continue
        marker_handles.append(
            mlines.Line2D(
                [], [],
                marker=_MEDIAN_MARKER_STYLE[m]["marker"],
                markersize=15 if m == "ORACLE_ACC" else 11,
                color=METHOD_STYLE[m]["color"],
                linestyle="None",
                markeredgecolor="white",
                markeredgewidth=_MEDIAN_MARKER_STYLE[m]["edge_width"],
            )
        )
    if marker_handles:
        handles.append(tuple(marker_handles) if len(marker_handles) > 1 else marker_handles[0])
        labels.append(r"median labels to $\epsilon$")

    handles.append(
        mlines.Line2D(
            [], [],
            color="#8A8A8A",
            linewidth=1.2,
            linestyle="--",
        )
    )
    labels.append(r"$\epsilon = 0.05$")
    return handles, labels


def plot_main_figure(
    df: pd.DataFrame,
    out_pdf: Path,
    out_png: Path,
    *,
    datasets: tuple[str, ...],
    methods: tuple[str, ...],
    panel_x_ends: dict[str, int],
    median_labels: dict[tuple[str, str], int] | None = None,
    median_marker_methods: tuple[str, ...] = _MEDIAN_MARKER_METHODS,
    y_lim: tuple[float, float] | None = Y_LIM,
) -> None:
    base.apply_rq1_nature_style()
    plt.rcParams.update(FONT_SIZES)
    n = len(datasets)
    fig, axes = plt.subplots(1, n, figsize=(13.2 if n == 3 else 5.6, 4.1), sharey=True)
    if n == 1:
        axes = [axes]
    for ax, dataset in zip(axes, datasets):
        sub = df[df["dataset"] == dataset]
        if sub.empty:
            raise ValueError(f"missing dataset rows for {dataset}")
        x_end = panel_x_ends[dataset]
        for method in methods:
            rows = sub[sub["method"] == method].sort_values("labels_used")
            if rows.empty:
                raise ValueError(f"missing dataset rows for {dataset} / {method}")
            x = rows["labels_used"].to_numpy(dtype=np.int64)
            mean = rows["mean_width"].to_numpy(dtype=np.float64)
            std = rows["std_width"].to_numpy(dtype=np.float64)
            x, mean, std = base._insert_boundary_anchors(x, mean, std, x_end=x_end)
            style = METHOD_STYLE[method]
            ax.plot(
                x, mean,
                color=style["color"],
                linewidth=2.2,
                linestyle=style["linestyle"],
                drawstyle="steps-post",
                zorder=3,
            )
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

        ax.axhline(EPSILON, color="#8A8A8A", linewidth=1.2, linestyle="--", zorder=1)

        # Median labels-to-ε markers on the ε hairline (Oracle + CELEUS only;
        # M3/M4 medians are off-panel and intentionally not drawn).
        # Star (Oracle), triangle (CELEUS), method-coloured with a thin white
        # edge for clean separation from the hairline.
        if median_labels:
            for method in median_marker_methods:
                if method not in _MEDIAN_MARKER_STYLE:
                    continue
                m = median_labels.get((dataset, method))
                if m is None or not (500 <= m <= x_end):
                    continue
                marker_style = _MEDIAN_MARKER_STYLE[method]
                ax.scatter(
                    [m], [EPSILON],
                    marker=marker_style["marker"],
                    s=marker_style["size"],
                    color=METHOD_STYLE[method]["color"],
                    edgecolor="white",
                    linewidths=marker_style["edge_width"],
                    zorder=6,
                    clip_on=False,
                )

        ax.set_title(DATASET_LABELS.get(dataset, dataset))
        ax.set_xlabel("Evaluated Samples")
        ax.xaxis.set_major_formatter(FuncFormatter(base._format_labels))
        ax.set_xlim(500, x_end)
        if y_lim is not None:
            ax.set_ylim(*y_lim)
        ax.grid(axis="y")
        ax.spines["left"].set_color("#7A7A7A")
        ax.spines["bottom"].set_color("#7A7A7A")
    axes[0].set_ylabel("Confidence Interval Width")
    legend_handles, legend_labels = _legend_handles(
        methods, median_marker_methods=median_marker_methods,
    )
    # 6 entries → 2 rows, 3 cols. Keeps full ε label visible.
    fig.legend(
        legend_handles, legend_labels,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.00),
        frameon=False,
        handler_map={tuple: HandlerTuple(ndivide=None, pad=0.6)},
        columnspacing=2.2,
    )
    fig.subplots_adjust(
        top=0.76, wspace=PANEL_WSPACE, bottom=0.18, left=0.07, right=0.995,
    )
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf)
    fig.savefig(out_png)
    plt.close(fig)


def build_rq1_artifacts(
    *,
    cfg_main,
    input_root: Path,
    output_root: Path,
    validate_only: bool = False,
    datasets: tuple[str, ...] | None = None,
    pairs: tuple[common.SelectedPair, ...] | None = None,
    loss: str = "accuracy",
) -> dict[str, object]:
    methods = METHOD_ORDER_BY_LOSS.get(loss, METHOD_ORDER)
    median_marker_methods = MEDIAN_MARKER_METHODS_BY_LOSS.get(loss, _MEDIAN_MARKER_METHODS)
    y_lim = Y_LIM_BY_LOSS.get(loss, Y_LIM)
    if pairs is None:
        pairs = tuple(
            common.SelectedPair(p["surrogate"], p["target"])
            for p in cfg_main.paper_pairs
        )
    datasets_main = datasets or common.selected_dataset_names(cfg_main)

    trajectories, diag = common.load_selected_seed_trajectories(
        input_root, cfg=cfg_main, methods=methods,
        datasets=datasets_main, pairs=pairs, loss=loss,
    )
    pair_df, dataset_df = aggregate_pair_then_dataset(
        trajectories, methods=methods, pairs=pairs, datasets=datasets_main,
    )
    pair_df["figure_scope"] = "main"
    dataset_df["figure_scope"] = "main"

    panels: dict[str, dict[str, object]] = {}
    panel_x_ends: dict[str, int] = {}
    median_labels = compute_median_labels_to_eps(
        trajectories, datasets=tuple(datasets_main),
        methods=median_marker_methods,
    )
    for d in datasets_main:
        x_end, rule, crossed = _panel_x_end_celeus(dataset_df, d)
        panel_x_ends[d] = x_end
        panels[d] = {
            "x_end": int(x_end),
            "x_end_method": "M1",
            "x_end_rule": rule,
            "epsilon_crossed": bool(crossed),
            "celeus_pair_count": int(len(pairs)),
            "median_labels_to_eps": {
                m: median_labels.get((d, m))
                for m in _MEDIAN_MARKER_METHODS
            },
            "note": (
                f"x_end set to first label where CELEUS dataset-mean <= {EPSILON}"
                if crossed
                else f"CELEUS dataset-mean never crossed {EPSILON}; x_end fell back "
                     "to CELEUS's rightmost label."
            ),
        }

    output_root.mkdir(parents=True, exist_ok=True)
    combined = pd.concat([pair_df, dataset_df], ignore_index=True)
    combined.to_csv(output_root / "aggregated_trajectories.csv", index=False)

    suffix = {"accuracy": "", "cross_entropy": "_ce"}.get(loss, f"_{loss}")
    figure_files: list[str] = []
    if not validate_only:
        out_pdf = output_root / f"label_efficiency_main{suffix}.pdf"
        out_png = output_root / f"label_efficiency_main{suffix}.png"
        plot_main_figure(
            dataset_df, out_pdf, out_png,
            datasets=datasets_main, methods=methods,
            panel_x_ends=panel_x_ends,
            median_labels=median_labels,
            median_marker_methods=median_marker_methods,
            y_lim=y_lim,
        )
        figure_files.extend([str(out_pdf), str(out_png)])

    manifest: dict[str, object] = {
        "variant": "rq1-efficiency-new",
        "selected_datasets": list(datasets_main),
        "selected_pairs": [
            {"surrogate": p.surrogate, "target": p.target} for p in pairs
        ],
        "methods": [
            {"method_id": m, "label": METHOD_STYLE[m]["label"]} for m in methods
        ],
        "epsilon": EPSILON,
        "y_lim": list(Y_LIM),
        "input_root": str(input_root),
        "output_root": str(output_root),
        "input_paths": diag["input_paths"],
        "included_seed_count": int(diag["included_seed_count"]),
        "excluded_non_stopping_count": int(diag["excluded_non_stopping_count"]),
        "excluded_non_stoppers": diag["excluded_non_stoppers"],
        "pair_rows": int(len(pair_df)),
        "dataset_rows": int(len(dataset_df)),
        "figure_files": figure_files,
        "panels": panels,
    }
    (output_root / "rq1_manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg_main = load_config(args.config)
    build_rq1_artifacts(
        cfg_main=cfg_main,
        input_root=args.input_root,
        output_root=args.output_root,
        validate_only=args.validate_only,
    )
    print(f"wrote RQ1 (new variant) artifacts to {args.output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
