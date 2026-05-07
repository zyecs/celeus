#!/usr/bin/env python3
"""Render the standalone RQ1 label-efficiency figure and aggregates."""
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

from save.paper_experiment.config import load_config  # noqa: E402

import rq1_efficiency_common as common  # noqa: E402


DATASET_LABELS = {"sst2": "SST-2", "mmlu": "MMLU", "agnews": "AG News"}
EPSILON = 0.05
METHOD_ORDER = ("M1", "ORACLE_ACC", "M4")
METHOD_ORDER_CEREVAL = ("M1", "ORACLE_ACC", "M4", "M5")
METHOD_STYLE = {
    "M1": {"label": "SAVE-ADA", "color": "#B64A3B"},
    "ORACLE_ACC": {"label": "Oracle", "color": "#2B7A78"},
    "M4": {"label": "Baseline e-value", "color": "#4C566A"},
    "M5": {"label": "Cer-Eval", "color": "#7B3294"},
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path,
        default=_REPO / "configs" / "paper_experiment.yaml",
        help="Main paper_experiment config.",
    )
    parser.add_argument(
        "--input-root", type=Path,
        default=_REPO / "results" / "paper_experiment",
        help="Root holding merged main paper_experiment trajectories.",
    )
    parser.add_argument(
        "--cereval-config", type=Path,
        default=_REPO / "configs" / "paper_experiment_cereval_3seed_cpu.yaml",
        help="Cer-Eval config (for the M5 sub-figure).",
    )
    parser.add_argument(
        "--cereval-input-root", type=Path,
        default=_REPO / "results" / "paper_experiment_cereval_3seed_cpu",
        help="Root holding merged Cer-Eval trajectories.",
    )
    parser.add_argument(
        "--output-root", type=Path,
        default=_REPO / "results" / "paper_experiment" / "rq1-efficiency",
        help="Output directory for RQ1 artifacts (singular paper_experiment).",
    )
    parser.add_argument(
        "--skip-cereval", action="store_true",
        help="Render only the main figure; skip the Cer-Eval sub-figure.",
    )
    parser.add_argument(
        "--validate-only", action="store_true",
        help="Aggregate inputs and write CSV+manifest, but do not render figures.",
    )
    parser.add_argument(
        "--paper-pairs", action="store_true",
        help="Filter cells to cfg.paper_pair_keys (v0502 scope).",
    )
    parser.add_argument(
        "--out-root", type=Path, default=None,
        help="Override results root for input/output paths.",
    )
    return parser.parse_args(argv)


def apply_rq1_nature_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "font.size": 10,
            "axes.titlesize": 14,
            "axes.labelsize": 10,
            "legend.fontsize": 12,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "grid.color": "#D9D9D9",
            "grid.linewidth": 0.6,
            "grid.alpha": 0.7,
            "axes.grid": False,
        }
    )


def right_continuous_resample(
    labels: np.ndarray,
    widths: np.ndarray,
    grid: np.ndarray,
) -> np.ndarray:
    if labels.ndim != 1 or widths.ndim != 1 or grid.ndim != 1:
        raise ValueError("labels, widths, and grid must be 1D arrays")
    if labels.size == 0 or widths.size == 0 or grid.size == 0:
        raise ValueError("labels, widths, and grid must be non-empty")
    if labels.shape != widths.shape:
        raise ValueError("labels and widths must have matching shapes")
    if np.any(np.diff(labels) < 0):
        raise ValueError("labels must be monotone nondecreasing")
    idx = np.searchsorted(labels, grid, side="right") - 1
    idx = np.clip(idx, 0, len(widths) - 1)
    return widths[idx]


def aggregate_pair_then_dataset(
    trajectories: list[common.SeedWidthTrajectory],
    *,
    methods: tuple[str, ...],
    pairs: tuple[common.SelectedPair, ...],
    datasets: tuple[str, ...],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate seed → pair → dataset, parametrized by the caller's scope.

    Pure function: no module-level constants are read (no ``selected_pairs()``,
    no ``DATASET_ORDER``, no ``METHOD_ORDER``). The orchestrator tags rows with
    ``figure_scope`` after this returns.
    """
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
            [right_continuous_resample(item.labels, item.widths, grid) for item in items]
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
                    right_continuous_resample(labels, mean_width, grid)
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


_LAST_RENDER_USED_STEPS_POST: bool = False


def _verify_steps_post_used() -> bool:
    """Test hook: set by plot_main_figure / plot_cereval_figure on success."""
    return _LAST_RENDER_USED_STEPS_POST


def _insert_boundary_anchors(
    x: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    *,
    x_end: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Insert (500, ...) and (x_end, ...) anchors with right-continuous lookup.

    - Left anchor (500): skipped if ``x.min() > 500`` (curve genuinely starts
      after 500, do not extrapolate). Otherwise inserted only if 500 is not
      already present.
    - Right anchor (x_end): inserted whenever ``x_end >= x.min()`` (i.e. the
      anchor would land within or after the curve's data range). This is an
      intentional divergence from the spec's "do not suppress flat-line"
      wording: by always inserting the anchor, the rendered CSV is explicit
      about the value at ``x_end`` for every method, instead of relying on
      the consumer to interpret the implicit flat-line of a steps-post line.
      The visual outcome is identical (the right-continuous lookup returns
      the last data value when ``x_end > max(labels)``).
      When already present, no duplicate anchor is inserted.
    """
    x_arr = np.asarray(x, dtype=np.int64)
    mean_arr = np.asarray(mean, dtype=np.float64)
    std_arr = np.asarray(std, dtype=np.float64)
    if x_arr.size == 0:
        return x_arr, mean_arr, std_arr

    out_x = list(x_arr.tolist())
    out_m = list(mean_arr.tolist())
    out_s = list(std_arr.tolist())

    def _add(boundary: int) -> None:
        if boundary in out_x:
            return
        w = float(right_continuous_resample(x_arr, mean_arr, np.array([boundary]))[0])
        s = float(right_continuous_resample(x_arr, std_arr, np.array([boundary]))[0])
        pos = 0
        while pos < len(out_x) and out_x[pos] < boundary:
            pos += 1
        out_x.insert(pos, int(boundary))
        out_m.insert(pos, w)
        out_s.insert(pos, s)

    if int(x_arr.min()) <= 500:
        _add(500)
    if x_end >= int(x_arr.min()):
        _add(int(x_end))

    return (
        np.asarray(out_x, dtype=np.int64),
        np.asarray(out_m, dtype=np.float64),
        np.asarray(out_s, dtype=np.float64),
    )


def _legend_handles(methods: tuple[str, ...]) -> list[mlines.Line2D]:
    handles = [
        mlines.Line2D(
            [], [],
            color=METHOD_STYLE[method]["color"],
            linewidth=2.2,
            label=METHOD_STYLE[method]["label"],
        )
        for method in methods
    ]
    handles.append(
        mlines.Line2D(
            [], [],
            color="#8A8A8A",
            linewidth=1.2,
            linestyle="--",
            label=r"$\epsilon = 0.05$",
        )
    )
    return handles


def _format_labels(value: float, _pos: int) -> str:
    if value < 0:
        return ""
    return f"{int(value):,}"


def _panel_x_end(dataset_df: pd.DataFrame, panel: str, *, scope: str = "main") -> int:
    """Rightmost label of the aggregated M4 mean curve in ``panel``.

    Raises ``ValueError`` if no M4 rows for ``panel`` or if x_end < 500.
    """
    rows = dataset_df.loc[
        (dataset_df["dataset"] == panel) & (dataset_df["method"] == "M4"),
        "labels_used",
    ]
    if rows.empty:
        raise ValueError(
            f"no M4 dataset rows for scope={scope!r} panel={panel!r}"
        )
    x_end = int(rows.max())
    if x_end < 500:
        raise ValueError(
            f"x_end<500 for scope={scope!r} panel={panel!r}: m4_max={x_end}"
        )
    return x_end


def plot_main_figure(
    df: pd.DataFrame,
    out_pdf: Path,
    out_png: Path,
    *,
    datasets: tuple[str, ...],
    methods: tuple[str, ...],
) -> None:
    global _LAST_RENDER_USED_STEPS_POST
    _LAST_RENDER_USED_STEPS_POST = False  # reset; True only on successful render
    apply_rq1_nature_style()
    n = len(datasets)
    fig, axes = plt.subplots(1, n, figsize=(13.2 if n == 3 else 5.6, 4.1), sharey=True)
    if n == 1:
        axes = [axes]
    for ax, dataset in zip(axes, datasets):
        sub = df[df["dataset"] == dataset]
        if sub.empty:
            raise ValueError(f"missing dataset rows for {dataset}")
        x_end = _panel_x_end(df, dataset, scope="main")
        for method in methods:
            rows = sub[sub["method"] == method].sort_values("labels_used")
            if rows.empty:
                raise ValueError(f"missing dataset rows for {dataset} / {method}")
            x = rows["labels_used"].to_numpy(dtype=np.int64)
            mean = rows["mean_width"].to_numpy(dtype=np.float64)
            std = rows["std_width"].to_numpy(dtype=np.float64)
            x, mean, std = _insert_boundary_anchors(x, mean, std, x_end=x_end)
            color = METHOD_STYLE[method]["color"]
            ax.plot(x, mean, color=color, linewidth=2.2, drawstyle="steps-post", zorder=3)
            ax.fill_between(
                x,
                np.maximum(0.0, mean - std),
                mean + std,
                color=color,
                alpha=0.16,
                linewidth=0.0,
                step="post",
                zorder=2,
            )

        ax.axhline(EPSILON, color="#8A8A8A", linewidth=1.2, linestyle="--", zorder=1)
        ax.set_title(DATASET_LABELS.get(dataset, dataset))
        ax.set_xlabel("Labels Used")
        ax.xaxis.set_major_formatter(FuncFormatter(_format_labels))
        ax.set_xlim(500, x_end)
        ax.set_ylim(0.04, 0.20)
        ax.grid(axis="y")
        ax.spines["left"].set_color("#7A7A7A")
        ax.spines["bottom"].set_color("#7A7A7A")
    axes[0].set_ylabel("Confidence Interval Width")
    fig.legend(
        handles=_legend_handles(methods),
        ncol=len(methods) + 1,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.00),
        frameon=False,
    )
    fig.subplots_adjust(top=0.83, wspace=0.14, bottom=0.16, left=0.08, right=0.99)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf)
    fig.savefig(out_png)
    plt.close(fig)
    _LAST_RENDER_USED_STEPS_POST = True


_CEREVAL_PAIRS_HUMAN = "llama2_7b → llama3_70b, llama2_7b → Mixtral_8x7b"


def plot_cereval_figure(
    df: pd.DataFrame,
    out_pdf: Path,
    out_png: Path,
    *,
    m5_last_label: int | None = None,
) -> str:
    """Render the SST-2-only 4-line Cer-Eval sub-figure. Returns the caption."""
    global _LAST_RENDER_USED_STEPS_POST
    _LAST_RENDER_USED_STEPS_POST = False  # reset; True only on successful render
    apply_rq1_nature_style()
    unique_datasets = set(df["dataset"].unique())
    if not unique_datasets.issubset({"sst2"}):
        raise NotImplementedError(
            f"plot_cereval_figure only supports SST-2 today; got datasets={sorted(unique_datasets)}. "
            "Extend the renderer to loop over datasets if Cer-Eval coverage expands."
        )
    fig, ax = plt.subplots(1, 1, figsize=(5.6, 4.1))
    sub = df[df["dataset"] == "sst2"]
    if sub.empty:
        raise ValueError("missing sst2 rows for cereval figure")
    x_end = _panel_x_end(df, "sst2", scope="cereval")
    for method in METHOD_ORDER_CEREVAL:
        rows = sub[sub["method"] == method].sort_values("labels_used")
        if rows.empty:
            raise ValueError(f"missing sst2 rows for cereval / {method}")
        x = rows["labels_used"].to_numpy(dtype=np.int64)
        mean = rows["mean_width"].to_numpy(dtype=np.float64)
        std = rows["std_width"].to_numpy(dtype=np.float64)
        x, mean, std = _insert_boundary_anchors(x, mean, std, x_end=x_end)
        color = METHOD_STYLE[method]["color"]
        ax.plot(x, mean, color=color, linewidth=2.2, drawstyle="steps-post", zorder=3)
        ax.fill_between(
            x,
            np.maximum(0.0, mean - std),
            mean + std,
            color=color,
            alpha=0.16,
            linewidth=0.0,
            step="post",
            zorder=2,
        )

    ax.axhline(EPSILON, color="#8A8A8A", linewidth=1.2, linestyle="--", zorder=1)
    ax.set_title("SST-2 (Cer-Eval scope: 2 selected pairs)")
    ax.set_xlabel("Labels Used")
    ax.set_ylabel("Confidence Interval Width")
    ax.xaxis.set_major_formatter(FuncFormatter(_format_labels))
    ax.set_xlim(500, x_end)
    ax.set_ylim(0.04, 0.20)
    ax.grid(axis="y")
    ax.spines["left"].set_color("#7A7A7A")
    ax.spines["bottom"].set_color("#7A7A7A")
    fig.legend(
        handles=_legend_handles(METHOD_ORDER_CEREVAL),
        ncol=len(METHOD_ORDER_CEREVAL) + 1,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.00),
        frameon=False,
    )
    fig.subplots_adjust(top=0.83, bottom=0.16, left=0.16, right=0.97)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf)
    fig.savefig(out_png)
    plt.close(fig)
    _LAST_RENDER_USED_STEPS_POST = True

    last_n = (
        f"label {int(m5_last_label):,}"
        if m5_last_label is not None
        else "the last M5 label"
    )
    return (
        "SST-2; M1, Oracle, M4, and Cer-Eval (M5) averaged over the 2 pairs "
        f"Cer-Eval covers in the RQ1 selection ({_CEREVAL_PAIRS_HUMAN}). "
        "M4 here therefore differs from the main figure's M4 line. "
        f"If the M5 curve flat-lines past {last_n}, M5 stopped earlier than M4 on this slice."
    )


def build_rq1_artifacts(
    *,
    cfg_main,
    cfg_cereval,
    input_root: Path,
    cereval_input_root: Path,
    cereval_config_path: Path,
    output_root: Path,
    validate_only: bool = False,
    datasets: tuple[str, ...] | None = None,
    pairs: tuple[common.SelectedPair, ...] | None = None,
    skip_cereval: bool = False,
) -> dict[str, object]:
    """Build main + cereval artifacts in one pass; write CSV+manifest exactly once."""
    methods_main = METHOD_ORDER
    pairs_main = pairs or common.selected_pairs()
    datasets_main = datasets or common.selected_dataset_names(cfg_main)

    # ---- Main scope ----
    trajectories_main, diag_main = common.load_selected_seed_trajectories(
        input_root, cfg=cfg_main, datasets=datasets_main, pairs=pairs_main,
    )
    pair_df_main, dataset_df_main = aggregate_pair_then_dataset(
        trajectories_main,
        methods=methods_main, pairs=pairs_main, datasets=datasets_main,
    )
    pair_df_main["figure_scope"] = "main"
    dataset_df_main["figure_scope"] = "main"

    panels_main: dict[str, dict[str, object]] = {}
    for d in datasets_main:
        x_end = _panel_x_end(dataset_df_main, d, scope="main")
        panels_main[d] = {
            "x_end": int(x_end),
            "m4_pair_count": int(len(pairs_main)),
            "m5_last_label": None,
            "note": "M4 (naive e-value) averaged over "
                    f"{len(pairs_main)} selected pairs.",
        }

    # ---- Cereval scope (optional) ----
    pair_df_cereval = pd.DataFrame()
    dataset_df_cereval = pd.DataFrame()
    panels_cereval: dict[str, dict[str, object]] = {}
    cereval_diag_main: dict | None = None
    cereval_diag_m5: dict | None = None
    cereval_caption = None
    cereval_datasets_used: tuple[str, ...] = ()
    cereval_pairs_used: tuple[common.SelectedPair, ...] = ()

    if not skip_cereval:
        cereval_datasets_used, cereval_pairs_used = common.derive_cereval_scope(
            cfg_cereval, cereval_input_root,
        )
        cereval_repair_targets = common.collect_selected_cereval_repair_targets(
            cfg_cereval, cereval_input_root,
        )
        # M1/Oracle/M4 trajectories for the cereval scope live in the MAIN root
        # (cereval_input_root has only M5). Load them with cfg_main, restricted
        # to the cereval scope's pairs/datasets.
        trajectories_main_for_cereval, cereval_diag_main = (
            common.load_selected_seed_trajectories(
                input_root,
                cfg=cfg_main,
                methods=("M1", "ORACLE_ACC", "M4"),
                datasets=cereval_datasets_used,
                pairs=cereval_pairs_used,
            )
        )
        # M5 trajectories live in the cereval root.
        trajectories_m5, cereval_diag_m5 = common.load_selected_seed_trajectories(
            cereval_input_root,
            cfg=cfg_cereval,
            methods=("M5",),
            datasets=cereval_datasets_used,
            pairs=cereval_pairs_used,
        )
        trajectories_cereval = trajectories_main_for_cereval + trajectories_m5
        methods_cereval = METHOD_ORDER_CEREVAL
        pair_df_cereval, dataset_df_cereval = aggregate_pair_then_dataset(
            trajectories_cereval,
            methods=methods_cereval,
            pairs=cereval_pairs_used,
            datasets=cereval_datasets_used,
        )
        pair_df_cereval["figure_scope"] = "cereval"
        dataset_df_cereval["figure_scope"] = "cereval"

        for d in cereval_datasets_used:
            x_end = _panel_x_end(dataset_df_cereval, d, scope="cereval")
            m5_rows = dataset_df_cereval[
                (dataset_df_cereval["dataset"] == d)
                & (dataset_df_cereval["method"] == "M5")
            ]
            m5_last = int(m5_rows["labels_used"].max()) if not m5_rows.empty else None
            panels_cereval[d] = {
                "x_end": int(x_end),
                "m4_pair_count": int(len(cereval_pairs_used)),
                "m5_last_label": m5_last,
                "note": (
                    f"SST-2; M1, Oracle, M4, and Cer-Eval (M5) averaged over the "
                    f"{len(cereval_pairs_used)} pairs Cer-Eval covers in the RQ1 "
                    f"selection ({_CEREVAL_PAIRS_HUMAN}). M4 here therefore differs "
                    "from the main figure's M4 line."
                ),
            }

    # ---- Write CSV (once) ----
    output_root.mkdir(parents=True, exist_ok=True)
    combined = pd.concat(
        [pair_df_main, dataset_df_main, pair_df_cereval, dataset_df_cereval],
        ignore_index=True,
    )
    combined.to_csv(output_root / "aggregated_trajectories.csv", index=False)

    figure_files: list[str] = []
    if not validate_only:
        out_pdf = output_root / "label_efficiency_main.pdf"
        out_png = output_root / "label_efficiency_main.png"
        plot_main_figure(
            dataset_df_main, out_pdf, out_png,
            datasets=datasets_main, methods=methods_main,
        )
        figure_files.extend([str(out_pdf), str(out_png)])
        if not skip_cereval and not pair_df_cereval.empty:
            cer_pdf = output_root / "label_efficiency_cereval.pdf"
            cer_png = output_root / "label_efficiency_cereval.png"
            cereval_caption = plot_cereval_figure(
                dataset_df_cereval, cer_pdf, cer_png,
                m5_last_label=panels_cereval.get("sst2", {}).get("m5_last_label"),
            )
            figure_files.extend([str(cer_pdf), str(cer_png)])

    # ---- Manifest ----
    manifest: dict[str, object] = {
        "selected_datasets": list(datasets_main),
        "selected_pairs": [
            {"surrogate": p.surrogate, "target": p.target} for p in pairs_main
        ],
        "methods": [
            {"method_id": m, "label": METHOD_STYLE[m]["label"]} for m in methods_main
        ],
        "epsilon": EPSILON,
        "input_root": str(input_root),
        "output_root": str(output_root),
        "input_paths": diag_main["input_paths"],
        "included_seed_count": int(diag_main["included_seed_count"]),
        "excluded_non_stopping_count": int(diag_main["excluded_non_stopping_count"]),
        "excluded_non_stoppers": diag_main["excluded_non_stoppers"],
        "pair_rows": int(len(pair_df_main) + len(pair_df_cereval)),
        "dataset_rows": int(len(dataset_df_main) + len(dataset_df_cereval)),
        "figure_files": figure_files,
        "panels": panels_main,
        "cereval": (
            None if skip_cereval else {
                "config_path": str(cereval_config_path),
                "input_root": str(cereval_input_root),
                "seeds_expected": [int(s) for s in cfg_cereval.seeds_main],
                "datasets": list(cereval_datasets_used),
                "pairs": [
                    {"surrogate": p.surrogate, "target": p.target}
                    for p in cereval_pairs_used
                ],
                "merge_targets": [
                    {
                        "method_id": t.method_id,
                        "dataset": t.dataset,
                        "surrogate": t.surrogate,
                        "target": t.target,
                        "expected_seeds": list(t.expected_seeds),
                        "present_seeds": list(t.present_seeds),
                        "run_indices": list(t.run_indices),
                        "merge_index": int(t.merge_index),
                    }
                    for t in cereval_repair_targets
                ],
                "panels": panels_cereval,
                "main_root_input_paths": (
                    cereval_diag_main["input_paths"] if cereval_diag_main else []
                ),
                "cereval_root_input_paths": (
                    cereval_diag_m5["input_paths"] if cereval_diag_m5 else []
                ),
                "caption": cereval_caption,
            }
        ),
    }
    (output_root / "rq1_manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg_main = load_config(args.config)
    cfg_cereval = load_config(args.cereval_config)
    # --out-root overrides the output directory.
    output_root = args.out_root if args.out_root else args.output_root
    # --paper-pairs builds pairs_override directly from cfg.paper_pairs.
    # We do NOT intersect with common.selected_pairs() because that legacy
    # 4-pair set may not overlap with the v0502 paper_pair_keys (e.g. v0502
    # has Qwen / Mixtral targets the legacy set lacks). Filtering by overlap
    # would drop v0502-only pairs from the aggregation, making per-dataset
    # std (computed across pair means) collapse toward zero.
    pairs_override = None
    if args.paper_pairs:
        pairs_override = tuple(
            common.SelectedPair(p["surrogate"], p["target"])
            for p in cfg_main.paper_pairs
        )
    build_rq1_artifacts(
        cfg_main=cfg_main,
        cfg_cereval=cfg_cereval,
        input_root=args.input_root,
        cereval_input_root=args.cereval_input_root,
        cereval_config_path=args.cereval_config,
        output_root=output_root,
        validate_only=args.validate_only,
        skip_cereval=args.skip_cereval,
        pairs=pairs_override,
    )
    print(f"wrote RQ1 artifacts to {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
