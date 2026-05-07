#!/usr/bin/env python3
"""Generate RQ3 estimation-error plots for paper_experiment."""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

_TMP_CACHE = Path("/tmp/save-rq3-mpl")
os.environ.setdefault("MPLCONFIGDIR", str(_TMP_CACHE / "mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(_TMP_CACHE / "xdg-cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from save.paper_experiment.cell_paths import parse_main_cell_filename  # noqa: E402
from save.paper_experiment.cell_schema import compute_labels_to_stop, load_cell  # noqa: E402

DATASET_ORDER = ("sst2", "mmlu", "agnews")
LOSS_ORDER = ("accuracy", "cross_entropy")
METHOD_ORDER = ("SAVE-ADA", "Sample Mean")
FOCUS_PAIRS = (
    ("llama3_8b", "Mixtral_8x7b"),
    ("llama3_8b", "qwen25_72b"),
    ("llama2_7b", "qwen25_72b"),
)
STYLE = {
    "SAVE-ADA": {"color": "#2E5E8A", "line": "-", "stop_marker": "o", "final_marker": "s"},
    "Sample Mean": {"color": "#C07A2C", "line": "-", "stop_marker": "D", "final_marker": "^"},
}


@dataclass
class SeedTrajectory:
    dataset: str
    loss: str
    surrogate: str
    target: str
    seed: int
    t_max: int
    epsilon: float
    save_fraction: np.ndarray
    base_fraction: np.ndarray
    save_abs_err: np.ndarray
    base_abs_err: np.ndarray
    save_stop_fraction: float | None
    base_stop_fraction: float | None
    save_stop_err: float | None
    base_stop_err: float | None
    save_final_err: float
    base_final_err: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--loss",
        choices=LOSS_ORDER,
        help="Restrict generation to a single loss.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=_REPO / "results" / "paper_experiment" / "rq3-estimation-error",
        help="Output directory root for generated figures and aggregates.",
    )
    parser.add_argument(
        "--paper-root",
        type=Path,
        default=_REPO / "results" / "paper_experiment",
        help="Root directory containing paper_experiment trajectories.",
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        default=200,
        help="Number of points in the shared fraction grid.",
    )
    parser.add_argument(
        "--max-cells",
        type=int,
        default=None,
        help="Optional cap on the number of M1 main cells per loss (for smoke runs).",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Load and aggregate trajectories without writing figures.",
    )
    parser.add_argument(
        "--paper-pairs",
        action="store_true",
        help="Filter cells to cfg.paper_pair_keys (v0502 scope).",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=None,
        help="Override results root for input/output paths.",
    )
    return parser.parse_args()


def apply_rq3_nature_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "grid.color": "#D6D6D6",
            "grid.linewidth": 0.6,
            "grid.alpha": 0.6,
            "axes.grid": False,
        }
    )


def build_fraction_grid(grid_size: int = 200) -> np.ndarray:
    if grid_size < 2:
        raise ValueError(f"grid_size must be >= 2, got {grid_size}")
    return np.linspace(0.0, 1.0, grid_size, dtype=np.float64)


def iter_main_cells(out_root: Path, loss: str, method: str) -> list[Path]:
    traj_dir = Path(out_root) / "trajectories" / "main"
    paths = []
    allowed_pairs = set(FOCUS_PAIRS)
    for path in sorted(traj_dir.glob("cell__*.npz")):
        try:
            keys = parse_main_cell_filename(path.name)
        except ValueError:
            continue
        pair = (keys["surrogate"], keys["target"])
        if keys["method"] == method and keys["loss"] == loss and pair in allowed_pairs:
            paths.append(path)
    return paths


def _collapse_step_trajectory(labels: np.ndarray, values: np.ndarray, t_max: int) -> tuple[np.ndarray, np.ndarray]:
    mask = (labels >= 0) & np.isfinite(values)
    labels = labels[mask].astype(np.int64, copy=False)
    values = values[mask].astype(np.float64, copy=False)
    if labels.size == 0:
        raise ValueError("trajectory has no valid labels/values")
    if np.any(np.diff(labels) < 0):
        raise ValueError("labels must be monotone nondecreasing")

    # Trajectories are stored for every round, so labels repeat until another
    # label is acquired. Keep only the final value from each repeated-label run.
    keep = np.ones(labels.shape[0], dtype=bool)
    keep[:-1] = labels[:-1] != labels[1:]
    labels = labels[keep]
    values = values[keep]

    out_frac = np.clip(labels / float(t_max), 0.0, 1.0).astype(np.float64, copy=False)
    out_vals = values
    if out_frac[-1] < 1.0 - 1e-12:
        raise ValueError(
            f"trajectory does not reach the final budget fraction 1.0 (got {out_frac[-1]:.6f})"
        )
    out_frac[-1] = 1.0
    return out_frac, out_vals


def step_resample(
    fractions: np.ndarray,
    values: np.ndarray,
    grid: np.ndarray,
) -> np.ndarray:
    if fractions.ndim != 1 or values.ndim != 1:
        raise ValueError("fractions and values must be 1D arrays")
    if fractions.size == 0 or values.size == 0:
        raise ValueError("fractions and values must be non-empty")
    if fractions.shape != values.shape:
        raise ValueError("fractions and values must have matching shapes")
    idx = np.searchsorted(fractions, grid, side="right") - 1
    idx = np.clip(idx, 0, len(values) - 1)
    return values[idx]


def extract_save_stop_fraction(
    save_labels: np.ndarray,
    save_lo: np.ndarray,
    save_hi: np.ndarray,
    epsilon: float,
    t_max: int,
) -> float | None:
    labels_to_stop = compute_labels_to_stop(save_lo, save_hi, save_labels, epsilon=epsilon)
    if labels_to_stop <= 0:
        return None
    return float(np.clip(labels_to_stop / float(t_max), 0.0, 1.0))


def _value_at_fraction(fractions: np.ndarray, values: np.ndarray, fraction: float) -> float:
    return float(step_resample(fractions, values, np.asarray([fraction], dtype=np.float64))[0])


def save_midpoint_abs_error(save_lo: np.ndarray, save_hi: np.ndarray, true_risk: float) -> np.ndarray:
    midpoint = 0.5 * (save_lo + save_hi)
    return np.abs(midpoint - true_risk)


def load_paired_seed_trajectories(m1_path: Path, m4_path: Path) -> list[SeedTrajectory]:
    m1_meta, m1_results = load_cell(m1_path)
    m4_meta, m4_results = load_cell(m4_path)
    if m1_meta.method_id != "M1":
        raise ValueError(f"expected M1 cell, got {m1_meta.method_id!r}")
    if m4_meta.method_id != "M4":
        raise ValueError(f"expected M4 cell, got {m4_meta.method_id!r}")

    key_fields = ("dataset", "surrogate", "target", "loss", "T_max", "epsilon")
    for field in key_fields:
        if getattr(m1_meta, field) != getattr(m4_meta, field):
            raise ValueError(f"M1/M4 metadata mismatch for {field}")

    out: list[SeedTrajectory] = []
    shared_seeds = sorted(set(m1_results) & set(m4_results))
    for seed in shared_seeds:
        m1_result = m1_results[seed]
        m4_result = m4_results[seed]
        if not np.isclose(m1_result.true_R, m4_result.true_R):
            raise ValueError(f"M1/M4 true_R mismatch for seed={seed}")
        try:
            save_fraction, save_abs_err = _collapse_step_trajectory(
                m1_result.save_labels,
                save_midpoint_abs_error(m1_result.save_lo, m1_result.save_hi, m1_result.true_R),
                m1_meta.T_max,
            )
            base_fraction, base_abs_err = _collapse_step_trajectory(
                m4_result.save_labels,
                np.abs(m4_result.save_rhat - m4_result.true_R),
                m4_meta.T_max,
            )
        except ValueError:
            continue

        save_stop_fraction = extract_save_stop_fraction(
            m1_result.save_labels,
            m1_result.save_lo,
            m1_result.save_hi,
            epsilon=m1_meta.epsilon,
            t_max=m1_meta.T_max,
        )
        if save_stop_fraction is None:
            save_stop_err = None
        else:
            save_stop_err = _value_at_fraction(save_fraction, save_abs_err, save_stop_fraction)

        base_stop_fraction = extract_save_stop_fraction(
            m4_result.save_labels,
            m4_result.save_lo,
            m4_result.save_hi,
            epsilon=m4_meta.epsilon,
            t_max=m4_meta.T_max,
        )
        if base_stop_fraction is None:
            base_stop_err = None
        else:
            base_stop_err = _value_at_fraction(base_fraction, base_abs_err, base_stop_fraction)

        out.append(
            SeedTrajectory(
                dataset=m1_meta.dataset,
                loss=m1_meta.loss,
                surrogate=m1_meta.surrogate,
                target=m1_meta.target,
                seed=int(seed),
                t_max=m1_meta.T_max,
                epsilon=m1_meta.epsilon,
                save_fraction=save_fraction,
                base_fraction=base_fraction,
                save_abs_err=save_abs_err,
                base_abs_err=base_abs_err,
                save_stop_fraction=save_stop_fraction,
                base_stop_fraction=base_stop_fraction,
                save_stop_err=save_stop_err,
                base_stop_err=base_stop_err,
                save_final_err=float(save_abs_err[-1]),
                base_final_err=float(base_abs_err[-1]),
            )
        )
    return out


def aggregate_group(
    trajectories: list[SeedTrajectory],
    grid: np.ndarray,
    dataset: str | None,
) -> pd.DataFrame:
    selected = trajectories if dataset is None else [t for t in trajectories if t.dataset == dataset]
    if not selected:
        return pd.DataFrame(
            columns=["loss", "scope", "dataset", "kind", "method", "fraction", "mean_error", "std_error", "n"]
        )

    scope = "pooled" if dataset is None else dataset
    loss = selected[0].loss
    rows: list[dict[str, object]] = []
    method_specs = (
        ("SAVE-ADA", "save_fraction", "save_abs_err", "save_final_err", "save_stop_fraction", "save_stop_err"),
        ("Sample Mean", "base_fraction", "base_abs_err", "base_final_err", "base_stop_fraction", "base_stop_err"),
    )

    for method, frac_attr, err_attr, final_attr, stop_frac_attr, stop_err_attr in method_specs:
        curve_matrix = np.vstack(
            [step_resample(getattr(t, frac_attr), getattr(t, err_attr), grid) for t in selected]
        )
        curve_mean = curve_matrix.mean(axis=0)
        curve_std = curve_matrix.std(axis=0, ddof=0)
        for frac, mean_err, std_err in zip(grid, curve_mean, curve_std):
            rows.append(
                {
                    "loss": loss,
                    "scope": scope,
                    "dataset": dataset,
                    "kind": "curve",
                    "method": method,
                    "fraction": float(frac),
                    "mean_error": float(mean_err),
                    "std_error": float(std_err),
                    "n": int(curve_matrix.shape[0]),
                }
            )

        stop_fracs = [getattr(t, stop_frac_attr) for t in selected if getattr(t, stop_frac_attr) is not None]
        mean_stop_frac = float(np.mean(stop_fracs)) if stop_fracs else np.nan
        stop_values = [getattr(t, stop_err_attr) for t in selected if getattr(t, stop_err_attr) is not None]
        if stop_values:
            stop_arr = np.asarray(stop_values, dtype=np.float64)
            rows.append(
                {
                    "loss": loss,
                    "scope": scope,
                    "dataset": dataset,
                    "kind": "stopping",
                    "method": method,
                    "fraction": mean_stop_frac,
                    "mean_error": float(stop_arr.mean()),
                    "std_error": float(stop_arr.std(ddof=0)),
                    "n": int(stop_arr.size),
                }
            )

        final_arr = np.asarray([getattr(t, final_attr) for t in selected], dtype=np.float64)
        rows.append(
            {
                "loss": loss,
                "scope": scope,
                "dataset": dataset,
                "kind": "final",
                "method": method,
                "fraction": 1.0,
                "mean_error": float(final_arr.mean()),
                "std_error": float(final_arr.std(ddof=0)),
                "n": int(final_arr.size),
            }
        )

    return pd.DataFrame(rows)


def _metadata_for_loss(
    trajectories: list[SeedTrajectory],
    grid_size: int,
    m1_cell_paths: list[Path],
    m4_cell_paths: list[Path],
    output_dir: Path,
) -> dict[str, object]:
    counts_by_dataset = {
        dataset: sum(1 for t in trajectories if t.dataset == dataset)
        for dataset in DATASET_ORDER
    }
    stop_counts_by_dataset = {
        dataset: sum(
            1 for t in trajectories if t.dataset == dataset and t.save_stop_fraction is not None
        )
        for dataset in DATASET_ORDER
    }
    return {
        "source_glob": "results/paper_experiment/trajectories/main/cell__M1__*.npz",
        "selected_pairs": [
            {"surrogate": surrogate, "target": target}
            for surrogate, target in FOCUS_PAIRS
        ],
        "output_dir": str(output_dir),
        "method": "M1_vs_M4",
        "comparison": {
            "save_ada": "cs_midpoint_from_save_lo_hi",
            "sample_mean": "running_mean_from_m4_save_rhat",
        },
        "stopping_point_semantics": {
            "save_ada": "own_M1_stopping_fraction",
            "sample_mean": "own_M4_stopping_fraction",
        },
        "grid_size": int(grid_size),
        "aggregation": "mean_with_std_band",
        "x_axis": "fraction_of_budget_used",
        "y_axis": "absolute_estimation_error",
        "n_m1_cells": len(m1_cell_paths),
        "n_m4_cells": len(m4_cell_paths),
        "n_seed_trajectories": len(trajectories),
        "n_seed_trajectories_by_dataset": counts_by_dataset,
        "n_save_stopping_points_total": sum(t.save_stop_fraction is not None for t in trajectories),
        "n_save_stopping_points_by_dataset": stop_counts_by_dataset,
        "n_base_stopping_points_total": sum(t.base_stop_fraction is not None for t in trajectories),
        "n_base_stopping_points_by_dataset": {
            dataset: sum(
                1 for t in trajectories if t.dataset == dataset and t.base_stop_fraction is not None
            )
            for dataset in DATASET_ORDER
        },
    }


def _plot_scope(ax: plt.Axes, df: pd.DataFrame, title: str, show_ylabel: bool) -> None:
    ax.set_title(title)
    for method in METHOD_ORDER:
        style = STYLE[method]
        curve = df[(df["kind"] == "curve") & (df["method"] == method)].sort_values("fraction")
        if curve.empty:
            continue
        x = curve["fraction"].to_numpy(dtype=np.float64)
        mean = curve["mean_error"].to_numpy(dtype=np.float64)
        std = curve["std_error"].to_numpy(dtype=np.float64)
        ax.plot(x, mean, color=style["color"], linestyle=style["line"], linewidth=2.0, label=method, zorder=3)
        ax.fill_between(
            x,
            np.maximum(0.0, mean - std),
            mean + std,
            color=style["color"],
            alpha=0.18,
            linewidth=0.0,
            zorder=2,
        )
        stop = df[(df["kind"] == "stopping") & (df["method"] == method)]
        if not stop.empty:
            row = stop.iloc[0]
            ax.scatter(
                [float(row["fraction"])],
                [float(row["mean_error"])],
                color=style["color"],
                marker=style["stop_marker"],
                s=48,
                edgecolors="white",
                linewidths=0.8,
                zorder=4,
            )
        final = df[(df["kind"] == "final") & (df["method"] == method)]
        if not final.empty:
            row = final.iloc[0]
            ax.scatter(
                [float(row["fraction"])],
                [float(row["mean_error"])],
                color=style["color"],
                marker=style["final_marker"],
                s=52,
                edgecolors="white",
                linewidths=0.8,
                zorder=4,
            )

    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("Fraction of Budget Used")
    ax.set_ylabel("Absolute Estimation Error |R_hat - R|" if show_ylabel else "")
    ax.grid(axis="y")
    ax.spines["left"].set_color("#7A7A7A")
    ax.spines["bottom"].set_color("#7A7A7A")


def _make_legend_handles() -> list[mlines.Line2D]:
    handles: list[mlines.Line2D] = []
    for method in METHOD_ORDER:
        style = STYLE[method]
        handles.append(
            mlines.Line2D(
                [], [],
                color=style["color"],
                linestyle=style["line"],
                linewidth=2.0,
                label=method,
            )
        )
    handles.append(
        mlines.Line2D(
            [], [],
            color="#666666",
            marker="o",
            linestyle="None",
            markersize=6,
            label="Stopping Point",
        )
    )
    handles.append(
        mlines.Line2D(
            [], [],
            color="#666666",
            marker="s",
            linestyle="None",
            markersize=6,
            label="Final Budget",
        )
    )
    return handles


def _scope_band_envelope(df: pd.DataFrame) -> tuple[float, float]:
    """Compute the (min, max) of mean ± std bands across all methods in a scope.

    Used by per-panel adaptive ylim so each dataset gets its own y-range.
    """
    lo = float("inf")
    hi = -float("inf")
    for method in METHOD_ORDER:
        curve = df[(df["kind"] == "curve") & (df["method"] == method)]
        if curve.empty:
            continue
        mean = curve["mean_error"].to_numpy(dtype=np.float64)
        std = curve["std_error"].to_numpy(dtype=np.float64)
        lo = min(lo, float(np.maximum(0.0, mean - std).min()))
        hi = max(hi, float((mean + std).max()))
    return lo, hi


def plot_pooled(df: pd.DataFrame, out_path: Path, loss: str) -> None:
    apply_rq3_nature_style()
    pooled = df[df["scope"] == "pooled"]
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    _plot_scope(ax, pooled, f"{loss.replace('_', ' ').title()} (Pooled)", True)
    lo, hi = _scope_band_envelope(pooled)
    if hi > lo:
        pad = (hi - lo) * 0.12
        ax.set_ylim(max(0.0, lo - pad), hi + pad)
    ax.yaxis.labelpad = 2
    ax.legend(handles=_make_legend_handles(), loc="upper right", frameon=False, ncol=2)
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def plot_per_dataset(df: pd.DataFrame, out_path: Path, loss: str) -> None:
    apply_rq3_nature_style()
    fig, axes = plt.subplots(1, 3, figsize=(8.5, 2.8), sharex=True, sharey=False)
    for ax, dataset in zip(axes, DATASET_ORDER):
        sub = df[df["scope"] == dataset]
        _plot_scope(ax, sub, dataset.upper(), dataset == DATASET_ORDER[0])
        lo, hi = _scope_band_envelope(sub)
        if hi > lo:
            # 25% headroom above so the panel title clears Oracle/Unweighted curves.
            pad_lo = (hi - lo) * 0.05
            pad_hi = (hi - lo) * 0.25
            ax.set_ylim(max(0.0, lo - pad_lo), hi + pad_hi)
        ax.yaxis.labelpad = 2
    handles = _make_legend_handles()
    fig.legend(handles=handles, loc="upper center", frameon=False, ncol=4,
               bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(f"{loss.replace('_', ' ').title()} Estimation Error by Dataset", y=1.07)
    fig.tight_layout(rect=(0, 0, 1, 0.93), h_pad=0.2, w_pad=0.3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def write_metadata(out_path: Path, payload: dict[str, object]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def run_loss(
    paper_root: Path,
    output_root: Path,
    loss: str,
    grid_size: int,
    max_cells: int | None,
    validate_only: bool,
) -> dict[str, object]:
    m1_cell_paths = iter_main_cells(paper_root, loss, method="M1")
    m4_cell_paths = iter_main_cells(paper_root, loss, method="M4")
    if max_cells is not None:
        m1_cell_paths = m1_cell_paths[:max_cells]
        m4_cell_paths = m4_cell_paths[:max_cells]
    if not m1_cell_paths:
        raise RuntimeError(f"no M1 main cells found for loss={loss!r} under {paper_root}")
    if not m4_cell_paths:
        raise RuntimeError(f"no M4 main cells found for loss={loss!r} under {paper_root}")

    m4_by_name = {path.name.replace("cell__M4__", "cell__M1__"): path for path in m4_cell_paths}

    trajectories: list[SeedTrajectory] = []
    paired_m1_paths: list[Path] = []
    paired_m4_paths: list[Path] = []
    for m1_path in m1_cell_paths:
        m4_path = m4_by_name.get(m1_path.name)
        if m4_path is None:
            continue
        paired_m1_paths.append(m1_path)
        paired_m4_paths.append(m4_path)
        trajectories.extend(load_paired_seed_trajectories(m1_path, m4_path))
    if not trajectories:
        raise RuntimeError(f"no valid seed trajectories found for loss={loss!r}")

    grid = build_fraction_grid(grid_size)
    frames = [aggregate_group(trajectories, grid, dataset=None)]
    frames.extend(aggregate_group(trajectories, grid, dataset=dataset) for dataset in DATASET_ORDER)
    aggregated = pd.concat(frames, ignore_index=True)
    meta = _metadata_for_loss(trajectories, grid_size, paired_m1_paths, paired_m4_paths, output_root / loss)

    if not validate_only:
        loss_dir = output_root / loss
        loss_dir.mkdir(parents=True, exist_ok=True)
        aggregated.to_csv(loss_dir / "aggregated_curves.csv", index=False)
        write_metadata(loss_dir / "metadata.json", meta)
        plot_pooled(aggregated, loss_dir / "pooled.pdf", loss)
        plot_per_dataset(aggregated, loss_dir / "per_dataset.pdf", loss)
    return meta


def main() -> int:
    args = parse_args()
    # When --paper-pairs is active, replace FOCUS_PAIRS with cfg.paper_pairs
    # directly. NOTE: do NOT intersect with the legacy FOCUS_PAIRS — the v0502
    # paper_pairs may not overlap (e.g. v0502 has Qwen / Mixtral targets the
    # legacy set lacks), and intersecting collapses the visualisation.
    global FOCUS_PAIRS
    if args.paper_pairs:
        from save.paper_experiment.config import load_config as _load_config
        _cfg = _load_config("configs/paper_experiment.yaml")
        FOCUS_PAIRS = tuple(
            (p["surrogate"], p["target"]) for p in _cfg.paper_pairs
        )
    out_root = args.out_root if args.out_root else args.output_root
    losses = [args.loss] if args.loss else list(LOSS_ORDER)
    summaries = {}
    for loss in losses:
        meta = run_loss(
            paper_root=args.paper_root,
            output_root=out_root,
            loss=loss,
            grid_size=args.grid_size,
            max_cells=args.max_cells,
            validate_only=args.validate_only,
        )
        summaries[loss] = meta
        print(
            f"{loss}: {meta['n_seed_trajectories']} trajectories "
            f"from {meta['n_m1_cells']} M1 cells and {meta['n_m4_cells']} M4 cells"
        )
    if args.validate_only:
        print(json.dumps(summaries, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
