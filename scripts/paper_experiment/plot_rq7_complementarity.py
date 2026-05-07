#!/usr/bin/env python
"""Plot and tabulate pre-registered RQ7 complementarity analysis."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from scripts.paper_experiment.compute_rq7_outcomes import ANCHORS  # noqa: E402
from scripts.paper_experiment.plot_style import apply_rc_helvetica  # noqa: E402

DEFAULT_OUT_ROOT = _REPO / "results" / "paper_experiments_v0502"
PREDICTORS = (
    "rho_acc_full",
    "rho_comp_full",
    "topk_lift_full",
    "mae_unc_full",
)
PREDICTOR_LABELS = {
    "rho_acc_full": r"$\rho_{\mathrm{acc}}$",
    "rho_comp_full": r"$\rho_{\mathrm{comp}}$",
    "topk_lift_full": "top-10% lift",
    "mae_unc_full": r"MAE proxy-uncertainty",
}
DATASET_COLORS = {
    "sst2": "#2b6cb0",
    "mmlu": "#c2410c",
    "agnews": "#047857",
}
MARKERS = {"same_family": "^", "cross_family": "o"}


def spearman_rs(x: Sequence[float], y: Sequence[float]) -> float:
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    if int(mask.sum()) < 2:
        return float("nan")
    if np.unique(x_arr[mask]).size < 2 or np.unique(y_arr[mask]).size < 2:
        return float("nan")
    return float(spearmanr(x_arr[mask], y_arr[mask]).statistic)


def _ci95(values: Sequence[float]) -> list[float]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return [float("nan"), float("nan")]
    return [float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))]


def _cluster_keys(df: pd.DataFrame, cluster_cols: tuple[str, ...]) -> list[tuple]:
    return list(df.loc[:, list(cluster_cols)].drop_duplicates().itertuples(index=False, name=None))


def _take_clusters(
    df: pd.DataFrame,
    clusters: Sequence[tuple],
    cluster_cols: tuple[str, ...],
) -> pd.DataFrame:
    chunks = []
    for draw_id, cluster in enumerate(clusters):
        mask = np.ones(len(df), dtype=bool)
        for col, value in zip(cluster_cols, cluster):
            mask &= df[col].to_numpy() == value
        chunk = df.loc[mask].copy()
        chunk["_boot_draw"] = draw_id
        chunks.append(chunk)
    if not chunks:
        return df.iloc[0:0].copy()
    return pd.concat(chunks, ignore_index=True)


def hierarchical_bootstrap(
    df: pd.DataFrame,
    *,
    predictors: tuple[str, ...] = PREDICTORS,
    outcome: str = "eta_bar_full",
    cluster_cols: tuple[str, ...] = ("dataset", "target"),
    n_boot: int = 10_000,
    seed: int = 42,
) -> dict:
    clusters = _cluster_keys(df, cluster_cols)
    rng = np.random.default_rng(seed)
    point = {p: spearman_rs(df[p], df[outcome]) for p in predictors}
    delta_point = float(point[predictors[1]] - point[predictors[0]])
    draws = {p: [] for p in predictors}
    delta_draws = []
    for _ in range(int(n_boot)):
        sampled = [clusters[i] for i in rng.integers(0, len(clusters), size=len(clusters))]
        boot_df = _take_clusters(df, sampled, cluster_cols)
        boot_rs = {p: spearman_rs(boot_df[p], boot_df[outcome]) for p in predictors}
        for p in predictors:
            draws[p].append(boot_rs[p])
        delta_draws.append(float(boot_rs[predictors[1]] - boot_rs[predictors[0]]))

    out = {
        p: {"point": float(point[p]), "ci95": _ci95(draws[p])}
        for p in predictors
    }
    out["delta"] = {"point": delta_point, "ci95": _ci95(delta_draws)}
    return out


def leave_one_cluster_out(
    df: pd.DataFrame,
    *,
    p1: str = "rho_acc_full",
    p2: str = "rho_comp_full",
    outcome: str = "eta_bar_full",
    cluster_cols: tuple[str, ...] = ("dataset", "target"),
) -> pd.DataFrame:
    rows = []
    for cluster in _cluster_keys(df, cluster_cols):
        keep = np.ones(len(df), dtype=bool)
        for col, value in zip(cluster_cols, cluster):
            keep &= df[col].to_numpy() != value
        sub = df.loc[keep]
        r1 = spearman_rs(sub[p1], sub[outcome])
        r2 = spearman_rs(sub[p2], sub[outcome])
        rows.append(
            {
                **{col: value for col, value in zip(cluster_cols, cluster)},
                "r_S_P1": r1,
                "r_S_P2": r2,
                "delta_rS": float(r2 - r1),
            }
        )
    return pd.DataFrame(rows)


def _joined_df(out_root: Path) -> pd.DataFrame:
    out_dir = out_root / "rq7-complementarity"
    pred = pd.read_csv(out_dir / "predictors.csv")
    out = pd.read_csv(out_dir / "outcomes.csv")
    return pred.merge(out, on=["dataset", "surrogate", "target"], how="inner")


def _scatter_panel(
    ax,
    df: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    xlabel: str,
    title: str,
    annotate_rs: bool = True,
) -> None:
    for rel, marker in MARKERS.items():
        sub_rel = df[df["family_relationship"] == rel]
        for dataset, sub in sub_rel.groupby("dataset"):
            ax.scatter(
                sub[x_col],
                sub[y_col],
                s=34,
                marker=marker,
                color=DATASET_COLORS.get(dataset, "0.3"),
                edgecolor="black",
                linewidth=0.35,
                alpha=0.9,
            )
    mask = np.isfinite(df[x_col]) & np.isfinite(df[y_col])
    if int(mask.sum()) >= 2 and df.loc[mask, x_col].nunique() >= 2:
        xs = df.loc[mask, x_col].to_numpy(dtype=float)
        ys = df.loc[mask, y_col].to_numpy(dtype=float)
        slope, intercept = np.polyfit(xs, ys, deg=1)
        grid = np.linspace(float(xs.min()), float(xs.max()), 100)
        ax.plot(grid, slope * grid + intercept, color="0.2", lw=1.0, ls="--")
    if annotate_rs:
        r_s = spearman_rs(df[x_col], df[y_col])
        ax.text(
            0.97,
            0.95,
            rf"$r_S={r_s:.2f}$",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8,
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"$\bar{\eta}$")
    ax.set_title(title, fontsize=9)
    ax.grid(True, color="0.9", lw=0.5)


def _add_legends(fig) -> None:
    dataset_handles = [
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=color,
                   markeredgecolor="black", markersize=5, label=dataset)
        for dataset, color in DATASET_COLORS.items()
    ]
    family_handles = [
        plt.Line2D([0], [0], marker=marker, color="none", markerfacecolor="0.7",
                   markeredgecolor="black", markersize=5, label=label)
        for label, marker in MARKERS.items()
    ]
    handles = dataset_handles + family_handles
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=7)


def render_main(df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.0), sharey=True)
    _scatter_panel(
        axes[0],
        df,
        x_col="rho_acc_full",
        y_col="eta_bar_full",
        xlabel=PREDICTOR_LABELS["rho_acc_full"],
        title="A. Surrogate accuracy",
    )
    _scatter_panel(
        axes[1],
        df,
        x_col="rho_comp_full",
        y_col="eta_bar_full",
        xlabel=PREDICTOR_LABELS["rho_comp_full"],
        title="B. Complementarity",
    )
    fig.suptitle("RQ7: complementarity predicts variance reduction", fontsize=11)
    _add_legends(fig)
    fig.tight_layout(rect=(0, 0.08, 1, 0.94))
    fig.savefig(out_path)
    plt.close(fig)


def render_anchor_sensitivity(df: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    fig, axes = plt.subplots(2, len(ANCHORS), figsize=(11.5, 4.2), sharey=True)
    rows = []
    for predictor in PREDICTORS:
        for t in ANCHORS:
            y_col = f"eta_{t}"
            rows.append(
                {
                    "anchor_t": t,
                    "predictor": predictor,
                    "r_S": spearman_rs(df[predictor], df[y_col]),
                }
            )
    for row_idx, predictor in enumerate(("rho_acc_full", "rho_comp_full")):
        for col_idx, t in enumerate(ANCHORS):
            y_col = f"eta_{t}"
            ax = axes[row_idx][col_idx]
            _scatter_panel(
                ax,
                df,
                x_col=predictor,
                y_col=y_col,
                xlabel=PREDICTOR_LABELS[predictor] if row_idx == 1 else "",
                title=f"t={t}" if row_idx == 0 else "",
                annotate_rs=False,
            )
            r_s = spearman_rs(df[predictor], df[y_col])
            ax.text(0.97, 0.95, rf"$r_S={r_s:.2f}$", transform=ax.transAxes,
                    ha="right", va="top", fontsize=6)
            if col_idx > 0:
                ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return pd.DataFrame(rows)


def render_topk_mae(df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.0), sharey=True)
    for ax, predictor, title in [
        (axes[0], "topk_lift_full", "P3. Top-10% lift"),
        (axes[1], "mae_unc_full", "P4. MAE proxy-uncertainty"),
    ]:
        _scatter_panel(
            ax,
            df,
            x_col=predictor,
            y_col="eta_bar_full",
            xlabel=PREDICTOR_LABELS[predictor],
            title=title,
        )
    fig.suptitle("RQ7 appendix predictors", fontsize=11)
    _add_legends(fig)
    fig.tight_layout(rect=(0, 0.08, 1, 0.94))
    fig.savefig(out_path)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--n-boot", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    apply_rc_helvetica()
    out_dir = args.out_root / "rq7-complementarity"
    out_dir.mkdir(parents=True, exist_ok=True)
    df = _joined_df(args.out_root)
    df = df[np.isfinite(df["eta_bar_full"])].copy()

    render_main(df, out_dir / "fig_rq7_main.pdf")
    anchor_df = render_anchor_sensitivity(df, out_dir / "fig_rq7_anchor_sensitivity.pdf")
    anchor_df.to_csv(out_dir / "anchor_sensitivity.csv", index=False, float_format="%.18g")
    render_topk_mae(df, out_dir / "fig_rq7_topk_mae.pdf")

    boot = hierarchical_bootstrap(
        df,
        predictors=PREDICTORS,
        outcome="eta_bar_full",
        cluster_cols=("dataset", "target"),
        n_boot=args.n_boot,
        seed=args.seed,
    )
    (out_dir / "bootstrap.json").write_text(json.dumps(boot, indent=2) + "\n")

    loo = leave_one_cluster_out(df)
    loo.to_csv(out_dir / "loo.csv", index=False, float_format="%.18g")

    clusters = _cluster_keys(df, ("dataset", "target"))
    eta_b_computable = bool(df["eta_bar_B"].notna().all()) if "eta_bar_B" in df else False
    manifest = {
        "input_paths": {
            "predictors": str(out_dir / "predictors.csv"),
            "outcomes": str(out_dir / "outcomes.csv"),
            "rq6_per_cell_curves": str(
                args.out_root / "rq6-variance" / "accuracy" / "per_cell_curves.csv"
            ),
        },
        "predictor_list": list(PREDICTORS),
        "T_anchors": list(ANCHORS),
        "n_clusters": len(clusters),
        "n_cells": int(len(df)),
        "rng_seed": int(args.seed),
        "bootstrap_iters": int(args.n_boot),
        "analysis_mode": "same_data_full_pool",
        "eta_bar_B_computable": eta_b_computable,
        "headline": {
            "delta_rS_point": boot["delta"]["point"],
            "delta_rS_ci95": boot["delta"]["ci95"],
        },
        "notes": [
            "No OLS p-values or R^2 are computed; OLS lines are visual guides only.",
            "eta_bar_B is NaN because stored v0502 trajectory cells do not contain "
            "sampled item identities; deterministic replay would be required for "
            "item-mask restricted conditional variance.",
        ],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote RQ7 outputs under {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
