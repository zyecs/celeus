#!/usr/bin/env python
"""Compute pre-registered RQ7 complementarity predictors."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from save.paper_experiment.config import default_config_path, load_config  # noqa: E402
from scripts.paper_experiment.compute_mae_axis import load_cell_arrays  # noqa: E402

DEFAULT_OUT_ROOT = _REPO / "results" / "paper_experiments_v0502"
DEFAULT_CONFIG = default_config_path()


def _finite_pair_mask(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.isfinite(x) & np.isfinite(y)


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    mask = _finite_pair_mask(x, y)
    if int(mask.sum()) < 2:
        return float("nan")
    if np.unique(x[mask]).size < 2 or np.unique(y[mask]).size < 2:
        return float("nan")
    return float(pearsonr(x[mask], y[mask]).statistic)


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    mask = _finite_pair_mask(x, y)
    if int(mask.sum()) < 2:
        return float("nan")
    if np.unique(x[mask]).size < 2 or np.unique(y[mask]).size < 2:
        return float("nan")
    return float(spearmanr(x[mask], y[mask]).statistic)


def topk_lift(
    ell_proxy: np.ndarray,
    residual_sq: np.ndarray,
    *,
    top_frac: float = 0.10,
) -> float:
    """Mean residual mass in top proxy fraction divided by pool mean."""
    if not 0.0 < top_frac <= 1.0:
        raise ValueError(f"top_frac must be in (0, 1], got {top_frac}")
    mask = _finite_pair_mask(ell_proxy, residual_sq)
    proxy = ell_proxy[mask]
    resid = residual_sq[mask]
    if proxy.size == 0:
        return float("nan")
    denom = float(resid.mean())
    if not np.isfinite(denom) or denom == 0.0:
        return float("nan")
    k = max(1, int(np.ceil(proxy.size * top_frac)))
    top_idx = np.argsort(-proxy, kind="mergesort")[:k]
    return float(resid[top_idx].mean() / denom)


def compute_predictor_values(
    ell_proxy: np.ndarray,
    hat_ell: np.ndarray,
    ell: np.ndarray,
    *,
    top_frac: float = 0.10,
) -> dict[str, float]:
    """Return P1-P4 for one item pool.

    The mechanism axis uses the conditional-variance residual term
    ``(ell - hat_ell)^2`` from the LURE variance formula.
    """
    ell_proxy = np.asarray(ell_proxy, dtype=np.float64).reshape(-1)
    hat_ell = np.asarray(hat_ell, dtype=np.float64).reshape(-1)
    ell = np.asarray(ell, dtype=np.float64).reshape(-1)
    if not (ell_proxy.shape == hat_ell.shape == ell.shape):
        raise ValueError(
            "ell_proxy, hat_ell, and ell must have the same 1D shape; got "
            f"{ell_proxy.shape}, {hat_ell.shape}, {ell.shape}"
        )
    residual_sq = (ell - hat_ell) ** 2
    return {
        "rho_acc": _pearson(ell_proxy, ell),
        "rho_comp": _spearman(ell_proxy, residual_sq),
        "topk_lift": topk_lift(ell_proxy, residual_sq, top_frac=top_frac),
        "mae_unc": float(np.mean(np.abs(ell_proxy - hat_ell))),
    }


def split_by_item_parity(n_items: int) -> tuple[np.ndarray, np.ndarray]:
    idx = np.arange(int(n_items), dtype=np.int64)
    return (idx % 2) == 0, (idx % 2) == 1


def family_relationship(surrogate: str, target: str) -> str:
    llama_family = ("llama2_", "llama3_")
    if surrogate.startswith(llama_family) and target.startswith(llama_family):
        return "same_family"
    return "cross_family"


def rq7_cells(cfg) -> list[dict[str, str]]:
    cells = cfg.acquisition_sweep.get("cells_v0502") or []
    if cells:
        return [dict(c) for c in cells]
    return [
        {"dataset": dataset, "surrogate": pair["surrogate"], "target": pair["target"]}
        for dataset in cfg.datasets
        for pair in cfg.paper_pairs
    ]


def _trajectory_path(out_root: Path, dataset: str, surrogate: str, target: str) -> Path:
    return (
        out_root
        / "trajectories"
        / "main"
        / f"cell__M1__{dataset}__{surrogate}__{target}__accuracy.npz"
    )


def build_predictors(
    *,
    out_root: Path,
    config_path: Path,
    top_frac: float = 0.10,
) -> pd.DataFrame:
    cfg = load_config(config_path)
    rows = []
    for cell in rq7_cells(cfg):
        dataset = cell["dataset"]
        surrogate = cell["surrogate"]
        target = cell["target"]
        if not _trajectory_path(out_root, dataset, surrogate, target).is_file():
            continue
        arrays = load_cell_arrays(cfg, dataset, surrogate, target, loss="accuracy")
        n_pool = int(arrays["ell"].shape[0])
        a_mask, b_mask = split_by_item_parity(n_pool)
        full = compute_predictor_values(
            arrays["ell_proxy"], arrays["hat_ell"], arrays["ell"], top_frac=top_frac
        )
        split_a = compute_predictor_values(
            arrays["ell_proxy"][a_mask],
            arrays["hat_ell"][a_mask],
            arrays["ell"][a_mask],
            top_frac=top_frac,
        )
        rows.append(
            {
                "dataset": dataset,
                "surrogate": surrogate,
                "target": target,
                "family_relationship": family_relationship(surrogate, target),
                "rho_acc_full": full["rho_acc"],
                "rho_comp_full": full["rho_comp"],
                "topk_lift_full": full["topk_lift"],
                "mae_unc_full": full["mae_unc"],
                "rho_acc_A": split_a["rho_acc"],
                "rho_comp_A": split_a["rho_comp"],
                "topk_lift_A": split_a["topk_lift"],
                "mae_unc_A": split_a["mae_unc"],
                "N_pool": n_pool,
                "N_A_items": int(a_mask.sum()),
                "N_B_items": int(b_mask.sum()),
            }
        )
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--top-frac", type=float, default=0.10)
    args = parser.parse_args(argv)

    out_dir = args.out_root / "rq7-complementarity"
    out_dir.mkdir(parents=True, exist_ok=True)
    df = build_predictors(
        out_root=args.out_root,
        config_path=args.config,
        top_frac=args.top_frac,
    )
    df.to_csv(out_dir / "predictors.csv", index=False, float_format="%.18g")
    print(f"wrote {out_dir / 'predictors.csv'} ({len(df)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
