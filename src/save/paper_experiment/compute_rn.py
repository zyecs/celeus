"""Compute R_N(dataset, target, loss) = mean over the pool (spec §4)."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch


def compute_rn_for_combo(
    *,
    data_root: Path,
    dataset: str,
    target: str,
    loss: str,
    ce_nll_filter: dict | None = None,
) -> tuple[float, int, int]:
    """Return (R_N, kept, raw_n).

    For ``loss == "cross_entropy"`` and ``ce_nll_filter`` enabled, applies the
    same mask that ``save.filters.build_ce_nll_mask`` would produce inside the
    loader so ``R_N.csv`` matches the filtered pool used at evaluation time.
    Raises ``ValueError`` on zero-kept.
    """
    data_root = Path(data_root)
    if loss == "accuracy":
        per_sample = torch.load(
            data_root / dataset / target / "all_set_per_sample_accuracy.pt",
            weights_only=False,
        )
        # Loss_i = 1 - correct_i. [Spec §4]
        losses = 1.0 - np.asarray(per_sample, dtype=np.float64).ravel()
        raw_n = len(losses)
        return float(losses.mean()), raw_n, raw_n
    if loss == "cross_entropy":
        per_sample = torch.load(
            data_root / dataset / target / "all_set_per_sample_cross_entropy_loss.pt",
            weights_only=False,
        )
        losses = np.asarray(per_sample, dtype=np.float64).ravel()
        raw_n = len(losses)
        if ce_nll_filter and ce_nll_filter.get("enabled"):
            from save.filters import build_ce_nll_mask
            mask = build_ce_nll_mask(losses, float(ce_nll_filter["threshold"]))
            losses = losses[mask]
            if losses.size == 0:
                raise ValueError(
                    f"ce_nll_filter threshold={ce_nll_filter['threshold']} "
                    "kept 0 samples; relax threshold."
                )
        return float(losses.mean()), int(len(losses)), int(raw_n)
    raise ValueError(f"unknown loss {loss!r}")


def write_rn_csv(
    *,
    data_root: Path,
    datasets_pairs_losses: list[tuple[str, str, str]],
    out_path: Path,
    ce_nll_filter: dict | None = None,
) -> None:
    """Write R_N.csv with extended schema including filter audit columns.

    Schema (8 columns):
        dataset, target, loss, R_N,
        ce_nll_filter_enabled, ce_nll_filter_threshold,
        ce_nll_filter_kept, ce_nll_filter_original_n
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset", "target", "loss", "R_N",
        "ce_nll_filter_enabled", "ce_nll_filter_threshold",
        "ce_nll_filter_kept", "ce_nll_filter_original_n",
    ]
    rows = []
    for dataset, target, loss in datasets_pairs_losses:
        r_n, kept, raw = compute_rn_for_combo(
            data_root=data_root, dataset=dataset, target=target, loss=loss,
            ce_nll_filter=ce_nll_filter,
        )
        enabled = bool(
            ce_nll_filter
            and ce_nll_filter.get("enabled")
            and loss == "cross_entropy"
        )
        rows.append({
            "dataset": dataset,
            "target": target,
            "loss": loss,
            "R_N": r_n,
            "ce_nll_filter_enabled": enabled,
            "ce_nll_filter_threshold": (
                ce_nll_filter["threshold"] if enabled else ""
            ),
            "ce_nll_filter_kept": kept,
            "ce_nll_filter_original_n": raw,
        })
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
