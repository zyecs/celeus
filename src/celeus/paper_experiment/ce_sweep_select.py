"""CE mini-sweep selection rules (paper_experiment spec §5)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np


def _stop_rate(group: list[dict]) -> float:
    stops = sum(1 for r in group if r["labels_to_stop"] > 0)
    return stops / len(group) if group else 0.0


def _miscov_count(group: list[dict]) -> int:
    return sum(1 for r in group if r["ever_miss"])


def _median_labels_to_stop(group: list[dict]) -> float:
    stopped = [r["labels_to_stop"] for r in group if r["labels_to_stop"] > 0]
    return float(np.median(stopped)) if stopped else float("inf")


def select_ce_winner(
    groups: dict[tuple[str, float], list[dict]],
    *,
    min_stop_rate: float,
    required_miscoverages: int,
    tie_break_pct: float,
) -> dict:
    """Apply spec §5 filter A/B + minimum-median + tie-break rule."""
    # Filter A + B.
    survivors = [
        (key, group) for key, group in groups.items()
        if _miscov_count(group) <= required_miscoverages
        and _stop_rate(group) >= min_stop_rate
    ]
    if not survivors:
        raise RuntimeError("no configurations survive validity/non-degeneracy filters")

    # Rank by median labels-to-stop.
    ranked = sorted(
        survivors,
        key=lambda kg: (_median_labels_to_stop(kg[1]), kg[0][1], kg[0][0]),
    )
    best_median = _median_labels_to_stop(ranked[0][1])
    cutoff = best_median * (1.0 + tie_break_pct)
    within_pct = [
        kg for kg in ranked
        if _median_labels_to_stop(kg[1]) <= cutoff
    ]
    # Prefer higher stop rate, then lower beta.
    within_pct.sort(
        key=lambda kg: (-_stop_rate(kg[1]), kg[0][1]),
    )
    (stype, beta), winning_group = within_pct[0]
    return {
        "surrogate_type": stype,
        "beta_min": float(beta),
        "median_labels_to_stop": _median_labels_to_stop(winning_group),
        "miscoverage_count": _miscov_count(winning_group),
        "stop_rate": _stop_rate(winning_group),
        "surviving_configs": [
            {"surrogate_type": k[0], "beta_min": float(k[1])}
            for k, _ in survivors
        ],
    }


def load_ce_sweep_groups(
    out_root: Path, cells: Iterable,
) -> dict[tuple[str, float], list[dict]]:
    """Load ce_sweep cell files into (strategy, beta) → [{labels_to_stop, ever_miss}, ...]."""
    from .cell_paths import ce_sweep_cell_path
    from .cell_schema import load_cell

    groups: dict[tuple[str, float], list[dict]] = {}
    for cell in cells:
        path = ce_sweep_cell_path(
            out_root, dataset=cell.dataset, surrogate=cell.surrogate,
            target=cell.target, surrogate_type=cell.surrogate_type,
            beta_min=cell.beta_min,
        )
        if not path.exists():
            continue
        _, results = load_cell(path)
        key = (cell.surrogate_type, float(cell.beta_min))
        for r in results.values():
            groups.setdefault(key, []).append(
                {"labels_to_stop": r.labels_to_stop, "ever_miss": r.ever_miss}
            )
    return groups


def write_winner(winner: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(winner, indent=2))
