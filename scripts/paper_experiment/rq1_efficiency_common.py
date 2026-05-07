#!/usr/bin/env python3
"""Shared helpers for the standalone RQ1 efficiency workflow."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import sys

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from save.paper_experiment.cell_paths import (  # noqa: E402
    main_cell_path,
    oracle_accuracy_cell_path,
)
from save.paper_experiment.cell_schema import load_cell  # noqa: E402
from save.paper_experiment.index_maps import (  # noqa: E402
    chunked_assignments,
    main_cells_for_loss,
    oracle_accuracy_cells,
)


DATASET_ORDER = ("sst2", "mmlu", "agnews")
ORACLE_SURROGATE_TYPE = "remark2_oracle_strategy4"
RQ1_METHODS = ("M1", "ORACLE_ACC", "M4")


@dataclass(frozen=True)
class SelectedPair:
    surrogate: str
    target: str


@dataclass(frozen=True)
class MainRepairTarget:
    method_id: str
    dataset: str
    surrogate: str
    target: str
    expected_seeds: tuple[int, ...]
    present_seeds: tuple[int, ...]
    run_indices: tuple[int, ...]
    merge_index: int


@dataclass(frozen=True)
class SelectedCellStatus:
    method_id: str
    dataset: str
    surrogate: str
    target: str
    path: str
    exists: bool
    present_seeds: tuple[int, ...]
    complete: bool


@dataclass
class SeedWidthTrajectory:
    dataset: str
    surrogate: str
    target: str
    method: str
    seed: int
    labels: np.ndarray
    widths: np.ndarray


def selected_pairs() -> tuple[SelectedPair, ...]:
    return (
        SelectedPair("llama3_8b", "llama3_70b"),
        SelectedPair("llama2_7b", "llama3_70b"),
        SelectedPair("llama3_8b", "deepseek_67b"),
        SelectedPair("llama2_7b", "Mixtral_8x7b"),
    )


def cereval_pairs(
    cfg_cereval, *, selection: tuple[SelectedPair, ...] | None = None
) -> tuple[SelectedPair, ...]:
    """Intersect ``selected_pairs()`` with the cereval cfg's pair list.

    ``cfg_cereval.cereval_pairs`` returns ``list[dict]`` with ``"surrogate"``/
    ``"target"`` keys (config.py:56-65); we compare by string tuples.
    """
    selection = selection if selection is not None else selected_pairs()
    cereval_pairs_in_cfg = {
        (p["surrogate"], p["target"]) for p in cfg_cereval.cereval_pairs
    }
    return tuple(
        p for p in selection
        if (p.surrogate, p.target) in cereval_pairs_in_cfg
    )


def selected_dataset_names(cfg) -> tuple[str, ...]:
    return tuple(ds for ds in DATASET_ORDER if ds in cfg.datasets)


def _expected_seeds(cfg) -> tuple[int, ...]:
    return tuple(int(seed) for seed in cfg.seeds_main)


def _path_for_method(
    out_root: Path,
    method: str,
    dataset: str,
    surrogate: str,
    target: str,
    *,
    loss: str = "accuracy",
) -> Path:
    if method == "ORACLE_ACC":
        return oracle_accuracy_cell_path(
            out_root,
            dataset=dataset,
            surrogate=surrogate,
            target=target,
            surrogate_type=ORACLE_SURROGATE_TYPE,
        )
    if method not in {"M1", "M3", "M4", "M5"}:
        raise ValueError(f"unsupported RQ1 method: {method!r}")
    return main_cell_path(
        out_root,
        method=method,
        dataset=dataset,
        surrogate=surrogate,
        target=target,
        loss=loss,
    )


def _present_seeds(path: Path) -> tuple[int, ...]:
    if not path.exists():
        return ()
    _, results = load_cell(path)
    return tuple(sorted(int(seed) for seed in results))


def collect_selected_cell_statuses(
    cfg,
    out_root: Path,
    method: str,
    *,
    datasets: tuple[str, ...] | None = None,
    pairs: tuple[SelectedPair, ...] | None = None,
    loss: str = "accuracy",
) -> list[SelectedCellStatus]:
    expected = set(_expected_seeds(cfg))
    statuses: list[SelectedCellStatus] = []
    for dataset in (datasets or selected_dataset_names(cfg)):
        for pair in (pairs or selected_pairs()):
            path = _path_for_method(out_root, method, dataset, pair.surrogate, pair.target, loss=loss)
            present = _present_seeds(path)
            statuses.append(
                SelectedCellStatus(
                    method_id=method,
                    dataset=dataset,
                    surrogate=pair.surrogate,
                    target=pair.target,
                    path=str(path),
                    exists=path.exists(),
                    present_seeds=present,
                    complete=set(present) == expected,
                )
            )
    return statuses


def _main_run_indices(
    cfg,
    method_id: str,
    dataset: str,
    surrogate: str,
    target: str,
) -> tuple[int, ...]:
    cells = main_cells_for_loss(cfg, loss="accuracy", include_cereval=False)
    assignments = chunked_assignments(cells, cfg.seeds_main)
    out = [
        idx
        for idx, asn in enumerate(assignments)
        if asn.cell.method_id == method_id
        and asn.cell.dataset == dataset
        and asn.cell.surrogate == surrogate
        and asn.cell.target == target
        and asn.cell.loss == "accuracy"
    ]
    if not out:
        raise ValueError(
            "could not locate main-accuracy assignment indices for "
            f"{method_id} {dataset} {surrogate}->{target}"
        )
    return tuple(out)


def _main_merge_index(
    cfg,
    method_id: str,
    dataset: str,
    surrogate: str,
    target: str,
) -> int:
    targets = []
    for loss in cfg.losses:
        for cell in main_cells_for_loss(cfg, loss=loss, include_cereval=True):
            targets.append(("main", cell))
    for idx, (kind, cell) in enumerate(targets):
        if (
            kind == "main"
            and cell.method_id == method_id
            and cell.dataset == dataset
            and cell.surrogate == surrogate
            and cell.target == target
            and cell.loss == "accuracy"
        ):
            return idx
    raise ValueError(
        "could not locate merge-cells index for "
        f"{method_id} {dataset} {surrogate}->{target}"
    )


def collect_selected_main_repair_targets(
    cfg,
    out_root: Path,
    methods: tuple[str, ...] = ("M1", "M4"),
    *,
    datasets: tuple[str, ...] | None = None,
    pairs: tuple[SelectedPair, ...] | None = None,
) -> list[MainRepairTarget]:
    expected = _expected_seeds(cfg)
    expected_set = set(expected)
    targets: list[MainRepairTarget] = []
    for method in methods:
        if method not in {"M1", "M4"}:
            raise ValueError(f"main repair only supports M1/M4, got {method!r}")
        for status in collect_selected_cell_statuses(
            cfg, out_root, method, datasets=datasets, pairs=pairs
        ):
            if set(status.present_seeds) == expected_set:
                continue
            targets.append(
                MainRepairTarget(
                    method_id=status.method_id,
                    dataset=status.dataset,
                    surrogate=status.surrogate,
                    target=status.target,
                    expected_seeds=expected,
                    present_seeds=status.present_seeds,
                    run_indices=_main_run_indices(
                        cfg,
                        status.method_id,
                        status.dataset,
                        status.surrogate,
                        status.target,
                    ),
                    merge_index=_main_merge_index(
                        cfg,
                        status.method_id,
                        status.dataset,
                        status.surrogate,
                        status.target,
                    ),
                )
            )
    return targets


def oracle_stage_manifest(cfg) -> dict[str, object]:
    cells = oracle_accuracy_cells(cfg)
    assignments = chunked_assignments(cells, cfg.seeds_main)
    return {
        "surrogate_type": ORACLE_SURROGATE_TYPE,
        "total_cells": len(cells),
        "total_assignments": len(assignments),
        "total_merge_targets": len(cells),
        "assignment_index_range": [0, len(assignments) - 1] if assignments else [],
        "merge_index_range": [0, len(cells) - 1] if cells else [],
    }


def repair_target_to_dict(target: MainRepairTarget) -> dict[str, object]:
    data = asdict(target)
    data["run_commands"] = [
        f"python scripts/run_paper_experiment.py --stage main-accuracy --index {idx}"
        for idx in target.run_indices
    ]
    data["merge_command"] = (
        f"python scripts/run_paper_experiment.py --stage merge-cells --index {target.merge_index}"
    )
    return data


def status_to_dict(status: SelectedCellStatus) -> dict[str, object]:
    return asdict(status)


def _arrays_for_method(result, method: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if method == "M4":
        return result.base_labels, result.base_lo, result.base_hi
    # M5 (Cer-Eval) writes save_* and base_* to identical arrays in
    # run_cereval_for_seed (src/save/paper_experiment/runners.py:207-209), so
    # reading save_* keeps M5 in the same branch as M1/ORACLE_ACC.
    if method in {"M1", "M3", "ORACLE_ACC", "M5"}:
        return result.save_labels, result.save_lo, result.save_hi
    raise ValueError(f"unsupported method: {method!r}")


def _truncate_to_stop(
    labels: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    labels_to_stop: int,
) -> tuple[np.ndarray, np.ndarray]:
    if labels_to_stop <= 0:
        raise ValueError("labels_to_stop must be positive for truncated trajectories")

    widths = hi - lo
    mask = (labels >= 0) & np.isfinite(widths)
    labels = labels[mask].astype(np.int64, copy=False)
    widths = widths[mask].astype(np.float64, copy=False)
    if labels.size == 0:
        raise ValueError("trajectory has no valid labels")
    if np.any(np.diff(labels) < 0):
        raise ValueError("labels must be monotone nondecreasing")

    cross_idx = int(np.searchsorted(labels, labels_to_stop, side="left"))
    if cross_idx >= labels.size:
        raise ValueError(
            f"labels_to_stop={labels_to_stop} exceeds the recorded label axis"
        )
    labels = labels[: cross_idx + 1]
    widths = widths[: cross_idx + 1]

    keep = np.ones(labels.shape[0], dtype=bool)
    keep[:-1] = labels[:-1] != labels[1:]
    labels = labels[keep]
    widths = widths[keep]
    return labels, widths


def load_selected_seed_trajectories(
    out_root: Path,
    *,
    cfg,
    methods: tuple[str, ...] = RQ1_METHODS,
    datasets: tuple[str, ...] | None = None,
    pairs: tuple[SelectedPair, ...] | None = None,
    loss: str = "accuracy",
) -> tuple[list[SeedWidthTrajectory], dict[str, object]]:
    expected = set(_expected_seeds(cfg))
    trajectories: list[SeedWidthTrajectory] = []
    diagnostics = {
        "input_paths": [],
        "included_seed_count": 0,
        "excluded_non_stopping_count": 0,
        "excluded_non_stoppers": [],
    }

    for method in methods:
        for status in collect_selected_cell_statuses(
            cfg, out_root, method, datasets=datasets, pairs=pairs, loss=loss,
        ):
            if not status.exists:
                raise FileNotFoundError(
                    f"missing selected {method} cell: {status.dataset} "
                    f"{status.surrogate}->{status.target}"
                )
            if set(status.present_seeds) != expected:
                raise RuntimeError(
                    f"incomplete selected {method} cell: {status.dataset} "
                    f"{status.surrogate}->{status.target} has seeds {list(status.present_seeds)}"
                )

            path = Path(status.path)
            diagnostics["input_paths"].append(str(path))
            meta, results = load_cell(path)
            if meta.loss != loss:
                raise ValueError(f"RQ1 expected {loss!r} cells, got {meta.loss!r}")
            if method != meta.method_id:
                raise ValueError(
                    f"method mismatch for {path.name}: expected {method!r}, got {meta.method_id!r}"
                )

            excluded: list[int] = []
            included = 0
            for seed in sorted(results):
                result = results[seed]
                if not result.did_stop:
                    excluded.append(int(seed))
                    continue
                labels, lo, hi = _arrays_for_method(result, method)
                trunc_labels, trunc_widths = _truncate_to_stop(
                    labels, lo, hi, result.labels_to_stop
                )
                trajectories.append(
                    SeedWidthTrajectory(
                        dataset=status.dataset,
                        surrogate=status.surrogate,
                        target=status.target,
                        method=method,
                        seed=int(seed),
                        labels=trunc_labels,
                        widths=trunc_widths,
                    )
                )
                included += 1

            if included == 0:
                raise ValueError(
                    f"selected {method} cell has no stopped seeds after exclusion: "
                    f"{status.dataset} {status.surrogate}->{status.target}"
                )
            diagnostics["included_seed_count"] += included
            diagnostics["excluded_non_stopping_count"] += len(excluded)
            if excluded:
                diagnostics["excluded_non_stoppers"].append(
                    {
                        "method": method,
                        "dataset": status.dataset,
                        "surrogate": status.surrogate,
                        "target": status.target,
                        "seeds": excluded,
                    }
                )

    diagnostics["input_paths"] = sorted(set(diagnostics["input_paths"]))
    return trajectories, diagnostics


def _m5_cell_data_present(
    out_root: Path,
    dataset: str,
    pairs: tuple[SelectedPair, ...],
) -> bool:
    """Strictly data-driven scope predicate for the cereval scope.

    Returns True iff for at least one ``pair in pairs``, EITHER
    (a) the merged M5 cell exists at ``main_cell_path(...)``, OR
    (b) at least one chunked subcell file exists under ``_subcells/main/``
        matching ``subcell__M5__{dataset}__{surrogate}__{target}__accuracy__chunk*.npz``.

    The cfg-planned arm is intentionally excluded — see spec §5.2.
    """
    subcell_dir = out_root / "_subcells" / "main"
    for pair in pairs:
        merged = main_cell_path(
            out_root, method="M5", dataset=dataset,
            surrogate=pair.surrogate, target=pair.target, loss="accuracy",
        )
        if merged.exists():
            return True
        if subcell_dir.exists():
            pattern = (
                f"subcell__M5__{dataset}__{pair.surrogate}__{pair.target}"
                f"__accuracy__chunk*.npz"
            )
            if any(subcell_dir.glob(pattern)):
                return True
    return False


def derive_cereval_scope(
    cfg_cereval,
    out_root_cereval: Path,
    *,
    selection: tuple[SelectedPair, ...] | None = None,
) -> tuple[tuple[str, ...], tuple[SelectedPair, ...]]:
    """Resolve the (datasets, pairs) plot scope for the cereval sub-figure.

    Strictly data-driven on BOTH axes (per spec §5.2):
    - Pairs: intersection of ``selection`` (default ``selected_pairs()``) with
      ``cfg_cereval.cereval_pairs``, then further filtered to pairs that have
      at least one (dataset, pair) tuple with on-disk M5 evidence.
    - Datasets: ``cfg_cereval.datasets ∩ DATASET_ORDER`` filtered to those that
      have at least one in-scope pair with on-disk M5 evidence.

    Today: ({"sst2"}, 2 pairs). Auto-extends if cereval data appears for
    mmlu/agnews or for additional pairs.
    """
    candidate_pairs = cereval_pairs(cfg_cereval, selection=selection)
    candidate_datasets = tuple(
        d for d in DATASET_ORDER if d in cfg_cereval.datasets
    )
    # Filter pairs by per-pair data presence on at least one candidate dataset.
    pairs = tuple(
        p for p in candidate_pairs
        if any(
            _m5_cell_data_present(out_root_cereval, d, (p,))
            for d in candidate_datasets
        )
    )
    # Then filter datasets by data presence over the surviving pairs.
    datasets = tuple(
        d for d in candidate_datasets
        if _m5_cell_data_present(out_root_cereval, d, pairs)
    )
    assert pairs and datasets, (
        f"empty cereval scope: pairs={pairs}, datasets={datasets} "
        f"(out_root_cereval={out_root_cereval})"
    )
    return datasets, pairs


def _cereval_run_indices(
    cfg_cereval,
    dataset: str,
    surrogate: str,
    target: str,
) -> tuple[int, ...]:
    """Locate per-chunk run indices for an M5 cell in the cereval cfg's
    ``cereval-fresh`` stage assignment list (M5-only; index_maps:74-90).
    """
    cells = main_cells_for_loss(
        cfg_cereval, loss="accuracy", include_cereval=True, only_cereval=True,
    )
    assignments = chunked_assignments(cells, cfg_cereval.seeds_main)
    out = [
        idx
        for idx, asn in enumerate(assignments)
        if asn.cell.method_id == "M5"
        and asn.cell.dataset == dataset
        and asn.cell.surrogate == surrogate
        and asn.cell.target == target
        and asn.cell.loss == "accuracy"
    ]
    if not out:
        raise ValueError(
            "could not locate cereval run indices for "
            f"M5 {dataset} {surrogate}->{target}"
        )
    return tuple(out)


def _cereval_merge_index(
    cfg_cereval,
    dataset: str,
    surrogate: str,
    target: str,
) -> int:
    """Locate the merge-cells index for an M5 cell.

    The merge-cells stage iterates targets in this order
    (``src/save/paper_experiment/stages.py:run_merge_cells``, function spans
    lines 612-676; target-construction loop at lines 624-630):

        for loss in cfg.losses:
            for cell in main_cells_for_loss(cfg, loss=loss, include_cereval=True):
                targets.append(("main", cell, list(cfg.seeds_main)))
        for cell in ce_sweep_cells(cfg):
            targets.append(("ce_sweep", cell, list(cfg.seeds_ce_sweep)))
        for cell in beta_sweep_cells(cfg):
            targets.append(("beta_sweep", cell, list(cfg.seeds_beta_sweep)))

    The ``main`` block comes first, and M5 cells only appear within ``main``
    (``ce_sweep_cells`` and ``beta_sweep_cells`` exclude M5 by construction —
    see ``index_maps.py:97-129``). An M5 cell's index in the full target list
    therefore equals its index within the ``main``-only prefix, so this helper
    only needs to reconstruct the ``main`` block.

    The reconstruction-parity test
    (``test_cereval_merge_index_matches_run_merge_cells_targets``) builds the
    full target list (including ce_sweep / beta_sweep) and asserts the helper's
    index matches — this guards against drift if ``stages.py`` ever puts
    ``ce_sweep`` or ``beta_sweep`` before ``main``, or if M5 leaks into them.
    """
    targets = []
    for loss in cfg_cereval.losses:
        for cell in main_cells_for_loss(cfg_cereval, loss=loss, include_cereval=True):
            targets.append(("main", cell))
    for idx, (kind, cell) in enumerate(targets):
        if (
            kind == "main"
            and cell.method_id == "M5"
            and cell.dataset == dataset
            and cell.surrogate == surrogate
            and cell.target == target
            and cell.loss == "accuracy"
        ):
            return idx
    raise ValueError(
        "could not locate cereval merge-cells index for "
        f"M5 {dataset} {surrogate}->{target}"
    )


def collect_selected_cereval_repair_targets(
    cfg_cereval,
    out_root_cereval: Path,
    *,
    selection: tuple[SelectedPair, ...] | None = None,
) -> list[MainRepairTarget]:
    """Surface MainRepairTarget for any in-scope cereval (dataset, pair) that
    has subcells but no complete merged cell."""
    expected = tuple(int(s) for s in cfg_cereval.seeds_main)
    expected_set = set(expected)
    datasets, pairs = derive_cereval_scope(cfg_cereval, out_root_cereval, selection=selection)

    targets: list[MainRepairTarget] = []
    for dataset in datasets:
        for pair in pairs:
            # Per-pair data-presence guard (spec §5.2): only emit repair targets
            # for (dataset, pair) tuples that actually have subcells or a
            # merged cell on disk. Skips silent fallthrough on extras.
            if not _m5_cell_data_present(out_root_cereval, dataset, (pair,)):
                continue
            merged_path = main_cell_path(
                out_root_cereval, method="M5", dataset=dataset,
                surrogate=pair.surrogate, target=pair.target, loss="accuracy",
            )
            present = _present_seeds(merged_path)
            if set(present) == expected_set:
                continue  # already merged & complete
            targets.append(
                MainRepairTarget(
                    method_id="M5",
                    dataset=dataset,
                    surrogate=pair.surrogate,
                    target=pair.target,
                    expected_seeds=expected,
                    present_seeds=present,
                    run_indices=_cereval_run_indices(
                        cfg_cereval,
                        dataset=dataset,
                        surrogate=pair.surrogate,
                        target=pair.target,
                    ),
                    merge_index=_cereval_merge_index(
                        cfg_cereval,
                        dataset=dataset,
                        surrogate=pair.surrogate,
                        target=pair.target,
                    ),
                )
            )
    return targets
