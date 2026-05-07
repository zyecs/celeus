# src/save/paper_experiment/cell_runner.py
"""Run every seed for one cell and write one .npz (paper_experiment spec §4)."""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable, Optional

import numpy as np

from save.partition import EvaluationPool
from save.core.state import SAVEConfig

from .cell_paths import (
    acquisition_sweep_cell_path, acquisition_sweep_subcell_path,
    beta_sweep_cell_path, beta_sweep_subcell_path,
    ce_sweep_cell_path, ce_sweep_subcell_path,
    hparam_sweep_cell_path, hparam_sweep_subcell_path,
    main_cell_path, main_subcell_path,
    oracle_accuracy_cell_path, oracle_accuracy_subcell_path,
    wallclock_cell_path, wallclock_subcell_path,
)
from .cell_schema import CellMetadata, load_cell, save_cell
from .runners import run_cereval_for_seed, run_save_method_for_seed


def _target_path(kind: str, out_base: Path, **kwargs) -> Path:
    if kind == "main":
        return main_cell_path(out_base, **kwargs)
    if kind == "ce_sweep":
        return ce_sweep_cell_path(out_base, **kwargs)
    if kind == "beta_sweep":
        return beta_sweep_cell_path(out_base, **kwargs)
    if kind == "oracle_accuracy":
        return oracle_accuracy_cell_path(out_base, **kwargs)
    if kind == "acquisition_sweep":
        return acquisition_sweep_cell_path(out_base, **kwargs)
    if kind == "hparam_sweep":
        return hparam_sweep_cell_path(out_base, **kwargs)
    if kind == "wallclock":
        return wallclock_cell_path(out_base, **kwargs)
    raise ValueError(f"unknown kind: {kind!r}")


def run_cell(
    *,
    out_base: Path,
    method_id: str,
    dataset: str,
    surrogate: str,
    target: str,
    loss: str,
    seeds: Iterable[int],
    T_max: int,
    pool_factory: Callable[[], EvaluationPool],
    epsilon: float,
    alpha_1: float,
    alpha_2: float,
    theta: float,
    c_betting: float,
    c_fixed: float,
    cs_grid_size: int,
    beta_min: float,
    surrogate_type: str,
    adaptive_bounds: bool,
    uniform_acquisition: bool,
    monitor_to_T_max: bool,
    cereval_m_init: int = 100,
    kind: str = "main",
    # sweep-specific filename axes (only used when kind != "main"):
    surrogate_type_for_path: Optional[str] = None,
    config_name_for_path: Optional[str] = None,
) -> Path:
    """Execute all seeds for one cell and atomically write a .npz file."""
    pool = pool_factory()
    seeds = list(int(s) for s in seeds)

    results = {}
    for seed in seeds:
        if method_id == "M5":
            pseed = run_cereval_for_seed(
                pool=pool, T_max=T_max, epsilon=epsilon,
                alpha_1=alpha_1, alpha_2=alpha_2, m_init=cereval_m_init,
                seed=seed, monitor_to_T_max=monitor_to_T_max,
            )
        else:
            pseed = run_save_method_for_seed(
                method_id=method_id, pool=pool, T_max=T_max, loss_type=loss,
                beta_min=beta_min, surrogate_type=surrogate_type,
                adaptive_bounds=adaptive_bounds, uniform_acquisition=uniform_acquisition,
                epsilon=epsilon, alpha_1=alpha_1, alpha_2=alpha_2, theta=theta,
                c_betting=c_betting, c_fixed=c_fixed, cs_grid_size=cs_grid_size,
                monitor_to_T_max=monitor_to_T_max, seed=seed,
            )
        results[seed] = pseed

    _filter_meta = pool.metadata.get("ce_nll_filter")  # dict or None
    meta = CellMetadata(
        method_id=method_id, dataset=dataset,
        surrogate=surrogate, target=target, loss=loss,
        T_max=T_max, epsilon=epsilon, beta_min=beta_min,
        surrogate_type=surrogate_type, adaptive_bounds=adaptive_bounds,
        seeds=np.asarray(seeds, dtype=np.int64),
        ce_nll_filter_threshold=(
            _filter_meta["threshold"] if _filter_meta else None
        ),
        ce_nll_filter_kept=(
            _filter_meta["kept"] if _filter_meta else None
        ),
        ce_nll_filter_original_n=(
            _filter_meta["original_n"] if _filter_meta else None
        ),
    )

    if kind == "main":
        path = main_cell_path(
            out_base, method=method_id, dataset=dataset,
            surrogate=surrogate, target=target, loss=loss,
        )
    elif kind == "ce_sweep":
        path = ce_sweep_cell_path(
            out_base, dataset=dataset, surrogate=surrogate, target=target,
            surrogate_type=surrogate_type_for_path or surrogate_type,
            beta_min=beta_min,
        )
    elif kind == "beta_sweep":
        path = beta_sweep_cell_path(
            out_base, dataset=dataset, surrogate=surrogate, target=target,
            loss=loss, beta_min=beta_min,
        )
    elif kind == "oracle_accuracy":
        path = oracle_accuracy_cell_path(
            out_base, dataset=dataset, surrogate=surrogate, target=target,
            surrogate_type=surrogate_type_for_path or surrogate_type,
        )
    elif kind == "acquisition_sweep":
        path = acquisition_sweep_cell_path(
            out_base, dataset=dataset, surrogate=surrogate, target=target,
            loss=loss,
            surrogate_type=surrogate_type_for_path or surrogate_type,
        )
    elif kind == "hparam_sweep":
        if config_name_for_path is None:
            raise ValueError(
                "config_name_for_path is required for kind='hparam_sweep'"
            )
        path = hparam_sweep_cell_path(
            out_base, dataset=dataset, surrogate=surrogate, target=target,
            loss=loss, config_name=config_name_for_path,
        )
    elif kind == "wallclock":
        path = wallclock_cell_path(
            out_base, dataset=dataset, surrogate=surrogate, target=target,
            loss=loss,
        )
    else:
        raise ValueError(f"unknown kind: {kind!r}")

    # Atomic: write temp file, rename into place.
    # IMPORTANT: the tmp filename MUST end in ``.npz`` because
    # np.savez_compressed appends ``.npz`` to any filename that does not
    # already end in it — producing ``*.npz.tmp.npz`` and breaking the
    # subsequent rename. Use ``<stem>.tmp.npz`` instead.
    tmp = path.with_name(path.stem + ".tmp.npz")
    save_cell(tmp, meta, results)
    tmp.replace(path)
    return path


def _subcell_path_for_kind(kind: str, out_base: Path, **keys) -> Path:
    if kind == "main":
        return main_subcell_path(out_base, **keys)
    if kind == "ce_sweep":
        return ce_sweep_subcell_path(out_base, **keys)
    if kind == "beta_sweep":
        return beta_sweep_subcell_path(out_base, **keys)
    if kind == "oracle_accuracy":
        return oracle_accuracy_subcell_path(out_base, **keys)
    if kind == "acquisition_sweep":
        return acquisition_sweep_subcell_path(out_base, **keys)
    if kind == "hparam_sweep":
        return hparam_sweep_subcell_path(out_base, **keys)
    if kind == "wallclock":
        return wallclock_subcell_path(out_base, **keys)
    raise ValueError(f"unknown kind: {kind!r}")


def _final_cell_path_for_kind(kind: str, out_base: Path, **keys) -> Path:
    if kind == "main":
        return main_cell_path(out_base, **keys)
    if kind == "ce_sweep":
        return ce_sweep_cell_path(out_base, **keys)
    if kind == "beta_sweep":
        return beta_sweep_cell_path(out_base, **keys)
    if kind == "oracle_accuracy":
        return oracle_accuracy_cell_path(out_base, **keys)
    if kind == "acquisition_sweep":
        return acquisition_sweep_cell_path(out_base, **keys)
    if kind == "hparam_sweep":
        return hparam_sweep_cell_path(out_base, **keys)
    if kind == "wallclock":
        return wallclock_cell_path(out_base, **keys)
    raise ValueError(f"unknown kind: {kind!r}")


def run_subcell(
    *,
    out_base: Path,
    method_id: str,
    dataset: str,
    surrogate: str,
    target: str,
    loss: str,
    seeds: list,
    T_max: int,
    pool_factory,
    epsilon: float,
    alpha_1: float,
    alpha_2: float,
    theta: float,
    c_betting: float,
    c_fixed: float,
    cs_grid_size: int,
    beta_min: float,
    surrogate_type: str,
    adaptive_bounds: bool,
    uniform_acquisition: bool,
    monitor_to_T_max: bool,
    cereval_m_init: int,
    chunk: int,
    kind: str,
    surrogate_type_for_path: Optional[str] = None,
    config_name_for_path: Optional[str] = None,
) -> Path:
    """Run seeds for one sub-cell chunk and write an intermediate .npz."""
    pool = pool_factory()

    from .pool_loader import pool_sha256 as _pool_sha
    sha = _pool_sha(pool)

    seeds = [int(s) for s in seeds]
    results = {}
    for seed in seeds:
        if method_id == "M5":
            pseed = run_cereval_for_seed(
                pool=pool, T_max=T_max, epsilon=epsilon,
                alpha_1=alpha_1, alpha_2=alpha_2, m_init=cereval_m_init,
                seed=seed, monitor_to_T_max=monitor_to_T_max,
            )
        else:
            pseed = run_save_method_for_seed(
                method_id=method_id, pool=pool, T_max=T_max, loss_type=loss,
                beta_min=beta_min, surrogate_type=surrogate_type,
                adaptive_bounds=adaptive_bounds, uniform_acquisition=uniform_acquisition,
                epsilon=epsilon, alpha_1=alpha_1, alpha_2=alpha_2, theta=theta,
                c_betting=c_betting, c_fixed=c_fixed, cs_grid_size=cs_grid_size,
                monitor_to_T_max=monitor_to_T_max, seed=seed,
            )
        results[seed] = pseed

    import json
    _default_cfg = SAVEConfig()
    config_json = json.dumps({
        "method_id": method_id, "dataset": dataset,
        "surrogate": surrogate, "target": target, "loss": loss,
        "T_max": T_max, "epsilon": epsilon,
        "alpha_1": alpha_1, "alpha_2": alpha_2,
        "theta": theta, "c_betting": c_betting, "c_fixed": c_fixed,
        "cs_grid_size": cs_grid_size, "beta_min": beta_min,
        "surrogate_type": surrogate_type,
        "adaptive_bounds": adaptive_bounds,
        "uniform_acquisition": uniform_acquisition,
        "monitor_to_T_max": monitor_to_T_max, "cereval_m_init": cereval_m_init,
        "chunk": chunk, "seeds": seeds,
        # rq4-rq6 design §5.6.1: persist single-draw invariants + NumPy version
        # so replay can hard-gate on future cells. Legacy cells lack these keys.
        "K": _default_cfg.K,
        "B_round": _default_cfg.B_round,
        "numpy_version": np.__version__,
    })

    _filter_meta = pool.metadata.get("ce_nll_filter")  # dict or None
    meta = CellMetadata(
        method_id=method_id, dataset=dataset,
        surrogate=surrogate, target=target, loss=loss,
        T_max=T_max, epsilon=epsilon, beta_min=beta_min,
        surrogate_type=surrogate_type, adaptive_bounds=adaptive_bounds,
        seeds=np.asarray(seeds, dtype=np.int64),
        pool_sha256=sha, config_json=config_json,
        ce_nll_filter_threshold=(
            _filter_meta["threshold"] if _filter_meta else None
        ),
        ce_nll_filter_kept=(
            _filter_meta["kept"] if _filter_meta else None
        ),
        ce_nll_filter_original_n=(
            _filter_meta["original_n"] if _filter_meta else None
        ),
    )

    if kind == "main":
        keys = dict(method=method_id, dataset=dataset, surrogate=surrogate,
                    target=target, loss=loss)
    elif kind == "ce_sweep":
        keys = dict(dataset=dataset, surrogate=surrogate, target=target,
                    surrogate_type=surrogate_type_for_path or surrogate_type,
                    beta_min=beta_min)
    elif kind == "beta_sweep":
        keys = dict(dataset=dataset, surrogate=surrogate, target=target,
                    loss=loss, beta_min=beta_min)
    elif kind == "oracle_accuracy":
        keys = dict(
            dataset=dataset, surrogate=surrogate, target=target,
            surrogate_type=surrogate_type_for_path or surrogate_type,
        )
    elif kind == "acquisition_sweep":
        keys = dict(
            dataset=dataset, surrogate=surrogate, target=target,
            loss=loss,
            surrogate_type=surrogate_type_for_path or surrogate_type,
        )
    elif kind == "hparam_sweep":
        if config_name_for_path is None:
            raise ValueError(
                "config_name_for_path is required for kind='hparam_sweep'"
            )
        keys = dict(
            dataset=dataset, surrogate=surrogate, target=target,
            loss=loss, config_name=config_name_for_path,
        )
    elif kind == "wallclock":
        keys = dict(
            dataset=dataset, surrogate=surrogate, target=target,
            loss=loss,
        )
    else:
        raise ValueError(f"unknown kind: {kind!r}")

    path = _subcell_path_for_kind(kind, out_base, chunk=chunk, **keys)
    tmp = path.with_name(path.stem + ".tmp.npz")
    save_cell(tmp, meta, results)
    tmp.replace(path)
    return path


def merge_subcells(
    *,
    out_base: Path,
    kind: str,
    method_id: str = "",
    dataset: str,
    surrogate: str,
    target: str,
    loss: str = "",
    surrogate_type: str = "",
    beta_min: float = 0.0,
    config_name: str = "",
    expected_seeds: list,
    delete_subcells: bool = True,
) -> Path:
    """Merge sub-cells of one cell into a single bundled cell .npz."""
    # Locate sub-cells by glob. Keep the per-kind keys consistent with
    # run_subcell's filename scheme.
    subcells_dir = out_base / "_subcells" / kind
    if kind == "main":
        prefix = "subcell__" + "__".join([method_id, dataset, surrogate, target, loss])
    elif kind == "ce_sweep":
        beta_tok = f"beta{float(beta_min):g}"
        prefix = "subcell__" + "__".join([dataset, surrogate, target, surrogate_type, beta_tok])
    elif kind == "beta_sweep":
        beta_tok = f"beta{float(beta_min):g}"
        prefix = "subcell__" + "__".join([dataset, surrogate, target, loss, beta_tok])
    elif kind == "oracle_accuracy":
        prefix = "subcell__" + "__".join([
            "oracle_accuracy", dataset, surrogate, target, surrogate_type,
        ])
    elif kind == "acquisition_sweep":
        prefix = "subcell__" + "__".join([
            "acquisition_sweep", dataset, surrogate, target, loss,
            surrogate_type,
        ])
    elif kind == "hparam_sweep":
        prefix = "subcell__" + "__".join([
            "hparam_sweep", dataset, surrogate, target, loss,
            config_name,
        ])
    elif kind == "wallclock":
        prefix = "subcell__" + "__".join([
            "wallclock", dataset, surrogate, target, loss,
        ])
    else:
        raise ValueError(f"unknown kind: {kind!r}")

    subcells = sorted(subcells_dir.glob(prefix + "__chunk*.npz"))
    if not subcells:
        raise FileNotFoundError(f"no sub-cells for prefix {prefix!r} in {subcells_dir}")

    combined: dict = {}
    meta0 = None
    for sc in subcells:
        m, r = load_cell(sc)
        if meta0 is None:
            meta0 = m
        combined.update(r)

    expected = set(int(s) for s in expected_seeds)
    have = set(combined.keys())
    missing = expected - have
    if missing:
        raise RuntimeError(
            f"merge {prefix}: missing seeds {sorted(missing)} "
            f"(have {sorted(have)})"
        )

    # Build merged metadata with the full expected seed set (sorted).
    full_seeds = np.asarray(sorted(expected_seeds), dtype=np.int64)
    assert meta0 is not None
    merged_meta = CellMetadata(
        method_id=meta0.method_id, dataset=meta0.dataset,
        surrogate=meta0.surrogate, target=meta0.target, loss=meta0.loss,
        T_max=meta0.T_max, epsilon=meta0.epsilon, beta_min=meta0.beta_min,
        surrogate_type=meta0.surrogate_type,
        adaptive_bounds=meta0.adaptive_bounds,
        seeds=full_seeds,
        pool_sha256=meta0.pool_sha256, config_json=meta0.config_json,
        ce_nll_filter_threshold=meta0.ce_nll_filter_threshold,
        ce_nll_filter_kept=meta0.ce_nll_filter_kept,
        ce_nll_filter_original_n=meta0.ce_nll_filter_original_n,
    )

    if kind == "main":
        keys = dict(method=method_id, dataset=dataset,
                    surrogate=surrogate, target=target, loss=loss)
    elif kind == "ce_sweep":
        keys = dict(dataset=dataset, surrogate=surrogate, target=target,
                    surrogate_type=surrogate_type, beta_min=beta_min)
    elif kind == "oracle_accuracy":
        keys = dict(
            dataset=dataset, surrogate=surrogate, target=target,
            surrogate_type=surrogate_type,
        )
    elif kind == "acquisition_sweep":
        keys = dict(
            dataset=dataset, surrogate=surrogate, target=target,
            loss=loss, surrogate_type=surrogate_type,
        )
    elif kind == "hparam_sweep":
        keys = dict(
            dataset=dataset, surrogate=surrogate, target=target,
            loss=loss, config_name=config_name,
        )
    elif kind == "wallclock":
        keys = dict(
            dataset=dataset, surrogate=surrogate, target=target,
            loss=loss,
        )
    else:
        keys = dict(dataset=dataset, surrogate=surrogate, target=target,
                    loss=loss, beta_min=beta_min)

    final_path = _final_cell_path_for_kind(kind, out_base, **keys)
    tmp = final_path.with_name(final_path.stem + ".tmp.npz")
    save_cell(tmp, merged_meta, combined)
    tmp.replace(final_path)

    if delete_subcells:
        for sc in subcells:
            sc.unlink()

    return final_path
