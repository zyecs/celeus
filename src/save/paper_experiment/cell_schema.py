# src/save/paper_experiment/cell_schema.py
"""Cell-bundled .npz schema (paper_experiment spec §4).

v2 fields added (over v1):
  - did_stop: bool — whether pop_w ever crossed epsilon.
  - width_at_stop: float — CS width at the first-crossing index; NaN if never.
  - pop_inverted_count: int — number of rounds where pop_lower > pop_upper
    (pool-feasibility conflict at early t, diagnostic only).
  - pool_sha256: str — SHA-256 of the first-permutation pool indices (spec §9).
  - config_json: str — JSON-serialised full run config (spec §8).
Analysis code MUST filter on ``did_stop=True`` before aggregating
``coverage_at_stop`` / ``labels_to_stop`` / ``width_at_stop``.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Optional

import numpy as np


_TRAJ_FLOAT_KEYS = ("save_rhat", "save_lo", "save_hi",
                    "base_rhat", "base_lo", "base_hi")
_TRAJ_INT_KEYS = ("save_labels", "base_labels")
# Variable-length float trajectory key (only populated for M5 / Cer-Eval).
# Padded to T_max with NaN; an all-NaN row decodes back to ``None`` because
# non-M5 methods do not produce per-round timing. (paper §6.5 item #5, Task 9).
_TRAJ_OPTIONAL_FLOAT_KEYS = ("round_times",)
_PER_SEED_FLOAT_SCALARS = (
    "true_R", "rho", "final_width", "width_at_stop", "elapsed_seconds",
)
_PER_SEED_INT_SCALARS = ("labels_to_stop", "pop_inverted_count")
_PER_SEED_BOOL_SCALARS = ("coverage_at_stop", "ever_miss", "did_stop")
_PER_SEED_STR_SCALARS = ("git_commit", "hostname")

# Sentinels used to round-trip Optional[float] / Optional[int] through npz
# serialization. np.savez_compressed under allow_pickle=False cannot store a
# Python None (it would become an object-dtype array at save time and fail
# at load time). We encode absence as NaN for floats and -1 for ints, then
# decode back to None in load_cell.
_FLOAT_ABSENT = float("nan")
_INT_ABSENT = -1


@dataclass
class PerSeedResult:
    """One seed's full-trajectory output (shape-(T_max,) arrays) + scalars."""
    save_labels: np.ndarray
    save_rhat: np.ndarray
    save_lo: np.ndarray
    save_hi: np.ndarray
    base_labels: np.ndarray
    base_rhat: np.ndarray
    base_lo: np.ndarray
    base_hi: np.ndarray
    true_R: float
    rho: float
    did_stop: bool
    labels_to_stop: int
    width_at_stop: float
    final_width: float
    coverage_at_stop: bool
    ever_miss: bool
    pop_inverted_count: int
    elapsed_seconds: float
    git_commit: str
    hostname: str
    # Paper §6.5 item #5 (Task 9): per-iteration wall-clock for Cer-Eval (M5).
    # Variable-length raw values (one entry per Cer-Eval ``while True:`` round).
    # Padded to T_max with NaN at serialization for npz round-trip; trailing
    # NaNs encode "not recorded" (loop exited earlier). ``None`` for non-M5
    # methods — those record per-method timing via ``elapsed_seconds`` /
    # ``labels_to_stop`` already.
    round_times: Optional[np.ndarray] = None


@dataclass
class CellMetadata:
    method_id: str
    dataset: str
    surrogate: str
    target: str
    loss: str
    T_max: int
    epsilon: float
    beta_min: float
    surrogate_type: str
    adaptive_bounds: bool
    seeds: np.ndarray
    pool_sha256: str = ""
    config_json: str = ""
    # CE NLL filter audit fields — None when filter inactive (accuracy loss or
    # filter disabled in config). Serialized via sentinels (NaN / -1).
    ce_nll_filter_threshold: Optional[float] = None
    ce_nll_filter_kept: Optional[int] = None
    ce_nll_filter_original_n: Optional[int] = None


def compute_labels_to_stop(
    pop_lower: np.ndarray,
    pop_upper: np.ndarray,
    total_labels: np.ndarray,
    epsilon: float,
) -> int:
    """First ``total_labels`` where ``0 <= pop_upper - pop_lower <= epsilon``; -1 if never.

    Degenerate rows (``pop_upper < pop_lower`` from pool-feasibility
    conflicts at early t) are skipped — they don't count as crossings
    even though their nominal width is <= epsilon.
    """
    widths = pop_upper - pop_lower
    hits = np.where((widths >= 0) & (widths <= epsilon))[0]
    if hits.size == 0:
        return -1
    return int(total_labels[hits[0]])


def _empty_traj(T_max: int, dtype) -> np.ndarray:
    if dtype == np.int64:
        return np.full(T_max, -1, dtype=np.int64)
    return np.full(T_max, np.nan, dtype=np.float64)


def save_cell(path: Path, meta: CellMetadata, results: dict[int, PerSeedResult]) -> None:
    """Write one cell .npz. Missing seeds → NaN / -1 rows."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    T = meta.T_max
    seeds = np.asarray(meta.seeds, dtype=np.int64)
    n = len(seeds)
    buf: dict[str, np.ndarray] = {}
    for k in _TRAJ_FLOAT_KEYS:
        buf[k] = np.full((n, T), np.nan, dtype=np.float64)
    for k in _TRAJ_INT_KEYS:
        buf[k] = np.full((n, T), -1, dtype=np.int64)
    for k in _TRAJ_OPTIONAL_FLOAT_KEYS:
        # All-NaN sentinel decodes back to None on load.
        buf[k] = np.full((n, T), np.nan, dtype=np.float64)
    for k in _PER_SEED_FLOAT_SCALARS:
        buf[k] = np.full(n, np.nan, dtype=np.float64)
    for k in _PER_SEED_INT_SCALARS:
        buf[k] = np.full(n, -1, dtype=np.int64)
    for k in _PER_SEED_BOOL_SCALARS:
        buf[k] = np.zeros(n, dtype=bool)
    for k in _PER_SEED_STR_SCALARS:
        buf[k] = np.full(n, "", dtype="U40")

    present = np.zeros(n, dtype=bool)
    for i, s in enumerate(seeds):
        s = int(s)
        if s not in results:
            continue
        present[i] = True
        r = results[s]
        for k in _TRAJ_FLOAT_KEYS + _TRAJ_INT_KEYS:
            arr = getattr(r, k)
            if arr.shape[0] != T:
                raise ValueError(
                    f"seed {s}: array {k} has shape {arr.shape}; expected ({T},)"
                )
            buf[k][i] = arr
        # Optional variable-length trajectories (e.g., Cer-Eval round_times).
        # Only Cer-Eval populates these; other methods leave the field as
        # ``None`` and the slot stays all-NaN.
        for k in _TRAJ_OPTIONAL_FLOAT_KEYS:
            arr_opt = getattr(r, k, None)
            if arr_opt is None:
                continue
            arr_opt = np.asarray(arr_opt, dtype=np.float64)
            if arr_opt.shape[0] > T:
                raise ValueError(
                    f"seed {s}: optional array {k} has length "
                    f"{arr_opt.shape[0]} > T_max={T}"
                )
            buf[k][i, : arr_opt.shape[0]] = arr_opt
        for k in _PER_SEED_FLOAT_SCALARS + _PER_SEED_INT_SCALARS + _PER_SEED_BOOL_SCALARS:
            buf[k][i] = getattr(r, k)
        for k in _PER_SEED_STR_SCALARS:
            buf[k][i] = str(getattr(r, k))[:40]

    # Sentinel-swap the 3 optional ce_nll_filter_* fields before asdict() so
    # np.savez_compressed (allow_pickle=False) never encounters Python None.
    meta = replace(
        meta,
        ce_nll_filter_threshold=(
            _FLOAT_ABSENT
            if meta.ce_nll_filter_threshold is None
            else float(meta.ce_nll_filter_threshold)
        ),
        ce_nll_filter_kept=(
            _INT_ABSENT
            if meta.ce_nll_filter_kept is None
            else int(meta.ce_nll_filter_kept)
        ),
        ce_nll_filter_original_n=(
            _INT_ABSENT
            if meta.ce_nll_filter_original_n is None
            else int(meta.ce_nll_filter_original_n)
        ),
    )

    meta_dict = asdict(meta)
    meta_dict["seeds"] = seeds
    meta_dict["adaptive_bounds"] = np.bool_(meta.adaptive_bounds)

    np.savez_compressed(
        path,
        present=present,
        **buf,
        **{f"meta__{k}": np.asarray(v) for k, v in meta_dict.items()},
    )


def load_cell(path: Path) -> tuple[CellMetadata, dict[int, PerSeedResult]]:
    """Read one cell .npz. Returns (metadata, {seed: PerSeedResult}) over PRESENT seeds."""
    with np.load(path, allow_pickle=False) as f:
        meta_dict = {
            k.removeprefix("meta__"): f[k] for k in f.files if k.startswith("meta__")
        }
        # Round-trip CE NLL filter audit fields: sentinel → None.
        # Absent keys (pre-v8 cells) default to None via .get().
        thr_arr = meta_dict.get("ce_nll_filter_threshold")
        kept_arr = meta_dict.get("ce_nll_filter_kept")
        orig_arr = meta_dict.get("ce_nll_filter_original_n")
        if thr_arr is None:
            thr_val = None
        else:
            thr_float = float(thr_arr)
            thr_val = None if np.isnan(thr_float) else thr_float
        if kept_arr is None:
            kept_val = None
        else:
            kept_int = int(kept_arr)
            kept_val = None if kept_int == _INT_ABSENT else kept_int
        if orig_arr is None:
            orig_val = None
        else:
            orig_int = int(orig_arr)
            orig_val = None if orig_int == _INT_ABSENT else orig_int

        meta = CellMetadata(
            method_id=str(meta_dict["method_id"]),
            dataset=str(meta_dict["dataset"]),
            surrogate=str(meta_dict["surrogate"]),
            target=str(meta_dict["target"]),
            loss=str(meta_dict["loss"]),
            T_max=int(meta_dict["T_max"]),
            epsilon=float(meta_dict["epsilon"]),
            beta_min=float(meta_dict["beta_min"]),
            surrogate_type=str(meta_dict["surrogate_type"]),
            adaptive_bounds=bool(meta_dict["adaptive_bounds"]),
            seeds=np.asarray(meta_dict["seeds"], dtype=np.int64),
            pool_sha256=str(meta_dict.get("pool_sha256", "")),
            config_json=str(meta_dict.get("config_json", "")),
            ce_nll_filter_threshold=thr_val,
            ce_nll_filter_kept=kept_val,
            ce_nll_filter_original_n=orig_val,
        )
        present = np.asarray(f["present"], dtype=bool)
        buf = {k: f[k] for k in (
            _TRAJ_FLOAT_KEYS + _TRAJ_INT_KEYS
            + _PER_SEED_FLOAT_SCALARS + _PER_SEED_INT_SCALARS
            + _PER_SEED_BOOL_SCALARS + _PER_SEED_STR_SCALARS
        )}
        # Optional variable-length trajectories (Task 9). Pre-Task-9 cells
        # don't have this key; ``.get()`` defaults to None which decodes to
        # ``round_times=None`` per seed.
        opt_buf = {
            k: f[k] for k in _TRAJ_OPTIONAL_FLOAT_KEYS if k in f.files
        }
    out: dict[int, PerSeedResult] = {}
    for i, s in enumerate(meta.seeds):
        if not bool(present[i]):
            continue
        # Decode optional variable-length round_times: trim trailing NaNs;
        # an all-NaN row means the seed did not produce timing (non-M5).
        if "round_times" in opt_buf:
            row = np.asarray(opt_buf["round_times"][i], dtype=np.float64)
            valid = ~np.isnan(row)
            if not np.any(valid):
                round_times_val: Optional[np.ndarray] = None
            else:
                # Last non-NaN index + 1 → length.
                last = int(np.max(np.flatnonzero(valid))) + 1
                round_times_val = row[:last].copy()
        else:
            round_times_val = None
        out[int(s)] = PerSeedResult(
            save_labels=buf["save_labels"][i].copy(),
            save_rhat=buf["save_rhat"][i].copy(),
            save_lo=buf["save_lo"][i].copy(),
            save_hi=buf["save_hi"][i].copy(),
            base_labels=buf["base_labels"][i].copy(),
            base_rhat=buf["base_rhat"][i].copy(),
            base_lo=buf["base_lo"][i].copy(),
            base_hi=buf["base_hi"][i].copy(),
            true_R=float(buf["true_R"][i]),
            rho=float(buf["rho"][i]),
            did_stop=bool(buf["did_stop"][i]),
            labels_to_stop=int(buf["labels_to_stop"][i]),
            width_at_stop=float(buf["width_at_stop"][i]),
            final_width=float(buf["final_width"][i]),
            coverage_at_stop=bool(buf["coverage_at_stop"][i]),
            ever_miss=bool(buf["ever_miss"][i]),
            pop_inverted_count=int(buf["pop_inverted_count"][i]),
            elapsed_seconds=float(buf["elapsed_seconds"][i]),
            git_commit=str(buf["git_commit"][i]),
            hostname=str(buf["hostname"][i]),
            round_times=round_times_val,
        )
    return meta, out
