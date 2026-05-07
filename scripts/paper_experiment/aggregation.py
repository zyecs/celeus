"""Aggregation helpers for rq4-rq6 plotting (spec §8).

Single source of truth for:
  - parse_cell_filename: regex + post-classification for 3 trajectory dir formats
  - per_dataset_pool:    per-section cell-normalization + dataset pooling
  - per_pair_filter:     filter to PAIR_DEFS demo pairs

Imports from scripts.paper_experiment.plot_style for PAIR_DEFS only.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

_log = logging.getLogger(__name__)

# Round-2 fix: original regex hardcoded loss=accuracy|cross_entropy and failed
# on oracle_accuracy filenames whose tail is e.g. remark2_oracle_strategy4.
# Solution: capture the tail un-anchored, then post-classify based on method
# name. Verified live on 3 trajectory dir formats.
_CELL_RE = re.compile(
    r"^cell__"
    r"(?P<method>[A-Za-z0-9_]+?)__"
    r"(?P<dataset>[A-Za-z0-9]+)__"
    r"(?P<surrogate>[A-Za-z0-9_]+?)__"
    r"(?P<target>[A-Za-z0-9_]+?)__"
    r"(?P<tail>[A-Za-z0-9_]+)"
    r"\.npz$"
)

_LOSS_TOKENS = {"accuracy", "cross_entropy"}


def parse_cell_filename(cell_path) -> dict:
    """Parse a cell filename into {method, dataset, surrogate, target, loss,
    surrogate_type}. Accepts str or Path-like. Raises ValueError on no match.

    Covers main-corpus and oracle_accuracy formats:
      cell__M1__sst2__llama2_7b__Mixtral_8x7b__accuracy.npz
        -> method=M1, loss=accuracy, surrogate_type=None
      cell__M3__agnews__llama3_8b__qwen25_72b__cross_entropy.npz
        -> method=M3, loss=cross_entropy, surrogate_type=None
      cell__oracle_accuracy__sst2__llama3_8b__deepseek_67b__remark2_oracle_strategy4.npz
        -> method=oracle_accuracy, loss=accuracy (derived), surrogate_type=remark2_oracle_strategy4

    Post-classification rules for `tail`:
      - tail in {"accuracy", "cross_entropy"} -> loss = tail; surrogate_type = None
      - method.startswith("oracle_") -> loss derived from method; surrogate_type = tail
      - else -> ValueError
    """
    name = Path(cell_path).name
    m = _CELL_RE.match(name)
    if m is None:
        raise ValueError(f"unparsable cell filename: {name}")
    parts = m.groupdict()
    tail = parts.pop("tail")
    method = parts["method"]
    if tail in _LOSS_TOKENS:
        parts["loss"] = tail
        parts["surrogate_type"] = None
    elif method.startswith("oracle_"):
        derived_loss = method.split("_", 1)[1]
        if derived_loss not in _LOSS_TOKENS:
            raise ValueError(
                f"oracle method {method!r} does not encode a recognised loss "
                f"({_LOSS_TOKENS}); cannot parse {name}"
            )
        parts["loss"] = derived_loss
        parts["surrogate_type"] = tail
    else:
        raise ValueError(
            f"unrecognised tail {tail!r} for non-oracle method {method!r}: {name}"
        )
    return parts


def _annotate_dataset_pair(df: pd.DataFrame) -> pd.DataFrame:
    """Add `dataset`, `surrogate`, `target` columns parsed from `cell` filename."""
    out = df.copy()
    parsed = out["cell"].apply(parse_cell_filename).apply(pd.Series)
    for col in ("dataset", "surrogate", "target"):
        out[col] = parsed[col]
    return out


def _drop_unequal_M_c(df: pd.DataFrame, expected_M_c: int = 50) -> pd.DataFrame:
    """Drop cells with seed-count != expected_M_c, scoped per-dataset.
    WARNs (does not HALT). One bad agnews cell never blocks sst2/mmlu rendering.
    """
    keep_mask = pd.Series(True, index=df.index)
    for dataset, sub in df.groupby("dataset"):
        m_cs = sub.groupby("cell")["seed"].nunique()
        bad = m_cs[m_cs != expected_M_c]
        if len(bad) > 0:
            _log.warning(
                "%s: dropping %d cell(s) with M_c != %d: %s",
                dataset, len(bad), expected_M_c, bad.index.tolist(),
            )
            keep_mask &= ~df["cell"].isin(bad.index)
    return df[keep_mask].copy()


def _pool_rq4(
    df: pd.DataFrame,
    *,
    rhat_cols: dict[str, str],
    rn_col: str,
) -> pd.DataFrame:
    """rq4 unbiasedness aggregation per spec §8.2.

    Per-cell: bias_per_cell(c, t, π, kind) = mean_seed( df[rhat_cols[kind]] − df[rn_col] )
    Per-cell SE: sd_seed(rhat − rn) / sqrt(M_c)
    Pool across cells within dataset:
      bias_pooled = mean_c
      MC_SE_pooled = sqrt((1/|C|^2) Σ_c MC_SE_per_cell^2)
      SD_between   = std_c(bias_per_cell, ddof=1)
    """
    rows = []
    for kind, rhat_col in rhat_cols.items():
        # Per-row error (cell-normalized)
        df_kind = df.assign(_err=df[rhat_col] - df[rn_col])
        # Per-cell mean and SE
        per_cell = (
            df_kind.groupby(["dataset", "acquisition", "cell", "t"])["_err"]
            .agg(_mean="mean", _sd=lambda v: v.std(ddof=1), _n="count")
            .reset_index()
        )
        per_cell["_se"] = per_cell["_sd"] / np.sqrt(per_cell["_n"])
        # Pool across cells within dataset
        for (dataset, acq, t), grp in per_cell.groupby(["dataset", "acquisition", "t"]):
            n_c = len(grp)
            bias_pooled = float(grp["_mean"].mean())
            mc_se_pooled = float(np.sqrt((grp["_se"] ** 2).sum() / n_c ** 2))
            sd_between = (
                float(grp["_mean"].std(ddof=1)) if n_c > 1 else 0.0
            )
            rows.append({
                "dataset": dataset, "acquisition": acq, "kind": kind, "t": int(t),
                "bias_pooled": bias_pooled,
                "MC_SE_pooled": mc_se_pooled,
                "SD_between": sd_between,
                "n_cells": n_c,
            })
    return pd.DataFrame(rows)


def _pool_rq5(df: pd.DataFrame, *, sq_err_cols: list[str]) -> pd.DataFrame:
    """rq5 signal MSE aggregation per spec §8.2.

    The is_sq and naive_sq columns are pre-computed per-seed at write time
    (R_N already subtracted-then-squared). So step 3 is just per-cell mean
    across seeds, then per-dataset pool across cells.
    """
    is_col, naive_col = sq_err_cols  # ["is_sq", "naive_sq"]
    per_cell = (
        df.groupby(["dataset", "cell", "t"])
        .agg(
            is_mean=(is_col, "mean"),
            is_sd=(is_col, lambda v: v.std(ddof=1)),
            naive_mean=(naive_col, "mean"),
            naive_sd=(naive_col, lambda v: v.std(ddof=1)),
            n_seeds=(is_col, "count"),
        )
        .reset_index()
    )
    per_cell["is_se"] = per_cell["is_sd"] / np.sqrt(per_cell["n_seeds"])
    per_cell["naive_se"] = per_cell["naive_sd"] / np.sqrt(per_cell["n_seeds"])

    rows = []
    for (dataset, t), grp in per_cell.groupby(["dataset", "t"]):
        n_c = len(grp)
        rows.append({
            "dataset": dataset, "t": int(t),
            "is_mse_pooled":    float(grp["is_mean"].mean()),
            "naive_mse_pooled": float(grp["naive_mean"].mean()),
            "is_se_pooled":     float(np.sqrt((grp["is_se"] ** 2).sum() / n_c ** 2)),
            "naive_se_pooled":  float(np.sqrt((grp["naive_se"] ** 2).sum() / n_c ** 2)),
            "SD_between_is":    float(grp["is_mean"].std(ddof=1)) if n_c > 1 else 0.0,
            "SD_between_naive": float(grp["naive_mean"].std(ddof=1)) if n_c > 1 else 0.0,
            "n_cells": n_c,
        })
    return pd.DataFrame(rows)


def _pool_rq6(
    df: pd.DataFrame,
    *,
    cond_var_col: str,
    rhat_col: str,
) -> pd.DataFrame:
    """rq6 variance comparison aggregation per spec §8.2.

    Per cell:
      cond_var_S = mean_seed(df[cond_var_col])  (already computed at write time)
      emp_var_R  = var(df[rhat_col], ddof=1)    across seeds (data-spec §4.3)
    Pool: equal-weight mean across cells per dataset.
    No se_emp_var_R column — sample-variance MC-SE misleading at M_c=50.
    """
    per_cell = (
        df.groupby(["dataset", "acquisition", "cell", "t"])
        .agg(
            cond_var_mean=(cond_var_col, "mean"),
            cond_var_sd=(cond_var_col, lambda v: v.std(ddof=1)),
            emp_var=(rhat_col, lambda v: v.var(ddof=1)),
            n_seeds=(rhat_col, "count"),
        )
        .reset_index()
    )
    per_cell["cond_var_se"] = per_cell["cond_var_sd"] / np.sqrt(per_cell["n_seeds"])

    rows = []
    for (dataset, acq, t), grp in per_cell.groupby(
        ["dataset", "acquisition", "t"]
    ):
        n_c = len(grp)
        rows.append({
            "dataset": dataset, "acquisition": acq, "t": int(t),
            "mean_cond_var_S":       float(grp["cond_var_mean"].mean()),
            "se_cond_var_S":         float(np.sqrt((grp["cond_var_se"] ** 2).sum() / n_c ** 2)),
            "sd_between_cond_var_S": float(grp["cond_var_mean"].std(ddof=1)) if n_c > 1 else 0.0,
            "mean_emp_var_R":        float(grp["emp_var"].mean()),
            "sd_between_emp_var_R":  float(grp["emp_var"].std(ddof=1)) if n_c > 1 else 0.0,
            "n_cells": n_c,
        })
    return pd.DataFrame(rows)


def per_dataset_pool(
    per_seed_df: pd.DataFrame,
    *,
    section: Literal["rq4", "rq5", "rq6"],
    value_spec: dict,
    expected_M_c: int = 50,
) -> pd.DataFrame:
    """Aggregate per-seed rows to (dataset, t, [acquisition], [kind]) groups.

    Pipeline:
      1. Annotate dataset/surrogate/target via parse_cell_filename(row.cell).
      2. Drop cells with M_c != expected_M_c per-dataset (WARN, not HALT).
      3. Per-section computation per spec §8.2.
      4. Output column shape per section.

    See scripts/paper_experiment/aggregation.py for section dispatch logic.
    """
    annotated = _annotate_dataset_pair(per_seed_df)
    filtered = _drop_unequal_M_c(annotated, expected_M_c=expected_M_c)
    if section == "rq4":
        return _pool_rq4(
            filtered,
            rhat_cols=value_spec["rhat_cols"],
            rn_col=value_spec["rn_col"],
        )
    if section == "rq5":
        return _pool_rq5(filtered, sq_err_cols=value_spec["sq_err_cols"])
    if section == "rq6":
        return _pool_rq6(
            filtered,
            cond_var_col=value_spec["cond_var_col"],
            rhat_col=value_spec["rhat_col"],
        )
    raise ValueError(f"unknown section: {section!r}")


def parse_cell_pair(cell_path: str) -> tuple[str, str]:
    """Extract (surrogate, target) from a `cell` column entry.

    cell column format: "results/.../cell__{method}__{ds}__{surrogate}__{target}__{loss_or_oracle}.npz"
    """
    name = cell_path.rsplit("/", 1)[-1].replace("cell__", "").replace(".npz", "")
    parts = name.split("__")
    return parts[2], parts[3]


def filter_per_seed_by_paper_pairs(per_seed_df, paper_pair_keys: set[tuple[str, str]]):
    """Filter a per_seed/per_cell DataFrame to rows whose cell column references a paper_pair."""
    keep_mask = per_seed_df["cell"].map(lambda c: parse_cell_pair(c) in paper_pair_keys)
    return per_seed_df[keep_mask]


def per_pair_filter(
    per_seed_df: pd.DataFrame,
    pair_defs: list[dict],
) -> pd.DataFrame:
    """Filter to rows whose (surrogate, target) matches any PAIR_DEFS slot.
    Annotates rows with `pair_slot`, `pair_label`, `pair_color`.

    Depends on parse_cell_filename to resolve (surrogate, target) from
    the `cell` column. Returns an empty DataFrame (with the annotation
    columns present) when pair_defs is empty or no rows match — needed
    so that `get_runtime_pairs()` returning [] (unknown SAVE_PLOT_PAIRS
    slot) does not crash showcase rendering.
    """
    annotated = _annotate_dataset_pair(per_seed_df)
    pair_lookup = {
        (p["surrogate"], p["target"]): {
            "pair_slot": p["slot"],
            "pair_label": p["label"],
            "pair_color": p["color"],
        }
        for p in pair_defs
    }
    if not pair_lookup:
        # SAVE_PLOT_PAIRS filtered to an unknown slot — return empty DF
        # with the expected annotation columns.
        empty = annotated.iloc[0:0].copy()
        for col in ("pair_slot", "pair_label", "pair_color"):
            empty[col] = pd.Series(dtype=object)
        return empty
    keep_mask = annotated[["surrogate", "target"]].apply(
        lambda row: (row["surrogate"], row["target"]) in pair_lookup, axis=1
    )
    filtered = annotated[keep_mask].copy()
    if len(filtered) == 0:
        # No cells in the data match any PAIR_DEFS pair.
        for col in ("pair_slot", "pair_label", "pair_color"):
            filtered[col] = pd.Series(dtype=object)
        return filtered
    filtered[["pair_slot", "pair_label", "pair_color"]] = filtered.apply(
        lambda row: pd.Series(pair_lookup[(row["surrogate"], row["target"])]),
        axis=1,
    )
    return filtered
