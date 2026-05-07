"""Render App.~``app:per-cell-mechanism`` table.

Produces one artifact under ``<out_root>/per_cell_mechanism/``:

* ``tab_per_cell.tex`` — 12-row table joining the three RQ3 diagnostic
  CSVs at ``t == max(t)`` per cell. Columns: Dataset, Pair, $|\text{bias}|$
  of $\hat R_t$ (RQ4), MSE of $\hat S_t$ (RQ5), and
  $\mathrm{Var}(\hat S_t \mid \mathcal{F}_{t-1})$ (RQ6).

The 12 rows correspond to the four main surrogate-target pairs of
``Tab.~(\ref{tab:setup-pairs})`` evaluated across the three datasets.

Sources:

* ``rq4-unbiasedness/accuracy/per_cell_curves.csv`` — filter ``acquisition
  == "ada"``, ``kind == "lure"``; take row at ``t == max(t)`` per cell.
  ``kind == "lure"`` is the LURE-weighted estimator $\hat R_t$ of
  ``\eqref{eq:lure_estimator}``.
* ``rq5-signal-mse/accuracy/per_cell_curves.csv`` — take ``is_mse``
  (IS-corrected signal MSE) at ``t == max(t)`` per cell.
* ``rq6-variance/accuracy/per_cell_curves.csv`` — filter ``acquisition
  == "ada"``; take ``cond_var_S_mean`` at ``t == max(t)`` per cell.

Cells are inner-joined on the ``cell`` column (full trajectory file path).
Cell paths are parsed for (dataset, surrogate, target).
"""
from __future__ import annotations

from pathlib import Path
import re
import sys

import pandas as pd

# Make src/ and the repo root importable when running as a script.
_REPO = Path(__file__).resolve().parents[2]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from save.paper_experiment.config import default_config_path, load_config


DATASET_LABEL = {"sst2": "SST-2", "mmlu": "MMLU", "agnews": "AG News"}
DATASET_ORDER = ("sst2", "mmlu", "agnews")
MAIN_PAIRS = (
    ("llama2_7b", "Mixtral_8x7b"),
    ("llama2_7b", "llama3_70b"),
    ("llama3_8b", "deepseek_67b"),
    ("llama3_8b", "llama3_70b"),
)

# cell filename pattern: cell__M1__sst2__llama2_7b__llama3_70b__accuracy.npz
# Surrogate/target name characters include alnum + underscore + literal "x"
# (Mixtral_8x7b). We split on ``__`` to recover the four ID components.
_CELL_RE = re.compile(
    r"cell__(?P<method>[A-Za-z0-9]+)__(?P<dataset>[A-Za-z0-9]+)__"
    r"(?P<surrogate>[A-Za-z0-9_]+?)__(?P<target>[A-Za-z0-9_]+?)__"
    r"(?P<loss>[A-Za-z0-9_]+)\.npz$"
)


def _parse_cell(cell_path: str) -> dict[str, str]:
    """Parse the canonical ``cell__<method>__...__<loss>.npz`` filename."""
    base = Path(cell_path).name
    # The filename has exactly 6 ``__``-separated tokens after stripping
    # ``cell__`` prefix and ``.npz`` suffix; split is safer than regex for
    # the dataset/surrogate/target components which contain underscores.
    stem = base.removesuffix(".npz")
    parts = stem.split("__")
    # cell, method, dataset, surrogate, target, loss
    if len(parts) != 6 or parts[0] != "cell":
        raise ValueError(f"Unexpected cell filename: {base}")
    _, method, dataset, surrogate, target, loss = parts
    return {
        "method": method,
        "dataset": dataset,
        "surrogate": surrogate,
        "target": target,
        "loss": loss,
    }


def _final_t_per_cell(df: pd.DataFrame, value_cols: list[str]) -> pd.DataFrame:
    """Return one row per cell at the maximum ``t`` value, keeping cell + value_cols."""
    idx = df.groupby("cell")["t"].idxmax()
    keep = ["cell", "t"] + value_cols
    return df.loc[idx, keep].reset_index(drop=True)


def _fmt_pair(surrogate: str, target: str) -> str:
    sur = surrogate.replace("_", r"\_")
    tgt = target.replace("_", r"\_")
    return f"{sur}$\\to${tgt}"


def build_table(merged: pd.DataFrame) -> str:
    """LaTeX table with 12 rows in (dataset, MAIN_PAIRS) order."""
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{Per-cell RQ3 diagnostics evaluated at the final available "
        r"$t$ per cell (accuracy loss): bias of the LURE-weighted estimator "
        r"$\hat R_t$ \eqref{eq:lure_estimator} (RQ4, ``ada'' acquisition; "
        r"smaller is closer to unbiased), MSE of the inferential signal "
        r"$\hat S_t$ \eqref{eq:signal_estimator} (RQ5, IS-corrected; smaller "
        r"is tighter), and the conditional variance "
        r"$\mathrm{Var}(\hat S_t \mid \mathcal{F}_{t-1})$ (RQ6, ``ada'' "
        r"acquisition). Each metric controls a different term in the width "
        r"bound of Thm.~(\ref{thm:conf_sequen}).}",
        r"\label{tab:app-per-cell-mechanism}",
        r"\begin{tabular}{llrrr}",
        r"\toprule",
        r"Dataset & Pair & $|\mathrm{bias}|$ & $\mathrm{MSE}(\hat S_t)$ & "
        r"$\mathrm{Var}(\hat S_t \mid \mathcal{F}_{t-1})$ \\",
        r"\midrule",
    ]
    for ds in DATASET_ORDER:
        for sur, tgt in MAIN_PAIRS:
            row = merged[
                (merged["dataset"] == ds)
                & (merged["surrogate"] == sur)
                & (merged["target"] == tgt)
            ]
            if row.empty:
                # All metrics missing for this pair.
                lines.append(
                    f"{DATASET_LABEL[ds]} & {_fmt_pair(sur, tgt)} & -- & -- & -- \\\\"
                )
                continue
            r = row.iloc[0]
            bias = abs(float(r["bias"]))
            mse = float(r["is_mse"])
            cvar = float(r["cond_var_S_mean"])
            lines.append(
                f"{DATASET_LABEL[ds]} & {_fmt_pair(sur, tgt)} & "
                f"{bias:.3g} & {mse:.3g} & {cvar:.3g} \\\\"
            )
        if ds != DATASET_ORDER[-1]:
            lines.append(r"\midrule")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", type=Path, default=None,
                    help="Override results root. Default: cfg.paths['out_root'].")
    ap.add_argument("--paper-pairs", action="store_true",
                    help="Filter to cfg.paper_pair_keys (v0502 scope).")
    args = ap.parse_args(argv)
    cfg = load_config(default_config_path())
    out_root = args.out_root if args.out_root else Path(cfg.paths["out_root"])
    pair_filter = cfg.paper_pair_keys if args.paper_pairs else None

    def _keep(cell_path: str) -> bool:
        name = cell_path.rsplit("/", 1)[-1].replace("cell__", "").replace(".npz", "")
        parts = name.split("__")
        return (parts[2], parts[3]) in pair_filter  # type: ignore[operator]

    # RQ4: bias of \hat R_t (ada/lure), final t per cell.
    rq4 = pd.read_csv(out_root / "rq4-unbiasedness/accuracy/per_cell_curves.csv")
    if pair_filter:
        rq4 = rq4[rq4["cell"].map(_keep)]
    rq4 = rq4[(rq4["acquisition"] == "ada") & (rq4["kind"] == "lure")]
    rq4 = _final_t_per_cell(rq4, ["bias", "MC_SE"]).rename(columns={"t": "t_rq4"})

    # RQ5: IS-corrected signal MSE, final t per cell.
    rq5 = pd.read_csv(out_root / "rq5-signal-mse/accuracy/per_cell_curves.csv")
    if pair_filter:
        rq5 = rq5[rq5["cell"].map(_keep)]
    rq5 = _final_t_per_cell(rq5, ["is_mse"]).rename(columns={"t": "t_rq5"})

    # RQ6: cond_var_S_mean (ada), final t per cell.
    rq6 = pd.read_csv(out_root / "rq6-variance/accuracy/per_cell_curves.csv")
    if pair_filter:
        rq6 = rq6[rq6["cell"].map(_keep)]
    rq6 = rq6[rq6["acquisition"] == "ada"]
    rq6 = _final_t_per_cell(rq6, ["cond_var_S_mean"]).rename(columns={"t": "t_rq6"})

    # Inner-join on cell.
    merged = rq4.merge(rq5, on="cell", how="inner").merge(rq6, on="cell", how="inner")

    # Parse cell -> (method, dataset, surrogate, target, loss).
    parsed = merged["cell"].apply(_parse_cell).apply(pd.Series)
    merged = pd.concat([parsed, merged.drop(columns=["cell"])], axis=1)

    out_dir = out_root / "per_cell_mechanism"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "tab_per_cell.tex").write_text(build_table(merged))
    print(f"wrote per_cell_mechanism artifacts -> {out_dir}")


if __name__ == "__main__":
    main()
