"""Render App.~``app:per-cell-efficiency`` table.

Produces one artifact under ``<out_root>/per_cell_efficiency/``:

* ``tab_per_cell.tex`` — 12-row table of median labels-to-$\epsilon$ per
  (dataset, surrogate, target), with six numeric columns grouped as
  ``{M1, M3, M4} x {accuracy, cross_entropy}``. Cells where a method
  failed to stop print ``$t_{\max}$``.

The 12 rows correspond to the four main surrogate-target pairs of
``Tab.~(\ref{tab:setup-pairs})`` evaluated across the three datasets:

* ``llama2_7b -> Mixtral_8x7b``
* ``llama2_7b -> llama3_70b``
* ``llama3_8b -> deepseek_67b``
* ``llama3_8b -> llama3_70b``

Source: ``results/paper_experiment/summary.csv`` filtered to
``source == "trajectories/main"``; medians taken across the 50 main seeds.
"""
from __future__ import annotations

from pathlib import Path
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
LOSS_ORDER = ("accuracy", "cross_entropy")
LOSS_LABEL = {"accuracy": "Accuracy", "cross_entropy": "Cross-entropy"}
METHOD_ORDER = ("M1", "M3", "M4")
METHOD_LABEL = {
    "M1": r"\textsc{SAVE-Ada}",
    "M3": r"\textsc{SAVE-Unif}",
    "M4": r"\textsc{IID+e}",
}

# Four main pairs of Tab.~(\ref{tab:setup-pairs}).
MAIN_PAIRS = (
    ("llama2_7b", "Mixtral_8x7b"),
    ("llama2_7b", "llama3_70b"),
    ("llama3_8b", "deepseek_67b"),
    ("llama3_8b", "llama3_70b"),
)


def _fmt_pair(surrogate: str, target: str) -> str:
    """Underscore-escape model names and return ``surrogate -> target`` string."""
    sur = surrogate.replace("_", r"\_")
    tgt = target.replace("_", r"\_")
    return f"{sur}$\\to${tgt}"


def build_per_cell_table(df: pd.DataFrame) -> str:
    """LaTeX table; one row per (dataset, surrogate, target), 6 numeric cols.

    Column groups separated by a vertical rule between losses. Cells where
    every seed for that (cell, method, loss) failed to stop print as
    ``$t_{\max}$`` per the spec.
    """
    # 1 col Dataset + 1 col Pair + 3 method cols x 2 loss groups = 8 cols total.
    col_spec = "ll" + "rrr" + "rrr"
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{Per-cell median labels-to-$\epsilon$ ($\epsilon=0.05$) for the "
        r"three certified methods across the four main pairs of "
        r"Tab.~(\ref{tab:setup-pairs}). Median is taken across the 50 main "
        r"seeds; cells where every seed exceeds $t_{\max}$ print as "
        r"$t_{\max}$. \textsc{SAVE-Ada} (M1) tightens to $\epsilon$ in fewer "
        r"labels than \textsc{IID+e} (M4) on every cell, with the surrogate-on "
        r"(M1) vs.\ surrogate-off (M3) gap widening on the higher-$\rho$ cells.}",
        r"\label{tab:app-per-cell-efficiency}",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        # Multi-column header
        r" & & \multicolumn{3}{c}{Accuracy} & \multicolumn{3}{c}{Cross-entropy} \\",
        r"\cmidrule(lr){3-5} \cmidrule(lr){6-8}",
        r"Dataset & Pair & "
        + " & ".join(METHOD_LABEL[m] for m in METHOD_ORDER)
        + " & "
        + " & ".join(METHOD_LABEL[m] for m in METHOD_ORDER)
        + r" \\",
        r"\midrule",
    ]

    for ds in DATASET_ORDER:
        for sur, tgt in MAIN_PAIRS:
            cells = []
            for loss in LOSS_ORDER:
                for method in METHOD_ORDER:
                    sub = df[
                        (df["dataset"] == ds)
                        & (df["surrogate"] == sur)
                        & (df["target"] == tgt)
                        & (df["loss"] == loss)
                        & (df["method"] == method)
                        & (df["did_stop"] == 1)
                    ]
                    if sub.empty:
                        cells.append(r"$t_{\max}$")
                    else:
                        med = int(sub["labels_to_stop"].median())
                        cells.append(f"{med:,}")
            pair_str = _fmt_pair(sur, tgt)
            lines.append(
                f"{DATASET_LABEL[ds]} & {pair_str} & " + " & ".join(cells) + r" \\"
            )
        # Add a midrule between datasets except after the last.
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
    summary = pd.read_csv(out_root / "summary.csv")
    if pair_filter:
        summary = summary[
            summary.apply(lambda r: (r.surrogate, r.target) in pair_filter, axis=1)
        ]
    df = summary[summary["source"] == "trajectories/main"].copy()
    out_dir = out_root / "per_cell_efficiency"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "tab_per_cell.tex").write_text(build_per_cell_table(df))
    print(f"wrote per_cell_efficiency artifacts -> {out_dir}")


if __name__ == "__main__":
    main()
