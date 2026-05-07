"""Render §6.4 MAE inset + App. C ``app:mae-axis`` artifacts.

Produces two artifacts under ``<out_root>/mae_axis_figs/``:

* ``fig_mae_inset.pdf`` — two-panel scatter of width@stop versus pool MAE
  on a log10 x-axis, colour-coded by dataset and split into accuracy /
  cross-entropy panels. Drops into the §6.4 inset slot.
* ``tab_mae_full.tex`` — full per-(dataset, pair, loss) MAE table with
  width@stop and median V_t. 60 rows = 3 datasets x 10 pairs x 2 losses.
  Lands behind ``app:mae-axis``.

Input is ``<out_root>/mae_axis.csv`` written by ``compute_mae_axis.py``.
"""
from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import matplotlib.pyplot as plt

# Make src/ and the repo root importable when running as a script.
_REPO = Path(__file__).resolve().parents[2]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from save.paper_experiment.config import default_config_path, load_config
from scripts.paper_experiment.plot_style import apply_rc_helvetica


DATASET_LABEL = {"sst2": "SST-2", "mmlu": "MMLU", "agnews": "AG News"}
DATASET_COLOR = {"sst2": "C0", "mmlu": "C1", "agnews": "C2"}
LOSS_LABEL = {"accuracy": "Accuracy", "cross_entropy": "Cross-entropy"}


def build_full_table(df: pd.DataFrame) -> str:
    """LaTeX table of every (dataset, surrogate, target, loss) row.

    Columns: Dataset | Pair (surrogate->target) | Loss | MAE | width@stop | V_t.
    NaN ``V_t_med`` (CE rows do not have an RQ6 V_t entry) prints as
    ``--``. Sorted by (dataset, loss, MAE) so the reader can scan tertile
    structure within each (dataset, loss) slice.
    """
    lines = [
        r"\begin{tabular}{llllrrr}",
        r"\toprule",
        r"Dataset & Pair & Loss & MAE & width@stop & $V_t$ \\",
        r"\midrule",
    ]
    sorted_df = df.sort_values(["dataset", "loss", "MAE"])
    for _, r in sorted_df.iterrows():
        # Escape underscores in model names (e.g. llama2_7b → llama2\_7b)
        # and the loss label (cross_entropy → cross\_entropy) for text mode.
        sur = str(r.surrogate).replace("_", r"\_")
        tgt = str(r.target).replace("_", r"\_")
        loss_str = str(r.loss).replace("_", r"\_")
        pair = f"{sur}$\\to${tgt}"
        v_t_str = "--" if pd.isna(r.V_t_med) else f"{float(r.V_t_med):.4f}"
        ds_label = DATASET_LABEL.get(r.dataset, r.dataset)
        lines.append(
            f"{ds_label} & {pair} & {loss_str} & "
            f"{float(r.MAE):.3f} & {float(r.width_at_stop_med):.4f} & {v_t_str} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def render_inset(df: pd.DataFrame, out_path: Path) -> None:
    """Two-panel scatter: log10(MAE) vs width@stop, color by dataset, panel by loss."""
    apply_rc_helvetica()
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))
    for ax, loss in zip(axes, ("accuracy", "cross_entropy")):
        sub = df[df.loss == loss]
        for ds, ds_label in DATASET_LABEL.items():
            slc = sub[sub.dataset == ds]
            if slc.empty:
                continue
            ax.scatter(
                slc.MAE,
                slc.width_at_stop_med,
                s=30,
                color=DATASET_COLOR[ds],
                label=ds_label,
                alpha=0.85,
                edgecolors="none",
            )
        ax.set_xscale("log")
        ax.set_xlabel(r"MAE $\mathbb{E}|\ell-\hat\ell|$")
        ax.set_ylabel(r"width@$\tau_\epsilon$")
        ax.set_title(LOSS_LABEL[loss])
    axes[0].legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", type=Path, default=None,
                    help="Override results root. Default: cfg.paths['out_root'].")
    ap.add_argument("--paper-pairs", action="store_true",
                    help="Filter to cfg.paper_pair_keys (v0502 scope); "
                         "compute_mae_axis.py handles the filter upstream.")
    args = ap.parse_args(argv)
    cfg = load_config(default_config_path())
    out_root = args.out_root if args.out_root else Path(cfg.paths["out_root"])
    df = pd.read_csv(out_root / "mae_axis.csv")
    out_dir = out_root / "mae_axis_figs"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "tab_mae_full.tex").write_text(build_full_table(df))
    render_inset(df, out_dir / "fig_mae_inset.pdf")
    print(f"wrote mae_axis_figs artifacts -> {out_dir}")


if __name__ == "__main__":
    main()
