"""Render M1 vs M3 (surrogate on/off) appendix for §6.5 item #2.

Produces two artifacts under ``<out_root>/surrogate_onoff/``:

* ``tab_delta.tex`` — per (dataset, loss) median labels-to-epsilon for
  M1 (Celeus, surrogate on) vs M3 (Celeus-no-surr, surrogate off) plus
  the relative delta in percent. Lands behind ``app:surrogate-onoff``.
* ``fig_width_per_dataset.pdf`` — six-panel width-vs-labels figure
  (3 datasets x 2 losses), M1 in blue and M3 in green per panel.

Reuses ``_load_method_curves`` from ``render_ce_appendix`` to keep the
trajectory aggregation identical to the CE appendix.
"""
from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import matplotlib.pyplot as plt

# Make src/ and the repo root importable when running as a script
# (parents[2] is the repo root containing both ``src/`` and ``scripts/``).
_REPO = Path(__file__).resolve().parents[2]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from save.paper_experiment.config import default_config_path, load_config
from scripts.paper_experiment.plot_style import apply_rc_helvetica
from scripts.paper_experiment.render_ce_appendix import _load_method_curves


DATASET_LABEL = {"sst2": "SST-2", "mmlu": "MMLU", "agnews": "AG News"}


def build_delta_table(summary: pd.DataFrame) -> str:
    """Rows = (dataset, loss); cols = (M1 median labels, M3 median, Δ%)."""
    df = summary[summary.method.isin(("M1", "M3")) & (summary.did_stop == 1)]
    lines = [
        r"\begin{tabular}{llrrr}",
        r"\toprule",
        r"Dataset & Loss & M1 (med) & M3 (med) & $\Delta$ (\%) \\",
        r"\midrule",
    ]
    for ds in ("sst2", "mmlu", "agnews"):
        for loss in ("accuracy", "cross_entropy"):
            sub = df[(df.dataset == ds) & (df.loss == loss)]
            m1_slc = sub[sub.method == "M1"]
            m3_slc = sub[sub.method == "M3"]
            if m1_slc.empty or m3_slc.empty:
                continue
            m1 = int(m1_slc.labels_to_stop.median())
            m3 = int(m3_slc.labels_to_stop.median())
            delta = 100.0 * (m3 - m1) / m1 if m1 else float("nan")
            loss_str = loss.replace("_", r"\_")
            lines.append(
                f"{DATASET_LABEL[ds]} & {loss_str} & {m1:,} & {m3:,} & {delta:+.1f} \\\\"
            )
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def render_width_figure(out_root: Path, cfg, out_path: Path) -> None:
    """Six-panel M1-vs-M3 width-vs-labels (rows = losses, cols = datasets)."""
    apply_rc_helvetica()
    fig, axes = plt.subplots(2, 3, figsize=(11, 5.5), sharey="row")
    for row_i, loss in enumerate(("accuracy", "cross_entropy")):
        for ax, ds in zip(axes[row_i], ("sst2", "mmlu", "agnews")):
            for m, color in [("M1", "C0"), ("M3", "C2")]:
                curve = _load_method_curves(out_root, m, loss, ds, cfg)
                if curve is None:
                    continue
                labels, mean_w, lo_w, hi_w = curve
                ax.plot(labels, mean_w, color=color, label=m)
                ax.fill_between(labels, lo_w, hi_w, color=color, alpha=0.2)
            if row_i == 0:
                ax.set_title(DATASET_LABEL[ds])
            if row_i == 1:
                ax.set_xlabel("labels acquired")
            if ds == "sst2":
                ax.set_ylabel(f"width ({loss})")
        axes[row_i, 0].legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


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
    out_dir = out_root / "surrogate_onoff"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "tab_delta.tex").write_text(build_delta_table(summary))
    render_width_figure(out_root, cfg, out_dir / "fig_width_per_dataset.pdf")
    print(f"wrote surrogate_onoff artifacts -> {out_dir}")


if __name__ == "__main__":
    main()
