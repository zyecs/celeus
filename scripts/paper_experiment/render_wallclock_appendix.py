"""Render wall-clock appendix (v0504-compatible).

Produces ``<out_root>/wallclock_appendix/tab_per_round.tex``: a 2-method
comparison (SAVE-ADA / IID+EValue) x N wallclock cells, listing
seconds-per-round.

* SAVE-ADA / IID+EValue per-round = ``elapsed_seconds / labels_to_stop``
  averaged across seeds (rows pulled from summary.csv; M1 / M4; the
  N wallclock cells x cfg.wallclock['seeds']).

Wallclock data is scope-agnostic (was set in advance for cell-cost
characterization, independent of paper_pair selection). When ``--out-root``
points at a snapshot dir like ``results/paper_experiments_v0504``, we still
read the timing rows from the canonical ``paper_experiment/summary.csv``
unless ``--summary-root`` overrides.

Cer-Eval (M5) is intentionally omitted (paper scope).
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

# Make src/ and the repo root importable when running as a script.
_REPO = Path(__file__).resolve().parents[2]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from save.paper_experiment.config import default_config_path, load_config


METHOD_LABEL = {
    "M1": "SAVE-ADA",
    "M4": "IID+EValue",
}


def _save_per_round(summary: pd.DataFrame, *, method: str, cell: dict,
                    seeds: list[int], loss: str) -> float:
    """Mean per-round wall time for SAVE-ADA / IID+EValue (one cell, k seeds).

    Per-seed per-round = ``elapsed_seconds / labels_to_stop``. Seeds
    that didn't stop (labels_to_stop <= 0) are dropped before averaging.
    Returns NaN if no seed produced a valid quotient.
    """
    sub = summary[
        (summary.source == "trajectories/main")
        & (summary.method == method)
        & (summary.loss == loss)
        & (summary.dataset == cell["dataset"])
        & (summary.surrogate == cell["surrogate"])
        & (summary.target == cell["target"])
        & summary.seed.isin(seeds)
    ]
    if sub.empty:
        return float("nan")
    valid = sub[(sub.labels_to_stop > 0) & sub.elapsed_seconds.notna()]
    if valid.empty:
        return float("nan")
    per_round = valid.elapsed_seconds.to_numpy() / valid.labels_to_stop.to_numpy()
    return float(np.mean(per_round))


def build_per_round_table(summary: pd.DataFrame, cfg) -> str:
    """2-method (SAVE / IID) x N-cell per-round seconds table."""
    cells = cfg.wallclock["cells"]
    seeds = list(cfg.wallclock["seeds"])
    loss = cfg.wallclock.get("loss", "accuracy")

    def _cell_header(c: dict) -> str:
        sur = c["surrogate"].replace("_", r"\_")
        tgt = c["target"].replace("_", r"\_")
        return f"{c['dataset']}/{sur}$\\to${tgt}"

    lines = [
        r"\begin{tabular}{l" + "r" * len(cells) + "}",
        r"\toprule",
        "Method & " + " & ".join(_cell_header(c) for c in cells) + r" \\",
        r"\midrule",
    ]
    for method in ("M1", "M4"):
        cells_str: list[str] = []
        for cell in cells:
            v = _save_per_round(
                summary, method=method, cell=cell, seeds=seeds, loss=loss,
            )
            cells_str.append("--" if not np.isfinite(v) else f"{v:.4f}")
        lines.append(f"{METHOD_LABEL[method]} & " + " & ".join(cells_str) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-root", type=Path, default=None,
                    help="Override results root (where wallclock_appendix/ is written).")
    ap.add_argument("--summary-root", type=Path, default=None,
                    help="Override summary.csv source root. Defaults to "
                         "results/paper_experiment (where wallclock-stage rows live).")
    args = ap.parse_args(argv)

    cfg = load_config(default_config_path())
    out_root = args.out_root if args.out_root else Path(cfg.paths["out_root"])
    summary_root = args.summary_root if args.summary_root else (_REPO / "results" / "paper_experiment")
    summary = pd.read_csv(summary_root / "summary.csv")

    out_dir = out_root / "wallclock_appendix"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "tab_per_round.tex").write_text(
        build_per_round_table(summary, cfg)
    )
    print(f"wrote wallclock_appendix artifacts -> {out_dir} (cells={len(cfg.wallclock['cells'])}, methods=M1+M4)")


if __name__ == "__main__":
    main()
