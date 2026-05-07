"""Side-by-side comparison of pooled metrics between v0502 and v0504.

Reads per_dataset_curves.csv from each scope for rq4/rq5/rq6 and prints a
table at fixed anchors t in {1000, 2000, 3000}. Flags any |relative delta|
> 5% as a regression candidate, so v0504 promotion has a quantitative gate
rather than a narrative one.

Usage::
    python scripts/paper_experiment/diff_v0502_v0504.py [--anchors 1000,2000,3000]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_REPO = Path(__file__).resolve().parents[2]
V0502 = _REPO / "results" / "paper_experiments_v0502"
V0504 = _REPO / "results" / "paper_experiments_v0504"

# (section_dir, csv_relative_to_section, key column(s) of interest, label, group_cols)
# group_cols defines what each row is keyed on within the CSV.
_SECTIONS = [
    {
        "label": "rq4 bias (LURE)",
        "v0502_csv": "rq4-unbiasedness/accuracy/per_dataset_curves.csv",
        "v0504_csv": "rq4-unbiasedness/accuracy/per_dataset_curves.csv",
        "key": "bias_pooled",
        "filters": [("acquisition", "ada"), ("kind", "lure")],
        "anchor_col": "t",
    },
    {
        "label": "rq4 bias (Unweighted)",
        "v0502_csv": "rq4-unbiasedness/accuracy/per_dataset_curves.csv",
        "v0504_csv": "rq4-unbiasedness/accuracy/per_dataset_curves.csv",
        "key": "bias_pooled",
        "filters": [("acquisition", "ada"), ("kind", "unweighted")],
        "anchor_col": "t",
    },
    {
        "label": "rq5 IS-MSE",
        "v0502_csv": "rq5-signal-mse/accuracy/per_dataset_curves.csv",
        "v0504_csv": "rq5-signal-mse/accuracy/per_dataset_curves.csv",
        "key": "is_mse_pooled",
        "filters": [],
        "anchor_col": "t",
    },
    {
        "label": "rq5 naive-MSE",
        "v0502_csv": "rq5-signal-mse/accuracy/per_dataset_curves.csv",
        "v0504_csv": "rq5-signal-mse/accuracy/per_dataset_curves.csv",
        "key": "naive_mse_pooled",
        "filters": [],
        "anchor_col": "t",
    },
    {
        "label": "rq6 cond_var_S (ADA)",
        "v0502_csv": "rq6-variance/accuracy/per_dataset_curves.csv",
        "v0504_csv": "rq6-variance/accuracy/per_dataset_curves.csv",
        "key": "mean_cond_var_S",
        "filters": [("acquisition", "ada")],
        "anchor_col": "t",
    },
    {
        "label": "rq6 cond_var_S (Uniform)",
        "v0502_csv": "rq6-variance/accuracy/per_dataset_curves.csv",
        "v0504_csv": "rq6-variance/accuracy/per_dataset_curves.csv",
        "key": "mean_cond_var_S",
        "filters": [("acquisition", "uniform")],
        "anchor_col": "t",
    },
    {
        "label": "rq6 emp_var_R (ADA)",
        "v0502_csv": "rq6-variance/accuracy/per_dataset_curves.csv",
        "v0504_csv": "rq6-variance/accuracy/per_dataset_curves.csv",
        "key": "mean_emp_var_R",
        "filters": [("acquisition", "ada")],
        "anchor_col": "t",
    },
    {
        "label": "rq6 emp_var_R (Uniform)",
        "v0502_csv": "rq6-variance/accuracy/per_dataset_curves.csv",
        "v0504_csv": "rq6-variance/accuracy/per_dataset_curves.csv",
        "key": "mean_emp_var_R",
        "filters": [("acquisition", "uniform")],
        "anchor_col": "t",
    },
]

DATASETS = ("sst2", "mmlu", "agnews")


def _load_filtered(csv_path: Path, filters):
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    for col, val in filters:
        df = df[df[col] == val]
    return df


def _row_at(df: pd.DataFrame, dataset: str, t: int, key: str) -> float | None:
    if df is None:
        return None
    sub = df[(df["dataset"] == dataset) & (df["t"] == t)]
    if sub.empty:
        return None
    return float(sub[key].iloc[0])


def _flag(rel_delta: float, *, threshold: float = 0.05) -> str:
    if rel_delta is None:
        return "    "
    if abs(rel_delta) > threshold:
        return " ⚠ "
    return "    "


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--anchors", default="1000,2000,3000",
                    help="comma-separated t anchors (default 1000,2000,3000)")
    ap.add_argument("--threshold", type=float, default=0.05,
                    help="|relative delta| flagging threshold (default 0.05 = 5%%)")
    args = ap.parse_args(argv)
    anchors = [int(s) for s in args.anchors.split(",") if s.strip()]

    if not V0504.is_dir():
        sys.exit(f"v0504 results dir not found: {V0504}\n"
                 f"Run build_v0504_outputs.py --phase B first.")

    print(f"\n=== v0502 ↔ v0504 pooled comparison (threshold ±{args.threshold:.0%}) ===\n")
    flagged_rows: list[str] = []

    for sec in _SECTIONS:
        df02 = _load_filtered(V0502 / sec["v0502_csv"], sec["filters"])
        df04 = _load_filtered(V0504 / sec["v0504_csv"], sec["filters"])
        print(f"## {sec['label']}  ({sec['key']})")
        header = f"  {'dataset':<7} {'t':>5}  {'v0502':>12}  {'v0504':>12}  {'Δ_abs':>11}  {'Δ_rel':>8}"
        print(header)
        print("  " + "-" * (len(header) - 2))
        for ds in DATASETS:
            for t in anchors:
                v02 = _row_at(df02, ds, t, sec["key"])
                v04 = _row_at(df04, ds, t, sec["key"])
                if v02 is None or v04 is None:
                    cell02 = f"{v02:>12}" if v02 is None else f"{v02:>12.4g}"
                    cell04 = f"{v04:>12}" if v04 is None else f"{v04:>12.4g}"
                    print(f"  {ds:<7} {t:>5}  {cell02}  {cell04}  {'?':>11}  {'?':>8}")
                    continue
                d_abs = v04 - v02
                d_rel = d_abs / v02 if v02 != 0 else float("inf")
                flag = _flag(d_rel, threshold=args.threshold)
                row = (f"  {ds:<7} {t:>5}  {v02:>12.4g}  {v04:>12.4g}  "
                       f"{d_abs:>+11.3g}  {d_rel:>+7.1%} {flag}")
                print(row)
                if abs(d_rel) > args.threshold:
                    flagged_rows.append(f"{sec['label']}  {ds} t={t}: "
                                        f"{v02:.4g} → {v04:.4g} ({d_rel:+.1%})")
        print()

    if flagged_rows:
        print(f"=== Flagged rows (|Δ_rel| > {args.threshold:.0%}) ===")
        for r in flagged_rows:
            print("  " + r)
    else:
        print("=== No flagged rows. v0504 within tolerance of v0502 across all checked metrics. ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
