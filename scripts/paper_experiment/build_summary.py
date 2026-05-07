"""Build results/paper_experiment/summary.csv (all trajectories, one row/seed)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "scripts"))

import paper_experiment_io as pio


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build paper-experiment summary.csv")
    p.add_argument(
        "--include-oracle-accuracy",
        action="store_true",
        help="Include trajectories/oracle_accuracy in the summary build.",
    )
    p.add_argument(
        "--include-hparam-sweep",
        action="store_true",
        help="Include trajectories/hparam_sweep in the summary build.",
    )
    p.add_argument(
        "--include-wallclock",
        action="store_true",
        help="Include trajectories/wallclock in the summary build.",
    )
    p.add_argument(
        "--out-root",
        type=Path,
        default=None,
        help="Override results root. Default: <repo>/results/paper_experiment.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    base = args.out_root if args.out_root else (_REPO / "results" / "paper_experiment")
    frames = []
    # v2: load_summary now dispatches parsers per subdir so ce_sweep and
    # beta_sweep trajectories are no longer silently dropped.
    subdirs = [
        "trajectories/main",
        "trajectories/ce_sweep",
        "trajectories/beta_sweep",
        "trajectories/acquisition_sweep",  # v0502: include acq_sweep cells
    ]
    if args.include_oracle_accuracy:
        subdirs.append("trajectories/oracle_accuracy")
    if args.include_hparam_sweep:
        subdirs.append("trajectories/hparam_sweep")
    if args.include_wallclock:
        subdirs.append("trajectories/wallclock")
    for sub in subdirs:
        if not (base / sub).exists():
            continue
        df = pio.load_summary(out_base=base, subdir=sub)
        if df.empty:
            continue
        df = df.assign(source=sub)
        frames.append(df)
    if not frames:
        print("no trajectories yet")
        return 1
    import pandas as pd

    combined = pd.concat(frames, ignore_index=True)
    out_path = base / "summary.csv"
    combined.to_csv(out_path, index=False)
    print(f"wrote {len(combined)} rows to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
