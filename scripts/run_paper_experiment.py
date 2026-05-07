#!/usr/bin/env python3
"""Paper-experiment orchestration CLI.

Usage:
  python scripts/run_paper_experiment.py --stage <STAGE> [--index <N>] [--config <PATH>]

Where <STAGE> is one of:
  compute-rn       — Compute R_N.csv across all (dataset,target,loss) combos.
  disk-audit       — Verify disk/inode budget + input readability (spec §9 Stage 0.5).
  cereval-fresh    — Stage 0: M5 fresh sweep (60 cells × 50 seeds).
  smoke            — Stage 1: smoke test (1 cell × 3 seeds × 4 methods).
  ce-sweep         — Stage 2: CE surrogate mini-sweep (90 cells).
  main-accuracy    — Stage 3: M1–M4 × 3 datasets × 10 pairs × accuracy (120 cells).
  main-ce          — Stage 4: same, CE loss (120 cells). Reads ce_sweep_winner.json.
  beta-sweep       — Stage 5: β-sensitivity regen (72 cells).
  oracle-accuracy  — Accuracy-only oracle acquisition upper-bound (30 cells, 300 chunked assignments).
  acquisition-sweep — §6.5 item #3: strategy sweep on 6 cells × 15 seeds. Requires --loss.
  hparam-sweep      — §6.5 item #4: hyperparam OAT sweep on 4 cells × 7 configs × 10 seeds (accuracy-only).
  wallclock         — §6.5 item #5: Cer-Eval per-round timing on 4 cells × 5 seeds (accuracy-only).
  merge-cells      — Consolidate subcells into bundled .npz (spec §4 bundling).
  merge-oracle-accuracy — Consolidate oracle subcells into bundled .npz.
  analyze          — Stage 6: build summary.csv + figures + tables.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


_STAGES = (
    "compute-rn", "disk-audit", "cereval-fresh", "smoke",
    "ce-sweep", "main-accuracy", "main-ce", "beta-sweep",
    "oracle-accuracy", "acquisition-sweep", "hparam-sweep", "wallclock",
    "merge-cells", "merge-oracle-accuracy", "analyze",
    "v0502-acq-sweep", "v0502-rebuild",
)


def _add_src_to_path() -> None:
    repo = Path(__file__).resolve().parents[1]
    src = repo / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def _apply_out_root_override(cfg_paths: dict, override: str | None) -> None:
    """Mutate cfg_paths['out_root'] if override is not None.

    Used by smoke tests to redirect writes to an isolated tmpdir.
    """
    if override is not None:
        cfg_paths["out_root"] = override


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Paper-experiment orchestration CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Stages: " + ", ".join(_STAGES),
    )
    p.add_argument("--stage", required=True, choices=_STAGES,
                   help="Which pipeline stage to run.")
    p.add_argument("--index", type=int, default=None,
                   help="SLURM array task index (0-based). Interpretation stage-specific.")
    p.add_argument("--config", type=Path,
                   default=Path("configs/paper_experiment.yaml"),
                   help="Path to paper_experiment.yaml.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the cell that would run without executing it.")
    p.add_argument(
        "--out-root", default=None, dest="out_root",
        help="Override cfg.paths.out_root (smoke-test convenience).",
    )
    p.add_argument(
        "--loss", default=None, choices=["accuracy", "cross_entropy"],
        help="Required when --stage acquisition-sweep (selects sub-array).",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    _add_src_to_path()
    args = parse_args(argv)

    from save.paper_experiment.config import default_config_path, load_config
    from save.paper_experiment import stages
    # If the user didn't pass --config, honor SAVE_PE_CONFIG (v0504 build path).
    if args.config == Path("configs/paper_experiment.yaml"):
        args.config = default_config_path()

    cfg = load_config(args.config)
    _apply_out_root_override(cfg.paths, args.out_root)

    if args.stage == "acquisition-sweep":
        if args.loss is None:
            raise SystemExit("--loss is required for --stage acquisition-sweep")
        return stages.run_acquisition_sweep(
            cfg=cfg, loss=args.loss, index=args.index, dry_run=args.dry_run,
        )

    if args.stage == "v0502-acq-sweep":
        if not args.loss:
            parser.error("--stage v0502-acq-sweep requires --loss accuracy|cross_entropy")
        return stages.run_acquisition_sweep(
            cfg=cfg, loss=args.loss, scope="v0502",
            index=args.index, dry_run=args.dry_run,
        )

    if args.stage == "v0502-rebuild":
        import subprocess
        import sys as _sys
        return subprocess.call(
            [_sys.executable, "scripts/paper_experiment/build_v0502_outputs.py"]
        )

    dispatch = {
        "compute-rn":    stages.run_compute_rn,
        "disk-audit":    stages.run_disk_audit,
        "cereval-fresh": stages.run_cereval_fresh,
        "smoke":         stages.run_smoke,
        "ce-sweep":      stages.run_ce_sweep,
        "main-accuracy": stages.run_main_accuracy,
        "main-ce":       stages.run_main_ce,
        "beta-sweep":    stages.run_beta_sweep,
        "oracle-accuracy": stages.run_oracle_accuracy,
        "hparam-sweep":  stages.run_hparam_sweep,
        "wallclock":     stages.run_wallclock,
        "merge-cells":   stages.run_merge_cells,
        "merge-oracle-accuracy": stages.run_merge_oracle_accuracy,
        "analyze":       stages.run_analyze,
    }
    return dispatch[args.stage](cfg=cfg, index=args.index, dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
