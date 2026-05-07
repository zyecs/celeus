#!/usr/bin/env python
"""Re-render rq4-rq6 figures from existing per_seed_curves.csv.

Use after editing plot_style.py or fixing layouts — bypasses the SLURM
data-layer job. Wallclock target: <30s for all 8 PDFs.

Usage from project root:
  PYTHONPATH=. python scripts/paper_experiment/regen_plots_only.py
  PYTHONPATH=. python scripts/paper_experiment/regen_plots_only.py --section rq4
  PYTHONPATH=. python scripts/paper_experiment/regen_plots_only.py --pairs cross_arch,strong
  PYTHONPATH=. python scripts/paper_experiment/regen_plots_only.py --greyscale-preview
  PYTHONPATH=. python scripts/paper_experiment/regen_plots_only.py --out-suffix v2
  PYTHONPATH=. python scripts/paper_experiment/regen_plots_only.py --purge-deprecated

Per spec §7 (commit 4c5596c).
"""
from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

logger = logging.getLogger("regen")

_SECTIONS = {
    "rq4": {
        "out": "results/paper_experiment/rq4-unbiasedness/accuracy",
        "script": "scripts/paper_experiment/plot_rq4_unbiasedness.py",
        "extra": [
            "--cells-root", "results/paper_experiment/trajectories/main",
            "--oracle-cells", "results/paper_experiment/trajectories/oracle_accuracy",
        ],
    },
    "rq5": {
        "out": "results/paper_experiment/rq5-signal-mse",
        "script": "scripts/paper_experiment/plot_rq5_signal_mse.py",
        "extra": [
            "--cells-root", "results/paper_experiment/trajectories/main",
            "--losses", "accuracy,cross_entropy",
        ],
    },
    "rq6": {
        "out": "results/paper_experiment/rq6-variance/accuracy",
        "script": "scripts/paper_experiment/plot_rq6_variance.py",
        "extra": [
            "--cells-root", "results/paper_experiment/trajectories/main",
            "--oracle-cells", "results/paper_experiment/trajectories/oracle_accuracy",
        ],
    },
}


def _rewrite_path(path_str: str, out_root: Path) -> str:
    """Rewrite a hardcoded results/paper_experiment/... path to use out_root."""
    prefix = "results/paper_experiment/"
    if path_str.startswith(prefix):
        return str(out_root / path_str[len(prefix):])
    return path_str


def _run_section(
    name: str,
    suffix: str | None,
    env: dict | None = None,
    paper_pairs: bool = False,
    out_root: Path | None = None,
) -> int:
    spec = _SECTIONS[name]
    out_dir = Path(spec["out"])
    if out_root:
        out_dir = out_root / Path(spec["out"]).relative_to("results/paper_experiment")
    if suffix:
        out_dir = out_dir.parent / f"{out_dir.name}_{suffix}"
    # Rewrite --cells-root / --oracle-cells paths when out_root is overridden.
    extra = spec["extra"][:]
    if out_root:
        for i, val in enumerate(extra):
            if not val.startswith("--"):
                extra[i] = _rewrite_path(val, out_root)
    cmd = [sys.executable, spec["script"]] + extra + [
        "--out", str(out_dir), "--force",
    ]
    if paper_pairs:
        cmd.append("--paper-pairs")
    if out_root:
        cmd.extend(["--out-root", str(out_root)])
    logger.info("running %s -> %s", name, out_dir)
    rc = subprocess.run(cmd, env=env, check=False).returncode
    return rc


def _purge_deprecated() -> None:
    n = 0
    for path in Path("results/paper_experiment").rglob("per_cell.pdf.deprecated"):
        path.unlink()
        n += 1
        logger.info("removed %s", path)
    logger.info("purged %d deprecated PDFs", n)


def _greyscale_preview(out_root: Path = Path("results/paper_experiment")) -> None:
    """Render a greyscale companion PDF next to each pooled.pdf / showcase.pdf.

    Uses ghostscript if available; falls back to Pillow conversion otherwise.
    """
    from shutil import which
    gs = which("gs")
    n = 0
    for pdf in out_root.rglob("*.pdf"):
        if pdf.name.endswith(".deprecated") or "_grey" in pdf.name:
            continue
        out = pdf.with_name(pdf.stem + "_grey.pdf")
        if gs:
            subprocess.run([
                gs, "-q", "-dNOPAUSE", "-dBATCH",
                "-sDEVICE=pdfwrite",
                "-sColorConversionStrategy=Gray",
                "-dProcessColorModel=/DeviceGray",
                f"-sOutputFile={out}", str(pdf),
            ], check=False)
        n += 1
    logger.info("rendered %d greyscale previews", n)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--section", choices=("rq4", "rq5", "rq6", "all"),
                        default="all")
    parser.add_argument("--pairs", type=str, default=None,
                        help="comma-separated slot names (cross_arch/weak/strong) "
                             "to restrict showcase rendering to those pairs")
    parser.add_argument("--greyscale-preview", action="store_true")
    parser.add_argument("--out-suffix", type=str, default=None)
    parser.add_argument("--purge-deprecated", action="store_true")
    parser.add_argument("--no-bands", action="store_true",
                        help="suppress ±SE/±SD bands in pooled and showcase figures")
    parser.add_argument("--font", type=str, default=None,
                        help="prepend this font name to apply_rc_helvetica's chain")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--paper-pairs", action="store_true",
                        help="Filter cells to cfg.paper_pair_keys (v0502 scope); "
                             "propagated to all child plotters.")
    parser.add_argument("--out-root", type=Path, default=None,
                        help="Override results root for input/output paths; "
                             "propagated to all child plotters.")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    if args.purge_deprecated:
        _purge_deprecated()
        return 0

    # Propagate runtime overrides to subprocess plot scripts via env vars.
    # plot_style.apply_rc_helvetica reads SAVE_PLOT_FONT;
    # plot_style.bands_enabled() reads SAVE_PLOT_NO_BANDS;
    # plot_style.get_runtime_pairs() reads SAVE_PLOT_PAIRS.
    env = os.environ.copy()
    if args.font:
        env["SAVE_PLOT_FONT"] = args.font
    if args.no_bands:
        env["SAVE_PLOT_NO_BANDS"] = "1"
    if args.pairs:
        env["SAVE_PLOT_PAIRS"] = args.pairs

    out_root = args.out_root if args.out_root else None
    sections = list(_SECTIONS) if args.section == "all" else [args.section]
    t0 = time.time()
    for sect in sections:
        rc = _run_section(
            sect,
            suffix=args.out_suffix,
            env=env,
            paper_pairs=args.paper_pairs,
            out_root=out_root,
        )
        if rc != 0:
            logger.error("%s exited %d", sect, rc)
            return rc

    if args.greyscale_preview:
        _greyscale_preview()

    logger.info("regen complete in %.1fs", time.time() - t0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
