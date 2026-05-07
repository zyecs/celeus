"""Build the v0502 snapshot: symlinks for legacy trajectories filtered to
paper_pairs (excluding M5), filter+rewrite per_seed_curves.csv files, copy
scalar inputs, and emit manifest.json with full provenance.

See spec §7.1 for the design.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))

from save.paper_experiment.config import default_config_path, load_config


def _parse_cell_path(path: str) -> tuple[str, str, str, str, str]:
    """Parse cell filename into (method, dataset, surrogate, target, loss_or_oracle)."""
    name = path.rsplit("/", 1)[-1]
    body = name.replace("cell__", "").replace(".npz", "")
    parts = body.split("__")
    return parts[0], parts[1], parts[2], parts[3], parts[4]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _preflight(target_dir: Path, min_free_gb: int = 10) -> None:
    """Pre-flight: verify legacy dir present, target writable, free space available."""
    legacy = _REPO / "results" / "paper_experiment"
    if not legacy.is_dir():
        sys.exit(f"legacy dir missing: {legacy}")
    free_bytes = shutil.disk_usage(target_dir).free
    free_gb = free_bytes // (1024 ** 3)
    if free_gb < min_free_gb:
        sys.exit(f"insufficient free space: {free_gb} GB < {min_free_gb} GB required")
    print(f"[preflight] free: {free_gb} GB; legacy dir present; target {target_dir}")


def _symlink_filtered_cells(
    legacy_root: Path,
    v0502_root: Path,
    subdir: str,
    paper_pairs: set[tuple[str, str]],
    *,
    skip_m5: bool = True,
) -> list[dict]:
    """Symlink cells under legacy_root/<subdir> into v0502_root/<subdir>,
    keeping only cells whose (surrogate, target) is in paper_pairs."""
    src_dir = legacy_root / subdir
    dst_dir = v0502_root / subdir
    dst_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    if not src_dir.is_dir():
        return manifest_rows
    for src in sorted(src_dir.glob("cell__*.npz")):
        method, ds, surr, tgt, loss = _parse_cell_path(str(src))
        if skip_m5 and method == "M5":
            continue
        if (surr, tgt) not in paper_pairs:
            continue
        dst = dst_dir / src.name
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        os.symlink(src.resolve(), dst)
        manifest_rows.append({
            "src": str(src), "dst": str(dst),
            "sha256": _sha256(src),
        })
    return manifest_rows


def _filter_csv_files(
    legacy_root: Path,
    v0502_root: Path,
    paper_pairs: set[tuple[str, str]],
) -> list[dict]:
    """Filter+rewrite per_seed_curves.csv AND per_cell_curves.csv files for rq4/5/6."""
    csv_targets = [
        "rq4-unbiasedness/accuracy/per_seed_curves.csv",
        "rq5-signal-mse/accuracy/per_seed_curves.csv",
        "rq5-signal-mse/cross_entropy/per_seed_curves.csv",
        "rq6-variance/accuracy/per_seed_curves.csv",
        "rq4-unbiasedness/accuracy/per_cell_curves.csv",
        "rq5-signal-mse/accuracy/per_cell_curves.csv",
        "rq5-signal-mse/cross_entropy/per_cell_curves.csv",
        "rq6-variance/accuracy/per_cell_curves.csv",
    ]
    manifest_rows = []
    for rel in csv_targets:
        src = legacy_root / rel
        if not src.is_file():
            continue
        df = pd.read_csv(src)
        if "cell" not in df.columns:
            print(f"[warn] {rel} has no 'cell' column; skipping filter")
            continue

        def keep_row(row) -> bool:
            try:
                _, _, surr, tgt, _ = _parse_cell_path(row["cell"])
            except (ValueError, IndexError):
                return False
            return (surr, tgt) in paper_pairs

        keep_mask = df.apply(keep_row, axis=1)
        filtered = df[keep_mask].copy()

        legacy_prefix = "results/paper_experiment/"
        v0502_prefix = "results/paper_experiments_v0502/"
        filtered["cell"] = filtered["cell"].str.replace(legacy_prefix, v0502_prefix, regex=False)

        dst = v0502_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        filtered.to_csv(dst, index=False)

        manifest_rows.append({
            "src": str(src), "dst": str(dst),
            "src_sha256": _sha256(src), "dst_sha256": _sha256(dst),
            "rows_in": len(df), "rows_out": len(filtered),
        })
    return manifest_rows


def _symlink_scalar_inputs(legacy_root: Path, v0502_root: Path) -> list[dict]:
    """Symlink R_N.csv, ce_sweep_winner.json, extreme_pairs.yaml into v0502."""
    files = ["R_N.csv", "ce_sweep_winner.json", "extreme_pairs.yaml"]
    manifest_rows = []
    for f in files:
        src = legacy_root / f
        if not src.is_file():
            continue
        dst = v0502_root / f
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        os.symlink(src.resolve(), dst)
        manifest_rows.append({"path": f, "src_sha256": _sha256(src)})
    return manifest_rows


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()


def _git_dirty() -> bool:
    out = subprocess.run(
        ["git", "status", "--porcelain"], check=True, capture_output=True, text=True
    ).stdout.strip()
    return bool(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", type=Path,
                    default=_REPO / "results" / "paper_experiments_v0502")
    args = ap.parse_args(argv)
    out_root: Path = args.out_root
    out_root.mkdir(parents=True, exist_ok=True)
    _preflight(out_root)

    cfg_path = default_config_path()
    cfg = load_config(cfg_path)
    paper_pairs = cfg.paper_pair_keys
    legacy_root = _REPO / "results" / "paper_experiment"

    main_rows = _symlink_filtered_cells(legacy_root, out_root, "trajectories/main", paper_pairs)
    oracle_rows = _symlink_filtered_cells(legacy_root, out_root, "trajectories/oracle_accuracy", paper_pairs)

    # Symlink legacy acq_sweep cells (intact) for the supplementary subfigure.
    legacy_acq = legacy_root / "trajectories" / "acquisition_sweep"
    legacy_acq_rows = []
    if legacy_acq.is_dir():
        dst_acq = out_root / "trajectories" / "acquisition_sweep"
        dst_acq.mkdir(parents=True, exist_ok=True)
        for src in sorted(legacy_acq.glob("cell__*.npz")):
            dst = dst_acq / src.name
            if not dst.exists():
                os.symlink(src.resolve(), dst)
                legacy_acq_rows.append({"src": str(src), "dst": str(dst), "sha256": _sha256(src)})

    per_seed_rows = _filter_csv_files(legacy_root, out_root, paper_pairs)
    scalar_rows = _symlink_scalar_inputs(legacy_root, out_root)

    pip_freeze = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"],
        capture_output=True, text=True,
    ).stdout
    cfg_rel = cfg_path.relative_to(_REPO) if cfg_path.is_absolute() else cfg_path
    config_diff = subprocess.run(
        ["git", "log", "-p", "--no-merges",
         str(cfg_rel)],
        capture_output=True, text=True,
    ).stdout[:8000]

    # Merge-not-overwrite: preserve fields populated by E3.
    manifest_path = out_root / "manifest.json"
    existing = {}
    if manifest_path.is_file():
        try:
            existing = json.loads(manifest_path.read_text())
        except json.JSONDecodeError:
            existing = {}
    preserved_acq_run = existing.get("scope_mapping", {}).get("acq_sweep_v0502_run", [])

    manifest = {
        "schema_version": "1",
        "v0502_built_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "git_dirty": _git_dirty(),
        "config_path": str(cfg_path),
        "config_diff_against_v0426": config_diff,
        "paper_pairs": cfg.paper_pairs,
        "scope_mapping": {
            "main_cells_symlinked": main_rows,
            "oracle_cells_symlinked": oracle_rows,
            "acq_sweep_legacy_symlinked": legacy_acq_rows,
            "acq_sweep_v0502_run": preserved_acq_run,
        },
        "per_seed_curves_filtered": per_seed_rows,
        "scalar_inputs_symlinked": scalar_rows,
        "selection_predicate": "(surrogate, target) in cfg.paper_pair_keys; method != 'M5'",
        "environment": {
            "python": sys.version.split()[0],
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "user": os.environ.get("USER", "unknown"),
            "pip_freeze": pip_freeze,
        },
        "row_count_checks": existing.get("row_count_checks", {}),
        "command_line": "python scripts/paper_experiment/build_v0502_snapshot.py "
                        + " ".join(argv if argv else []),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str))
    print(f"wrote manifest: {manifest_path}")
    print(f"summary: main={len(main_rows)}, oracle={len(oracle_rows)}, "
          f"acq_legacy={len(legacy_acq_rows)}, csvs={len(per_seed_rows)}, "
          f"scalar={len(scalar_rows)}, "
          f"preserved acq_v0502_run entries={len(preserved_acq_run)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
