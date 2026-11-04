"""Stage 0.5 disk + inode + probe audit (spec §9)."""
from __future__ import annotations

import os
import shutil
from pathlib import Path

import numpy as np


class AuditFailure(RuntimeError):
    pass


def check_bytes_free(path: Path, min_bytes: int) -> None:
    usage = shutil.disk_usage(path)
    if usage.free < min_bytes:
        raise AuditFailure(
            f"free bytes on {path} = {usage.free:,} < required {min_bytes:,}"
        )


def check_inodes_free(path: Path, min_inodes: int) -> None:
    vfs = os.statvfs(path)
    if vfs.f_favail < min_inodes:
        raise AuditFailure(
            f"free inodes on {path} = {vfs.f_favail} < required {min_inodes}"
        )


def check_readable(path: Path) -> None:
    if not path.exists():
        raise AuditFailure(f"input path {path} does not exist")
    if not os.access(path, os.R_OK):
        raise AuditFailure(f"input path {path} not readable")


def check_executable(path: Path) -> None:
    if not path.exists():
        raise AuditFailure(f"python interpreter {path} missing")
    if not os.access(path, os.X_OK):
        raise AuditFailure(f"python interpreter {path} not executable")


def write_probe_and_read_back(path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Ensure the path ends in .npz so numpy doesn't silently rename it.
    assert str(path).endswith(".npz"), f"probe path must end in .npz: {path}"
    a = np.arange(16, dtype=np.int64)
    np.savez_compressed(path, a=a)
    with np.load(path) as f:
        b = f["a"]
    if not np.array_equal(a, b):
        raise AuditFailure(f"probe roundtrip mismatch at {path}")
    path.unlink()


def check_no_uncommitted_src_deletions(repo_root: Path) -> None:
    """Spec §9 Stage 0.5: no uncommitted deleted files in src/."""
    import subprocess

    out = subprocess.run(
        ["git", "status", "--porcelain", "--", "src/"],
        cwd=repo_root, capture_output=True, text=True, check=False,
    )
    if out.returncode != 0:
        raise AuditFailure(f"git status failed: {out.stderr.strip()}")
    for line in out.stdout.splitlines():
        # Porcelain lines look like " D path" or "D  path" or "AD path" etc.
        if line[:2].strip().startswith("D") or line[:2].strip().endswith("D"):
            raise AuditFailure(
                f"uncommitted deleted file in src/: {line.strip()}"
            )
