"""Refuse to run on login nodes."""
from __future__ import annotations

import socket

LOGIN_NODE_PREFIXES: tuple[str, ...] = ("login",)


def is_login_node(hostname: str) -> bool:
    return any(hostname.startswith(p) for p in LOGIN_NODE_PREFIXES)


def ensure_compute_node() -> None:
    h = socket.gethostname()
    if is_login_node(h):
        raise RuntimeError(
            f"Refusing to run on login node {h!r}. "
            "Submit via srun/sbatch to a compute node."
        )
