"""Independent RNG streams, one per sampling role."""
from __future__ import annotations

import numpy as np


_ROLE_NAMES = ("celeus_acq", "baseline_order", "tiebreak", "reserve")


def spawn_role_rngs(seed: int) -> dict[str, np.random.Generator]:
    """Return one independent Generator per role, all spawned from ``seed``."""
    root = np.random.default_rng(int(seed))
    children = root.spawn(len(_ROLE_NAMES))
    return dict(zip(_ROLE_NAMES, children))
