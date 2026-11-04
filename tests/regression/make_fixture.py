"""Regenerate the trajectory fixture. Run only on a commit whose behaviour is trusted."""
from pathlib import Path

import numpy as np

from cases import run_all_cases

FIXTURE = Path(__file__).with_name("trajectories.npz")

if __name__ == "__main__":
    np.savez_compressed(FIXTURE, **run_all_cases())
    print(f"wrote {FIXTURE}")
