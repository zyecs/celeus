# src/save/paper_experiment/traj_utils.py
"""Small utilities for stretching ragged trajectories to a fixed T_max."""
from __future__ import annotations

import numpy as np


def pad_trajectory(arr: np.ndarray, T: int, fill: str = "last") -> np.ndarray:
    """Return a length-T copy of ``arr``, padded with either the last value or NaN/-1.

    - ``fill="last"`` — carry the final observed value forward (or leave untouched).
    - ``fill="nan"``  — pad with NaN (float64) / -1 (int).
    """
    arr = np.asarray(arr)
    if arr.shape[0] >= T:
        return arr[:T].copy()
    out = np.empty(T, dtype=arr.dtype)
    out[: arr.shape[0]] = arr
    if fill == "last":
        if arr.shape[0] == 0:
            raise ValueError("cannot pad empty array with fill='last'")
        out[arr.shape[0]:] = arr[-1]
    elif fill == "nan":
        if np.issubdtype(arr.dtype, np.floating):
            out[arr.shape[0]:] = np.nan
        else:
            out[arr.shape[0]:] = -1
    else:
        raise ValueError(f"unknown fill mode: {fill!r}")
    return out
