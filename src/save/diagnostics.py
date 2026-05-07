"""
SAVE trajectory recorder.

TrajectoryRecorder logs per-round statistics to a .npz file including
git commit hash and full config YAML.

Source: Spec §4.7 Algorithm 1 line 22 ("RECORD: Log R̂^t, pop_lo, pop_hi,
        pop_w, ESS, clip_rates"); CLAUDE.md Rule 6 (git hash + config).
"""

from __future__ import annotations

import subprocess
import os
from typing import List

import numpy as np
import yaml

from save.core.state import SAVEConfig, StratumState


def _get_git_hash(project_root: str | None = None) -> str:
    """
    Return the current git commit hash.

    Falls back to 'unknown' if git is unavailable (e.g., in CI).
    Source: CLAUDE.md Rule 6 ("Every experiment logs git commit hash").
    """
    try:
        cwd = project_root if project_root is not None else os.getcwd()
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=cwd,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "unknown"


class TrajectoryRecorder:
    """
    Records per-round SAVE statistics and saves them to .npz.

    Usage
    -----
    recorder = TrajectoryRecorder(K=1, config=config, git_hash=hash_str)
    recorder.record(t, R_hat, lower, upper, pop_lower, pop_upper,
                    total_labels, weight_clip_rate, output_clip_rate,
                    ess_per_stratum)
    recorder.save("output/trajectory.npz")

    Source: Spec §4.7 Algorithm 1 line 22; CLAUDE.md Rule 6.
    """

    def __init__(self, K: int, config: SAVEConfig, git_hash: str) -> None:
        """
        Parameters
        ----------
        K : int
            Number of strata (determines ESS shape).
        config : SAVEConfig
            Run configuration (serialized to YAML in output file).
        git_hash : str
            Git commit hash for reproducibility. [CLAUDE.md Rule 6]
        """
        self.K = K
        self.config = config
        self.git_hash = git_hash

        # Buffer lists — one entry per recorded round
        self._t: list[int] = []
        self._R_hat: list[float] = []
        self._lower: list[float] = []
        self._upper: list[float] = []
        self._pop_lower: list[float] = []
        self._pop_upper: list[float] = []
        self._total_labels: list[int] = []
        self._weight_clip_rate: list[float] = []
        self._output_clip_rate: list[float] = []
        self._ess: list[list[float]] = []  # shape (T, K) after conversion

    def record(
        self,
        t: int,
        R_hat: float,
        lower: float,
        upper: float,
        pop_lower: float,
        pop_upper: float,
        total_labels: int,
        weight_clip_rate: float,
        output_clip_rate: float,
        ess_per_stratum: List[float],
    ) -> None:
        """
        Append one row to all internal buffer lists.

        Source: Spec §4.7 Algorithm 1 line 22.
        """
        self._t.append(int(t))
        self._R_hat.append(float(R_hat))
        self._lower.append(float(lower))
        self._upper.append(float(upper))
        self._pop_lower.append(float(pop_lower))
        self._pop_upper.append(float(pop_upper))
        self._total_labels.append(int(total_labels))
        self._weight_clip_rate.append(float(weight_clip_rate))
        self._output_clip_rate.append(float(output_clip_rate))
        self._ess.append([float(e) for e in ess_per_stratum])

    @staticmethod
    def compute_ess(stratum: StratumState) -> float:
        """
        Effective sample size for a stratum.

        Under v0320, there are no IS weights — ESS equals M_k (all items
        have equal contribution). Retained for API compatibility with
        trajectory recording.

        [Paper v0320: no IS weights, ESS = M_k]
        """
        return float(stratum.M_k)

    def save(self, path: str) -> None:
        """
        Save trajectory to .npz file.

        Saved arrays:
          t, R_hat, lower, upper, pop_lower, pop_upper,
          total_labels, weight_clip_rate, output_clip_rate,
          ess (shape T x K), git_hash (string scalar), config_yaml (string scalar).

        Source: CLAUDE.md Rule 6 ("Every experiment logs git commit hash + full config").
        """
        # Convert buffers to numpy arrays
        t_arr = np.array(self._t, dtype=np.int64)
        R_hat_arr = np.array(self._R_hat, dtype=np.float64)
        lower_arr = np.array(self._lower, dtype=np.float64)
        upper_arr = np.array(self._upper, dtype=np.float64)
        pop_lower_arr = np.array(self._pop_lower, dtype=np.float64)
        pop_upper_arr = np.array(self._pop_upper, dtype=np.float64)
        total_labels_arr = np.array(self._total_labels, dtype=np.int64)
        weight_clip_rate_arr = np.array(self._weight_clip_rate, dtype=np.float64)
        output_clip_rate_arr = np.array(self._output_clip_rate, dtype=np.float64)

        # ESS: shape (T, K)
        if self._ess:
            ess_arr = np.array(self._ess, dtype=np.float64)  # (T, K)
        else:
            ess_arr = np.zeros((0, self.K), dtype=np.float64)

        # Serialize config to YAML string [CLAUDE.md Rule 6]
        import dataclasses
        config_dict = dataclasses.asdict(self.config)
        config_yaml_str = yaml.dump(config_dict, default_flow_style=False)

        # Ensure parent directory exists
        import os
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

        np.savez(
            path,
            t=t_arr,
            R_hat=R_hat_arr,
            lower=lower_arr,
            upper=upper_arr,
            pop_lower=pop_lower_arr,
            pop_upper=pop_upper_arr,
            total_labels=total_labels_arr,
            weight_clip_rate=weight_clip_rate_arr,
            output_clip_rate=output_clip_rate_arr,
            ess=ess_arr,
            git_hash=np.array(self.git_hash),
            config_yaml=np.array(config_yaml_str),
        )

    @classmethod
    def load(cls, path: str) -> dict:
        """
        Load a saved trajectory .npz file.

        Returns
        -------
        dict-like
            np.load result (NpzFile), accessible by key.
        """
        return np.load(path, allow_pickle=True)
