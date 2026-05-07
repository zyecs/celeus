"""
UniformAcquisition: uniform random without-replacement acquisition.

OQ-2 resolution [plan-final.md §4 OQ-2]: uses exact sequential q_k(m)
= 1/(len(remaining) - batch_position) for the j-th item selected within
the batch (Alternative B). This ensures u_{k,m} = 1.0 algebraically under
uniform sampling. [Spec §2.1 DR property; plan-final.md §2.2]

Source: plan-final.md §3.5; Spec §4.7 Algorithm 1 (acquisition step, Stage 1 uniform case).
CLAUDE.md Rule 5: rng passed explicitly, never global state.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from save.core.state import StratumState
from save.acquisition.base import AcquisitionPolicy


class UniformAcquisition(AcquisitionPolicy):
    """
    Uniform random without-replacement acquisition policy.

    Selects n_k items uniformly at random from unlabeled items in the stratum.

    q_k(m) computation [OQ-2 Alternative B]:
    The j-th item selected within the batch (0-indexed j=0,...,n_select-1)
    sees q_k(m) = 1 / (len(remaining) - j), i.e., the exact sequential
    without-replacement probability accounting for within-batch depletion.

    This ensures u_{k,m} = 1.0 algebraically when substituted into Eq. (2).
    [Spec §2.1 DR property; plan-final.md §2.2 trivial case verification]

    Source: plan-final.md §3.5 OQ-2 Alternative B; Spec §4.7 Algorithm 1.
    """

    def select(
        self,
        stratum: StratumState,
        n_k: int,
        rng: np.random.Generator,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Select n_k items uniformly at random without replacement.

        Parameters
        ----------
        stratum : StratumState
            Current stratum state.
        n_k : int
            Requested number of items to select.
        rng : np.random.Generator
            Seeded random generator. [CLAUDE.md Rule 5]

        Returns
        -------
        local_indices : np.ndarray
            Shape (n_select,) 0-indexed local indices selected. n_select = min(n_k, |remaining|).
        q_values : np.ndarray
            Shape (n_select,) exact sequential sampling probabilities.
            q_k(m_j) = 1 / (|remaining| - j) for the j-th item selected (0-indexed j).
            [OQ-2 Alternative B; plan-final.md §3.5; Spec §2.1 Eq. (2)]

        Source: plan-final.md §3.5 OQ-2 Alternative B.
        """
        # Get unlabeled items available for selection [Blueprint §1.1]
        remaining = stratum.remaining_indices()  # 0-indexed local indices
        n_remaining = len(remaining)
        n_select = min(n_k, n_remaining)

        if n_select == 0:
            return np.array([], dtype=int), np.array([], dtype=np.float64)

        # Uniform WOR selection [Spec §4.7 Algorithm 1 acquisition step]
        # [DERIVED — verify]: np.random.Generator.choice(replace=False) gives WOR.
        # ASSUMPTION: choice(replace=False) returns items in sequential draw order
        # (j-th output = j-th item drawn), matching Fisher-Yates internals.
        # This is required for q_values below to be correct. Empirically verified
        # by E4 gate (all weights = 1.0 to machine precision). Not formally
        # guaranteed in numpy API docs — monitor across numpy version upgrades.
        selected_local = rng.choice(remaining, size=n_select, replace=False)

        # Exact sequential q_k(m): the j-th draw (0-indexed) from |remaining| items
        # has probability 1 / (|remaining| - j).
        # [OQ-2 Alternative B; plan-final.md §3.5; Spec §2.1 DR property]
        # This gives u_{k,m} = 1.0 exactly via the algebra in plan-final.md §2.2.
        j_indices = np.arange(n_select, dtype=np.float64)
        q_values = 1.0 / (n_remaining - j_indices)  # shape (n_select,)

        return selected_local, q_values

    def get_proposal(
        self,
        stratum: StratumState,
        n_k: int,  # noqa: ARG002 — honoured for interface parity with select
    ) -> np.ndarray:
        """
        Return the uniform proposal over remaining items.

        For uniform WOR the first-draw proposal is flat: 1 / |remaining|.
        Subsequent draws within a batch renormalize the same uniform shape,
        but this method only exposes the step-0 distribution because replay
        analysis runs under n_select = 1 (guarded by G4 in the rq4-rq6 spec).

        Returns np.empty(0) when |remaining| == 0.

        Source: rq4-rq6 design §5.6 + §4.3.
        """
        remaining = stratum.remaining_indices()
        n_remaining = len(remaining)
        if n_remaining == 0:
            return np.empty(0, dtype=np.float64)
        return np.full(n_remaining, 1.0 / n_remaining, dtype=np.float64)
