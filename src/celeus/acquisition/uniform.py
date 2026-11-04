"""
UniformAcquisition: uniform random without-replacement acquisition.

Within a batch, the j-th item drawn (0-indexed) is assigned the exact
sequential probability q_k(m) = 1 / (len(remaining) - j). Accounting for
within-batch depletion this way makes the importance weights u_{k,m} equal to
1 algebraically, so under uniform sampling the estimator reduces to the plain
sample mean of the observed losses.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from celeus.core.state import StratumState
from celeus.acquisition.base import AcquisitionPolicy


class UniformAcquisition(AcquisitionPolicy):
    """
    Uniform random without-replacement acquisition policy.

    Selects n_k items uniformly at random from unlabeled items in the stratum.

    q_k(m) computation:
    The j-th item selected within the batch (0-indexed j=0,...,n_select-1)
    sees q_k(m) = 1 / (len(remaining) - j), i.e., the exact sequential
    without-replacement probability accounting for within-batch depletion.
    This makes u_{k,m} = 1.0 algebraically, so the estimator collapses to the
    sample mean. It is the trivial case used to check the estimator algebra.
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
            Seeded random generator.

        Returns
        -------
        local_indices : np.ndarray
            Shape (n_select,) 0-indexed local indices selected. n_select = min(n_k, |remaining|).
        q_values : np.ndarray
            Shape (n_select,) exact sequential sampling probabilities.
            q_k(m_j) = 1 / (|remaining| - j) for the j-th item selected (0-indexed j).
        """
        # Unlabeled items available for selection.
        remaining = stratum.remaining_indices()  # 0-indexed local indices
        n_remaining = len(remaining)
        n_select = min(n_k, n_remaining)

        if n_select == 0:
            return np.array([], dtype=int), np.array([], dtype=np.float64)

        # Uniform without-replacement selection.
        # ASSUMPTION: choice(replace=False) returns items in sequential draw order
        # (j-th output = j-th item drawn), matching Fisher-Yates internals.
        # This is required for q_values below to be correct. The tests check it
        # (all weights equal 1.0 to machine precision). NumPy does not document
        # the guarantee, so recheck it after a NumPy upgrade.
        selected_local = rng.choice(remaining, size=n_select, replace=False)

        # Exact sequential q_k(m): the j-th draw (0-indexed) from |remaining|
        # items has probability 1 / (|remaining| - j). This is what makes
        # u_{k,m} = 1.0 exactly.
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
        analysis runs with n_select = 1.

        Returns np.empty(0) when |remaining| == 0.
        """
        remaining = stratum.remaining_indices()
        n_remaining = len(remaining)
        if n_remaining == 0:
            return np.empty(0, dtype=np.float64)
        return np.full(n_remaining, 1.0 / n_remaining, dtype=np.float64)
