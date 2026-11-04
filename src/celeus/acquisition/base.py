"""
AcquisitionPolicy abstract base class for CELEUS.

Defines the interface for acquisition policies that select which items to label.

``select()`` returns both the chosen indices and their sampling probabilities.
The policy already knows those probabilities at selection time, so returning
them keeps the importance-weight logic in one place.

``get_proposal()`` exposes the full probability mass function over unlabeled
items without drawing. The replay diagnostics need it to recompute conditional
variances.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np

from celeus.core.state import StratumState


class AcquisitionPolicy(ABC):
    """
    Abstract base class for acquisition policies.

    Subclasses implement select() to choose which unlabeled items to query.

    Invariants enforced:
      - len(local_indices) == n_k
      - All selected indices are currently unlabeled (not in stratum.labeled_mask)
      - No repeats within a single call
      - len(q_values) == len(local_indices)
      - All q_values > 0
    """

    @abstractmethod
    def select(
        self,
        stratum: StratumState,
        n_k: int,
        rng: np.random.Generator,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Select n_k unlabeled items from the stratum for labeling.

        Parameters
        ----------
        stratum : StratumState
            Current stratum state. Use stratum.remaining_indices() to get
            available unlabeled items.
        n_k : int
            Number of items to select. May be capped by available capacity.
        rng : np.random.Generator
            Seeded random generator. No global random state is used.

        Returns
        -------
        local_indices : np.ndarray
            Shape (n_select,) 0-indexed local indices within this stratum.
            n_select = min(n_k, len(remaining)). No duplicates.
        q_values : np.ndarray
            Shape (n_select,) sampling probabilities q_k(m) for each selected item.
            For uniform sampling without replacement:
            q_k(m) = 1/(|remaining| - batch_position).
        """
        ...

    @abstractmethod
    def get_proposal(
        self,
        stratum: StratumState,
        n_k: int,
    ) -> np.ndarray:
        """
        Return the current proposal distribution over unlabeled items.

        Unlike ``select``, this does NOT draw — it only exposes the probability
        mass function q_t(j | F_{t-1}, D_N) that ``select`` would use on its
        first draw. Needed by replay analysis to compute conditional variances
        over J_{t-1}.

        Parameters
        ----------
        stratum : StratumState
            Current stratum state, providing ``remaining_indices()``,
            ``surrogate_scores``, and, for proxy-score policies, ``ell_proxy``.
        n_k : int
            Requested number of draws; used only to honour the early-return
            convention of ``select`` (an empty array when the pool is empty).

        Returns
        -------
        pool_probs : np.ndarray
            Shape (|remaining|,) float64. The probability mass function over
            ``stratum.remaining_indices()`` — indexed by position within that
            remaining array, NOT by global pool identity. Callers mapping back
            to global identities must use ``stratum.pool_indices[remaining[i]]``.
            All entries are ≥ 0; entries sum to 1.0. An empty array is returned
            when the pool is empty.
        """
        ...
