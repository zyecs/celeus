"""
AcquisitionPolicy abstract base class for SAVE.

Defines the interface for acquisition policies that select which items to label.

OQ-1 resolution [plan-final.md §4 OQ-1]: select() returns (local_indices, q_values).
  The acquisition policy knows sampling probabilities at selection time.
  Returning q_values from select() keeps the IS weight logic clean (Alternative A).

get_proposal() exposes the full probability mass function over unlabeled items
  without drawing, required by rq4-rq6 replay analysis.

Source: plan-final.md §3.4; Spec §4.7 Algorithm 1 (acquisition step).
CLAUDE.md G1: every formula cites source; Rule 5: rng passed explicitly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np

from save.core.state import StratumState


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

    Source: plan-final.md §3.4; Spec §4.7 Algorithm 1.
    OQ-1 [plan-final.md §4]: select() returns (local_indices, q_values) — Alternative A.
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
            available unlabeled items. [Blueprint §1.1]
        n_k : int
            Number of items to select. May be capped by available capacity.
        rng : np.random.Generator
            Seeded random generator. [CLAUDE.md Rule 5 — no global random state]

        Returns
        -------
        local_indices : np.ndarray
            Shape (n_select,) 0-indexed local indices within this stratum.
            n_select = min(n_k, len(remaining)). No duplicates.
        q_values : np.ndarray
            Shape (n_select,) sampling probabilities q_k(m) for each selected item.
            For uniform WOR: q_k(m) = 1/(|remaining| - batch_position).
            [Spec §2.1 Eq. (2); plan-final.md OQ-2 Alternative B]

        Source: plan-final.md §3.4; OQ-1 resolution (Alternative A).
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
            ``surrogate_scores``, and (for Remark 1/2 policies) ``ell_proxy``.
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

        Source: rq4-rq6 design §5.6.
        """
        ...
