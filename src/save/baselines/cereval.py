"""
Cer-Eval baseline — adaptive stratified sequential testing.

Faithful port of seq_test_adaptive_z_full_width from
save-v0320/examples/cer_eval.py.

Only changes from reference:
1. np.random.seed/shuffle/permutation → explicit rng (CLAUDE.md Rule 5)
2. SimData class → direct numpy arrays
3. partition_z gains rng parameter
4. Guard Nk[j]==0 in partition_z
5. Guard empty stratum in main loop
6. Guard N < m_init
7. Trajectory recording per label

"""

from __future__ import annotations

import time
import warnings

import numpy as np

from save.partition import EvaluationPool


def _partition_z(
    embeddings: np.ndarray,
    losses: np.ndarray,
    partition_labels: np.ndarray,
    labeled_mask: np.ndarray,
    delta: float,
    rng: np.random.Generator,
) -> tuple[int, object, np.ndarray]:
    """
    Find optimal number of strata K and train 1-NN classifier.

    Faithful port of partition_z() from cer_eval.py with RNG threading.

    Parameters
    ----------
    embeddings : np.ndarray, shape (N, D)
    losses : np.ndarray, shape (N,)
    partition_labels : np.ndarray, shape (N,) — mutated in place
    labeled_mask : np.ndarray, shape (N,) bool — which items are labeled
    delta : float
    rng : np.random.Generator — CLAUDE.md Rule 5

    Returns
    -------
    (K, clf, partition_labels) — optimal K, trained 1-NN, updated labels
    """
    # Lazy import — spec §3 dependencies
    try:
        from sklearn.neighbors import KNeighborsClassifier
    except ImportError:
        raise ImportError(
            "scikit-learn is required for CerEvalBaseline. "
            "Install with: pip install scikit-learn"
        )

    ind = np.where(labeled_mask)[0]
    N_total = len(embeddings)
    N_labeled = len(ind)
    z = losses[ind]

    max_part = min(int(np.sqrt(N_labeled)), 5)
    epsilons = np.full(max_part, np.inf)

    for k in range(1, max_part + 1):
        knn = KNeighborsClassifier(n_neighbors=1)

        # Assign bin labels to labeled items
        # Faithful: np.floor(z * (k - 0.01/k))
        partition_labels[ind] = np.floor(z * (k - 0.01 / k)).astype(int)

        idx = np.arange(N_labeled)
        rng.shuffle(idx)  # replaces np.random.shuffle(idx) — CLAUDE.md Rule 5

        # Subsampling — preserved from reference
        num = int(N_labeled * min(0.8, max((N_labeled / N_total) ** 1.5, 0.5)))
        knn.fit(embeddings[ind[idx[:num]]], partition_labels[ind[idx[:num]]])

        # Predict for ALL items — reference behavior
        partition_labels[:] = knn.predict(embeddings)
        Nk = np.array([len(np.where(partition_labels == i)[0]) for i in range(k)])
        density = Nk / N_total

        inds = [np.where(partition_labels[ind] == j)[0] for j in range(k)]
        nk = np.array([len(_) for _ in inds])

        # Guard: nk.min() <= 1 → break (reference behavior)
        if nk.min() <= 1:
            break

        # Guard: Nk[j] == 0 → skip this K candidate
        if np.any(Nk == 0):
            continue

        # Compute weighted epsilon for this K
        r = np.zeros(k)
        v = np.zeros(k)
        eta = np.zeros(k)
        epsilon = np.zeros(k)

        mu = np.mean(z)
        v_all = np.std(z) ** 2  # ddof=0 [POTENTIAL ISSUE]: biased variance, faithful to reference

        for j in range(k):
            r[j] = np.mean(z[inds[j]])
            v[j] = np.std(z[inds[j]]) ** 2  # ddof=0, faithful to reference
            p = nk[j] / Nk[j]
            mu_avg = p * r[j] + (1 - p) * mu
            # Variance mixture — law of total variance, faithful to reference
            v[j] = (
                p * v[j]
                + (1 - p) * v_all
                + p * (r[j] - mu_avg) ** 2
                + (1 - p) * (mu - mu_avg) ** 2
            )
            eta[j] = np.sqrt(
                (2 * np.log(np.log(nk[j]) + 1) + np.log(4 * k / delta))
                / nk[j]
                / 2
            )
            epsilon[j] = 2 * eta[j] ** 2 / 3 + 2 * np.sqrt(v[j]) * eta[j]

        epsilons[k - 1] = density @ epsilon

    # Select best K
    K = int(np.argmin(epsilons) + 1)

    # Retrain with optimal K — faithful to reference (redundant clf.predict preserved)
    clf = KNeighborsClassifier(n_neighbors=1)
    partition_labels[ind] = np.floor(z * (K - 0.01 / K)).astype(int)

    idx = np.arange(N_labeled)
    rng.shuffle(idx)  # replaces np.random.shuffle(idx)

    num = int(N_labeled * min(0.8, max((N_labeled / N_total) ** 1.5, 0.5)))
    clf.fit(embeddings[ind[idx[:num]]], partition_labels[ind[idx[:num]]])

    return K, clf, partition_labels


class CerEvalBaseline:
    """
    Cer-Eval adaptive stratified sequential testing baseline.

    Faithful port of seq_test_adaptive_z_full_width.
    Uses pool.ground_truth_losses directly (no oracle_fn indirection).

    Parameters
    ----------
    pool : EvaluationPool
        Must have pool.embeddings not None.
    C_full : float
        Full CI width stopping threshold (maps to config.epsilon).
    delta : float
        Error budget (maps to config.alpha_1 + config.alpha_2).
    m_init : int
        Number of warm-up labels.
    seed : int | None
        Random seed. CLAUDE.md Rule 4.
    rng : np.random.Generator | None
        Injected random generator. Mutually exclusive with seed.

    Source: spec §3; cer_eval.py reference.
    """

    def __init__(
        self,
        pool: EvaluationPool,
        C_full: float,
        delta: float,
        m_init: int,
        seed: int | None = None,
        T_max: int | None = None,
        monitor_to_T_max: bool = False,
        rng: np.random.Generator | None = None,
    ):
        assert pool.embeddings is not None, "CerEvalBaseline requires pool.embeddings"
        assert pool.ground_truth_losses is not None
        self.pool = pool
        self.C_full = C_full
        self.delta = delta
        if rng is not None and seed is not None:
            raise ValueError("pass exactly one of seed= or rng=")
        if rng is not None:
            self.rng = rng
            self.seed = None
        elif seed is not None:
            self.seed = seed
            self.rng = np.random.default_rng(seed)
        else:
            raise ValueError("must pass seed= or rng=")

        N = pool.N
        # Guard: N < m_init — clamp with warning
        if m_init >= N:
            warnings.warn(
                f"m_init ({m_init}) >= N ({N}); clamping to N-1={N-1}.",
                stacklevel=2,
            )
            m_init = N - 1
        self.m_init = m_init
        # paper_experiment spec §4: optional query-budget cap + monitor flag.
        self.T_max = int(T_max) if T_max is not None else pool.N
        self.monitor_to_T_max = bool(monitor_to_T_max)
        # Degenerate guard: T_max < m_init would make the guard fire on entry.
        if self.T_max < self.m_init:
            raise ValueError(
                f"T_max ({self.T_max}) must be >= m_init ({self.m_init})"
            )

    def run(self) -> dict:
        """
        Run Cer-Eval adaptive stratified sequential test.

        Returns
        -------
        dict
            8-key trajectory dict. ``pop_lower == lower``, ``pop_upper == upper``.
            Adds ``round_times`` (paper §6.5 item #5, Task 9): one
            ``time.perf_counter()`` delta per ``while True:`` iteration that
            recorded a trajectory row.
        """
        N = self.pool.N
        embeddings = self.pool.embeddings
        losses = self.pool.ground_truth_losses.copy()
        partition_labels = np.zeros(N, dtype=int)

        # Shuffle — replaces SimData.shuffle() + np.random.permutation
        perm = self.rng.permutation(N)
        embeddings = embeddings[perm].copy()
        losses = losses[perm].copy()

        # L2-normalize — faithful to reference build_dt_from_arrays
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / np.maximum(norms, 1e-12)

        eps_min = np.sqrt(np.log(1 / self.delta) / N / 2)
        m = self.m_init

        # labeled_mask tracks which items (in shuffled order) are labeled
        labeled_mask = np.zeros(N, dtype=bool)
        labeled_mask[:m] = True

        # Trajectory recording
        t_list = []
        r_hat_list = []
        lower_list = []
        upper_list = []
        total_labels_list = []
        # Per-round wall-clock instrumentation (paper §6.5 item #5, Task 9).
        # One entry per ``while True:`` iteration that records a trajectory
        # row. Append occurs before every ``break`` and at the natural
        # bottom of the loop, so the array length matches ``len(t_list)``.
        round_times: list[float] = []

        old = m  # re-partition threshold
        K = 1
        density = np.array([1.0])

        while True:
            _round_t0 = time.perf_counter()
            n_labeled = int(labeled_mask.sum())
            ind = np.where(labeled_mask)[0]
            remain = np.where(~labeled_mask)[0]

            if remain.size == 0:
                # paper_experiment (spec §4): record the true full-pool state
                # before exiting so the last trajectory row reflects
                # n_labeled == N with FRESH r_hat / epsilon (not stale carry-
                # forward from the previous iteration).
                if t_list and t_list[-1] < n_labeled:
                    ind_full = np.where(labeled_mask)[0]
                    z_full = losses[ind_full]
                    mu_full = np.mean(z_full)
                    v_all_full = np.std(z_full) ** 2
                    Nk_full = np.array([
                        len(np.where(partition_labels == k)[0]) for k in range(K)
                    ])
                    density_full = Nk_full / N

                    r_full = np.zeros(K)
                    v_full = np.zeros(K)
                    eta_full = np.zeros(K)
                    eps_k_full = np.zeros(K)
                    inds_full = [
                        np.where(partition_labels[ind_full] == k)[0] for k in range(K)
                    ]
                    nk_full = np.array([len(ii) for ii in inds_full])
                    for k in range(K):
                        if nk_full[k] == 0 or Nk_full[k] == 0:
                            continue
                        r_full[k] = np.mean(z_full[inds_full[k]])
                        v_full[k] = np.std(z_full[inds_full[k]]) ** 2
                        p = nk_full[k] / Nk_full[k]
                        mu_avg = p * r_full[k] + (1 - p) * mu_full
                        v_full[k] = (
                            p * v_full[k]
                            + (1 - p) * v_all_full
                            + p * (r_full[k] - mu_avg) ** 2
                            + (1 - p) * (mu_full - mu_avg) ** 2
                        )
                        eta_full[k] = np.sqrt(
                            (2 * np.log(np.log(nk_full[k]) + 1)
                             + np.log(4 * K / self.delta))
                            / nk_full[k] / 2
                        )
                        eps_k_full[k] = (
                            2 * eta_full[k] ** 2 / 3
                            + 2 * np.sqrt(v_full[k]) * eta_full[k]
                        )
                    r_hat_final = float(density_full @ r_full)
                    eps_final = max(float(density_full @ eps_k_full), eps_min)

                    t_list.append(n_labeled)
                    r_hat_list.append(r_hat_final)
                    lower_list.append(r_hat_final - eps_final)
                    upper_list.append(r_hat_final + eps_final)
                    total_labels_list.append(n_labeled)
                round_times.append(time.perf_counter() - _round_t0)
                break

            # Re-partition at thresholds — faithful to reference
            if n_labeled >= old:
                K, clf, partition_labels = _partition_z(
                    embeddings, losses, partition_labels, labeled_mask,
                    delta=self.delta, rng=self.rng,
                )
                partition_labels[:] = clf.predict(embeddings)
                Nk = np.array([len(np.where(partition_labels == k)[0]) for k in range(K)])
                density = Nk / N
                # Re-partition threshold — faithful: old = max(int(old*1.1), old+100)
                old = max(int(old * 1.1), old + 100)

            # Compute per-stratum stats
            inds = [np.where(partition_labels[ind] == k)[0] for k in range(K)]
            nk = np.array([len(ind_) for ind_ in inds])
            r = np.zeros(K)
            v = np.zeros(K)
            eta = np.zeros(K)
            epsilon = np.zeros(K)

            z = losses[ind]
            mu = np.mean(z)
            v_all = np.std(z) ** 2  # ddof=0, faithful to reference

            Nk_current = np.array([len(np.where(partition_labels == k)[0]) for k in range(K)])

            for k in range(K):
                if len(inds[k]) == 0 or Nk_current[k] == 0:
                    # Guard: empty stratum → skip (r=0, v=0, epsilon=0)
                    continue
                r[k] = np.mean(z[inds[k]])
                v[k] = np.std(z[inds[k]]) ** 2  # ddof=0, faithful
                p = nk[k] / Nk_current[k]
                mu_avg = p * r[k] + (1 - p) * mu
                v[k] = (
                    p * v[k]
                    + (1 - p) * v_all
                    + p * (r[k] - mu_avg) ** 2
                    + (1 - p) * (mu - mu_avg) ** 2
                )
                eta[k] = np.sqrt(
                    (2 * np.log(np.log(nk[k]) + 1) + np.log(4 * K / self.delta))
                    / nk[k]
                    / 2
                )
                epsilon[k] = 2 * eta[k] ** 2 / 3 + 2 * np.sqrt(v[k]) * eta[k]

            r_hat = float(density @ r)
            eps = max(float(density @ epsilon), eps_min)

            # Record trajectory
            t_list.append(n_labeled)
            r_hat_list.append(r_hat)
            lower_list.append(r_hat - eps)
            upper_list.append(r_hat + eps)
            total_labels_list.append(n_labeled)

            # Stopping criterion — faithful: 2 * eps <= C_full.
            # paper_experiment: monitor_to_T_max keeps labeling to T_max
            # so we can read the full width-vs-label trajectory.
            if 2 * eps <= self.C_full and not self.monitor_to_T_max:
                round_times.append(time.perf_counter() - _round_t0)
                break
            if int(labeled_mask.sum()) >= self.T_max:
                round_times.append(time.perf_counter() - _round_t0)
                break

            # Adaptive allocation: add 1 label to stratum with highest density*diff
            diff = np.zeros(K)
            for k in range(K):
                if nk[k] > 0:
                    diff[k] = (eta[k] * 4 / 3 + 2 * np.sqrt(v[k])) * eta[k] / nk[k]

            idx_sorted = np.argsort(density * diff)[::-1]
            added = False
            for l in range(len(idx_sorted)):
                k = idx_sorted[l]
                candidates = np.where((partition_labels[remain] == k))[0]
                if candidates.size > 0:
                    # Add one label from this stratum
                    new_idx = remain[candidates[0]]
                    labeled_mask[new_idx] = True
                    added = True
                    break

            if not added:
                round_times.append(time.perf_counter() - _round_t0)
                break

            # Natural bottom of iteration — successful label addition.
            round_times.append(time.perf_counter() - _round_t0)

        lower_arr = np.array(lower_list)
        upper_arr = np.array(upper_list)

        return {
            "t": np.array(t_list),
            "R_hat": np.array(r_hat_list),
            "lower": lower_arr,
            "upper": upper_arr,
            "pop_lower": lower_arr,   # No separate population correction
            "pop_upper": upper_arr,
            "total_labels": np.array(total_labels_list),
            # Paper §6.5 item #5 (Task 9): per-iteration wall-clock so the
            # ``wallclock`` stage can plot Cer-Eval round timing alongside
            # SAVE/IID. Length matches ``t_list``; entries are non-negative
            # ``time.perf_counter()`` deltas.
            "round_times": np.asarray(round_times, dtype=np.float64),
        }
