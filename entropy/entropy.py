"""Core entropy computation functions for the NBS experiment.

All functions are pure (no I/O). Entropy values are in bits (log base 2).
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Hashable, Sequence

import numpy as np

try:
    import ndd as _ndd  # NSB entropy estimator

    _HAS_NDD = True
except ImportError:
    _HAS_NDD = False


def compute_ngram_counts(
    sequence: Sequence[Hashable], order: int
) -> dict:
    """Count overlapping n-grams in *sequence*.

    For order=1, keys are individual symbols.
    For order>1, keys are tuples of symbols.
    """
    if len(sequence) < order:
        return {}
    counts: Counter = Counter()
    for i in range(len(sequence) - order + 1):
        gram = sequence[i : i + order]
        key = gram[0] if order == 1 else tuple(gram)
        counts[key] += 1
    return dict(counts)


def compute_entropy_miller_madow(counts: dict[Hashable, int]) -> float:
    """Shannon entropy (bits) with Miller-Madow bias correction.

    H_corrected = H_plugin + (k - 1) / (2 * N)
    where k = number of bins with count > 0, N = total count.
    """
    N = sum(counts.values())
    if N == 0:
        return 0.0
    k = len(counts)
    h_plugin = 0.0
    for c in counts.values():
        if c > 0:
            p = c / N
            h_plugin -= p * math.log2(p)
    correction = (k - 1) / (2 * N)
    return h_plugin + correction


def compute_conditional_entropy(
    sequence: Sequence[Hashable], order: int
) -> float:
    """Conditional entropy H(X_n | context) in bits.

    H(X | context) = H(context, X) - H(context)
    where context length = order - 1.

    Uses Miller-Madow correction on both joint and context distributions.
    Result is floored at 0.0 to handle estimation noise.
    """
    if order < 2:
        raise ValueError("order must be >= 2 for conditional entropy")
    joint_counts = compute_ngram_counts(sequence, order)
    context_counts = compute_ngram_counts(sequence, order - 1)
    if not joint_counts or not context_counts:
        return 0.0
    h_joint = compute_entropy_miller_madow(joint_counts)
    h_context = compute_entropy_miller_madow(context_counts)
    return max(0.0, h_joint - h_context)


def check_sparsity(
    alphabet_size: int, order: int, sample_size: int
) -> bool:
    """Return True if the n-gram space is sparse relative to sample size.

    Sparse when alphabet_size^order > sample_size / 5.
    """
    return alphabet_size ** order > sample_size / 5


def shuffle_control(
    sequence: Sequence[Hashable], seed: int = 42
) -> list[Hashable]:
    """Return a random permutation of *sequence*, preserving unigram frequencies."""
    rng = np.random.RandomState(seed)
    arr = list(sequence)
    rng.shuffle(arr)
    return arr


def compute_structure_score(sequence: Sequence[Hashable]) -> float:
    """Structure score: 1 - (H₃ / H₀).

    Measures how much trigram context reduces uncertainty relative to
    the unigram baseline.  Returns a value in [0, 1].
    """
    h0 = compute_entropy_miller_madow(compute_ngram_counts(sequence, 1))
    if h0 == 0.0:
        return 0.0
    h3 = compute_conditional_entropy(sequence, order=3)
    return max(0.0, min(1.0, 1 - h3 / h0))


def compute_sequential_score(
    sequence: Sequence[Hashable], seed: int = 42
) -> float:
    """Sequential score: 1 - (H₃ / H₂_shuffled).

    Compares conditional entropy of the real sequence against a shuffled
    baseline that destroys sequential structure but preserves unigrams.
    """
    shuffled = shuffle_control(sequence, seed=seed)
    h2_shuf = compute_conditional_entropy(shuffled, order=2)
    if h2_shuf == 0.0:
        return 0.0
    h3 = compute_conditional_entropy(sequence, order=3)
    return max(0.0, min(1.0, 1 - h3 / h2_shuf))


def compute_structure_scores_bootstrap(
    sequence: Sequence[Hashable],
    n_bootstrap: int = 1000,
    subsample_frac: float = 0.8,
    seed: int = 42,
) -> dict[str, float]:
    """Bootstrap confidence intervals for structure and sequential scores.

    Uses contiguous block subsampling (random start, take subsample_frac*N
    consecutive symbols) to preserve sequential structure.

    Returns dict with keys:
        mean, ci_low, ci_high           — structure score
        seq_mean, seq_ci_low, seq_ci_high — sequential score
    """
    rng = np.random.RandomState(seed)
    n = len(sequence)
    block_size = max(3, int(n * subsample_frac))  # need at least 3 for trigrams

    struct_scores: list[float] = []
    seq_scores: list[float] = []

    for _ in range(n_bootstrap):
        start = rng.randint(0, n - block_size + 1)
        sub = sequence[start : start + block_size]
        struct_scores.append(compute_structure_score(sub))
        seq_scores.append(compute_sequential_score(sub, seed=seed))

    struct_arr = np.array(struct_scores)
    seq_arr = np.array(seq_scores)

    return {
        "mean": float(np.mean(struct_arr)),
        "ci_low": float(np.percentile(struct_arr, 2.5)),
        "ci_high": float(np.percentile(struct_arr, 97.5)),
        "seq_mean": float(np.mean(seq_arr)),
        "seq_ci_low": float(np.percentile(seq_arr, 2.5)),
        "seq_ci_high": float(np.percentile(seq_arr, 97.5)),
    }


def _entropy_for_mi(counts: dict[Hashable, int], use_nsb: bool) -> float:
    """Entropy helper: use NSB (ndd) if available and requested, else Miller-Madow."""
    if use_nsb and _HAS_NDD:
        counts_arr = np.array(list(counts.values()), dtype=int)
        # ndd.entropy returns in nats; convert to bits
        return float(_ndd.entropy(counts_arr)) / math.log(2)
    return compute_entropy_miller_madow(counts)


def compute_mutual_information(
    sequence: Sequence[Hashable],
    lag: int,
    use_nsb: bool = True,
) -> float:
    """Mutual information I(X_t; X_{t+lag}) in bits.

    I(X;Y) = H(X) + H(Y) - H(X,Y)

    Falls back to Miller-Madow if ndd is unavailable.
    Result is floored at 0.0.
    """
    n = len(sequence)
    if lag >= n:
        return 0.0

    x_seq = sequence[: n - lag]
    y_seq = sequence[lag:]

    x_counts = Counter(x_seq)
    y_counts = Counter(y_seq)
    joint_counts: Counter = Counter()
    for xi, yi in zip(x_seq, y_seq):
        joint_counts[(xi, yi)] += 1

    h_x = _entropy_for_mi(dict(x_counts), use_nsb)
    h_y = _entropy_for_mi(dict(y_counts), use_nsb)
    h_joint = _entropy_for_mi(dict(joint_counts), use_nsb)

    return max(0.0, h_x + h_y - h_joint)


def mi_decay_profile(
    sequence: Sequence[Hashable],
    lags: list[int] | None = None,
    truncate_at_sparsity: bool = True,
) -> dict[int, float]:
    """Compute MI at multiple lags, optionally truncating at sparsity.

    Default lags: [1, 2, 5, 10, 50, 100, 500].
    Truncates where alphabet_size² > n_pairs / 5.
    """
    if lags is None:
        lags = [1, 2, 5, 10, 50, 100, 500]

    alphabet_size = len(set(sequence))
    joint_states = alphabet_size ** 2
    result: dict[int, float] = {}

    for lag in lags:
        n_pairs = len(sequence) - lag
        if n_pairs <= 0:
            continue
        if truncate_at_sparsity and joint_states > n_pairs / 5:
            continue
        result[lag] = compute_mutual_information(sequence, lag)

    return result


def compute_ngram_counts_pooled(
    sequences: Sequence[Sequence[Hashable]], order: int
) -> dict:
    """Pool n-gram counts across multiple sequences without cross-boundary concatenation.

    Each sequence is counted independently and the counts are summed.
    Sequences shorter than *order* are silently skipped.
    """
    pooled: Counter = Counter()
    for seq in sequences:
        counts = compute_ngram_counts(seq, order)
        pooled.update(counts)
    return dict(pooled)


def compute_conditional_entropy_pooled(
    sequences: Sequence[Sequence[Hashable]], order: int
) -> float:
    """Conditional entropy from pooled n-gram counts (no cross-boundary concatenation).

    H(X | context) = H_joint_pooled - H_context_pooled
    """
    if order < 2:
        raise ValueError("order must be >= 2 for conditional entropy")
    joint_counts = compute_ngram_counts_pooled(sequences, order)
    context_counts = compute_ngram_counts_pooled(sequences, order - 1)
    if not joint_counts or not context_counts:
        return 0.0
    h_joint = compute_entropy_miller_madow(joint_counts)
    h_context = compute_entropy_miller_madow(context_counts)
    return max(0.0, h_joint - h_context)
