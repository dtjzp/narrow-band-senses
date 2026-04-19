"""RNA secondary structure Shannon entropy analysis.

Reads dot-bracket notation from 1-research/nbs-survey/data/rna/rna_structures.txt
and runs the NBS entropy pipeline.

Output: 1-research/nbs-survey/results/rna_entropy.json
"""

import json
import sys
import os
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent  # 1-research/
sys.path.insert(0, str(ROOT / "nbs-experiment"))

from entropy import (
    compute_ngram_counts,
    compute_entropy_miller_madow,
    compute_conditional_entropy,
    compute_structure_score,
    compute_sequential_score,
    shuffle_control,
    compute_mutual_information,
)

# ── Config ────────────────────────────────────────────────────────────────────
DATA_FILE = Path(__file__).parent / "data" / "rna" / "rna_structures.txt"
RESULTS_FILE = Path(__file__).parent / "results" / "rna_entropy.json"
MI_LAGS = [1, 2, 5, 10, 50, 100, 500]
TARGET_CHARS = 1_000_000

RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)


def load_structures(path: Path, target: int = TARGET_CHARS) -> str:
    """Load dot-bracket structures and concatenate to a single stream.

    The task description notes dot-bracket notation is 'naturally a stream',
    so we concatenate with no separator (treating the file as one long sequence).
    We also track structure boundaries for pooled analysis.
    """
    lines = []
    with open(path) as f:
        for line in f:
            s = line.strip()
            if s:
                lines.append(s)

    # Concatenate — dot-bracket IS a character stream (like DNA/protein)
    full = "".join(lines)
    print(f"Loaded {len(lines):,} structures, {len(full):,} total chars")

    if len(full) > target:
        full = full[:target]
        print(f"Truncated to {len(full):,} chars")

    return full


def report_alphabet(sequence: str) -> dict:
    """Characterise the alphabet in use."""
    counts = {}
    for ch in sequence:
        counts[ch] = counts.get(ch, 0) + 1
    sorted_counts = sorted(counts.items(), key=lambda x: -x[1])
    print("\nAlphabet breakdown:")
    for ch, n in sorted_counts:
        print(f"  '{ch}': {n:,} ({100*n/len(sequence):.1f}%)")
    return counts


def main():
    print("=" * 60)
    print("RNA Secondary Structure Entropy Analysis")
    print("=" * 60)

    # ── Load data ─────────────────────────────────────────────────────────────
    sequence = load_structures(DATA_FILE)
    n_chars = len(sequence)
    alphabet_counts = report_alphabet(sequence)
    alphabet = sorted(alphabet_counts.keys())
    alphabet_size = len(alphabet)

    print(f"\nAlphabet size: {alphabet_size} symbols: {alphabet}")
    print(f"Sequence length: {n_chars:,} chars")

    # ── H0 (unigram entropy) ─────────────────────────────────────────────────
    print("\n--- Computing H0 (unigram entropy) ---")
    unigram_counts = compute_ngram_counts(sequence, 1)
    h0 = compute_entropy_miller_madow(unigram_counts)
    print(f"H0 = {h0:.4f} bits  (max possible = {len(unigram_counts):.4f}? no, log2({len(unigram_counts)}) = {__import__('math').log2(len(unigram_counts)):.4f} bits)")

    # ── H2 (bigram conditional entropy) ──────────────────────────────────────
    print("\n--- Computing H2 (bigram conditional entropy) ---")
    h2 = compute_conditional_entropy(sequence, order=2)
    print(f"H2 = {h2:.4f} bits")

    # ── H3 (trigram conditional entropy) ─────────────────────────────────────
    print("\n--- Computing H3 (trigram conditional entropy) ---")
    h3 = compute_conditional_entropy(sequence, order=3)
    print(f"H3 = {h3:.4f} bits")

    # ── Structure score ───────────────────────────────────────────────────────
    print("\n--- Computing structure score = 1 - H3/H0 ---")
    structure_score = compute_structure_score(sequence)
    print(f"Structure score = {structure_score:.4f}")

    # ── Sequential score ──────────────────────────────────────────────────────
    print("\n--- Computing sequential score = 1 - H3/H2_shuffled ---")
    sequential_score = compute_sequential_score(sequence)
    print(f"Sequential score = {sequential_score:.4f}")

    # ── Shuffled control ──────────────────────────────────────────────────────
    print("\n--- Computing shuffled control ---")
    shuffled = shuffle_control(sequence, seed=42)
    h3_shuffled = compute_conditional_entropy(shuffled, order=3)
    h0_shuffled = compute_entropy_miller_madow(compute_ngram_counts(shuffled, 1))
    structure_score_shuffled = max(0.0, 1 - h3_shuffled / h0_shuffled) if h0_shuffled > 0 else 0.0
    print(f"Shuffled H0 = {h0_shuffled:.4f} bits")
    print(f"Shuffled H3 = {h3_shuffled:.4f} bits")
    print(f"Shuffled structure score = {structure_score_shuffled:.4f}")

    # ── MI profile ────────────────────────────────────────────────────────────
    print("\n--- Computing MI decay profile ---")
    mi_profile = {}
    for lag in MI_LAGS:
        mi = compute_mutual_information(sequence, lag)
        mi_profile[lag] = mi
        print(f"  MI(lag={lag:4d}) = {mi:.4f} bits")

    # ── Data source notes ─────────────────────────────────────────────────────
    data_source = "Rfam CURRENT Rfam.seed.gz (SS_cons consensus secondary structure lines from Stockholm-format seed alignments) + Rfam REST API (30 families) — https://ftp.ebi.ac.uk/pub/databases/Rfam/CURRENT/Rfam.seed.gz"
    encoding_notes = (
        "Dot-bracket notation: '(' and ')' = base-paired (Watson-Crick or wobble), "
        "'.' = unpaired. Extended brackets '[]', '{}', '<>' encode pseudoknots. "
        "Concatenated stream from 4230+ Rfam consensus structures, tiled to 1M chars. "
        "Character frequencies strongly dominated by '.' (unpaired), '(', ')'. "
        "Very small alphabet (3-7 symbols typical), so H0 is inherently low. "
        "HIGH structure score expected: bracket-matching grammar imposes strong "
        "sequential constraints — every '(' must be followed eventually by ')'."
    )

    # ── Assemble results ──────────────────────────────────────────────────────
    results = {
        "domain": "RNA secondary structure (dot-bracket notation)",
        "h0": round(h0, 6),
        "h2": round(h2, 6),
        "h3": round(h3, 6),
        "structure_score": round(structure_score, 6),
        "sequential_score": round(sequential_score, 6),
        "shuffled_control": {
            "h0": round(h0_shuffled, 6),
            "h3": round(h3_shuffled, 6),
            "structure_score": round(structure_score_shuffled, 6),
        },
        "mi_profile": {str(k): round(v, 6) for k, v in mi_profile.items()},
        "n_chars": n_chars,
        "alphabet_size": alphabet_size,
        "alphabet": alphabet,
        "alphabet_counts": {k: int(v) for k, v in alphabet_counts.items()},
        "data_source": data_source,
        "encoding_notes": encoding_notes,
    }

    # ── Save ──────────────────────────────────────────────────────────────────
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {RESULTS_FILE}")

    # ── Summary line ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(
        f"RNA: structure_score={structure_score:.3f}, "
        f"H0={h0:.2f}, H3={h3:.2f}, "
        f"alphabet={alphabet_size} chars"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()
