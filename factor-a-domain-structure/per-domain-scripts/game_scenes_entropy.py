"""Game engine scene description Shannon entropy analysis.

Reads Godot .tscn (and .tres) files from godot-demo-projects
and runs the NBS entropy pipeline. Treats all scene text as a
single character stream (like Python code or G-code).

Output: 1-research/nbs-survey/results/game_scenes_entropy.json
"""

import json
import math
import sys
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent  # 1-research/
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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
DATA_DIR = Path(__file__).parent / "data" / "game_scenes"
RESULTS_FILE = Path(__file__).parent / "results" / "game_scenes_entropy.json"
MI_LAGS = [1, 2, 5, 10, 50, 100, 500]
TARGET_CHARS = 1_000_000

RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)


def collect_scene_files(data_dir: Path) -> list[Path]:
    """Find all .tscn and .tres files under data_dir."""
    files = []
    for ext in ("*.tscn", "*.tres"):
        files.extend(data_dir.rglob(ext))
    files.sort()
    print(f"Found {len(files):,} scene files under {data_dir}")
    return files


def load_scenes(files: list[Path], target: int = TARGET_CHARS) -> tuple[str, list[str]]:
    """Concatenate scene file contents into one stream up to target chars.

    Returns (full_text, list_of_source_labels).
    """
    parts = []
    sources = []
    total = 0
    skipped = 0

    for path in files:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            skipped += 1
            continue

        if total + len(text) > target:
            remaining = target - total
            parts.append(text[:remaining])
            sources.append(str(path.relative_to(path.parents[3])))
            total += remaining
            break
        else:
            parts.append(text)
            sources.append(str(path.relative_to(path.parents[3])))
            total += len(text)

    print(f"Loaded {len(parts):,} files  ({total:,} chars)  [{skipped} skipped]")
    return "".join(parts), sources


def report_alphabet(sequence: str) -> dict:
    """Count and report character frequencies."""
    counts: dict[str, int] = {}
    for ch in sequence:
        counts[ch] = counts.get(ch, 0) + 1

    sorted_counts = sorted(counts.items(), key=lambda x: -x[1])
    print("\nTop 20 characters by frequency:")
    for ch, n in sorted_counts[:20]:
        display = repr(ch)
        print(f"  {display:10s}: {n:8,}  ({100 * n / len(sequence):5.2f}%)")
    return counts


def main():
    print("=" * 60)
    print("Game Engine Scene Entropy Analysis (Godot .tscn / .tres)")
    print("=" * 60)

    # ── Collect files ─────────────────────────────────────────────────────────
    files = collect_scene_files(DATA_DIR)
    if not files:
        print("ERROR: No scene files found. Run git clone first.")
        sys.exit(1)

    # ── Load text ─────────────────────────────────────────────────────────────
    text, sources = load_scenes(files)
    seq = list(text)  # character list

    n_chars = len(seq)
    alphabet_counts = report_alphabet(text)
    alphabet = sorted(alphabet_counts.keys())
    alphabet_size = len(alphabet)

    print(f"\nAlphabet size: {alphabet_size} symbols")
    print(f"Sequence length: {n_chars:,} chars")
    print(f"Source files loaded: {len(sources):,}")

    # ── H0 (unigram entropy) ─────────────────────────────────────────────────
    print("\n--- Computing H0 (unigram entropy) ---")
    unigram_counts = compute_ngram_counts(seq, 1)
    h0 = compute_entropy_miller_madow(unigram_counts)
    max_h0 = math.log2(alphabet_size) if alphabet_size > 1 else 0.0
    print(f"H0 = {h0:.4f} bits  (max possible = {max_h0:.4f} bits for {alphabet_size} symbols)")

    # ── H2 (bigram conditional entropy) ──────────────────────────────────────
    print("\n--- Computing H2 (bigram conditional entropy) ---")
    h2 = compute_conditional_entropy(seq, order=2)
    print(f"H2 = {h2:.4f} bits")

    # ── H3 (trigram conditional entropy) ─────────────────────────────────────
    print("\n--- Computing H3 (trigram conditional entropy) ---")
    h3 = compute_conditional_entropy(seq, order=3)
    print(f"H3 = {h3:.4f} bits")

    # ── Structure score ───────────────────────────────────────────────────────
    print("\n--- Computing structure score = 1 - H3/H0 ---")
    structure_score = compute_structure_score(seq)
    print(f"Structure score = {structure_score:.4f}")

    # ── Sequential score ──────────────────────────────────────────────────────
    print("\n--- Computing sequential score = 1 - H3/H2_shuffled ---")
    sequential_score = compute_sequential_score(seq)
    print(f"Sequential score = {sequential_score:.4f}")

    # ── Shuffled control ──────────────────────────────────────────────────────
    print("\n--- Computing shuffled control ---")
    shuffled = shuffle_control(seq, seed=42)
    h3_shuffled = compute_conditional_entropy(shuffled, order=3)
    h0_shuffled = compute_entropy_miller_madow(compute_ngram_counts(shuffled, 1))
    structure_score_shuffled = max(0.0, 1 - h3_shuffled / h0_shuffled) if h0_shuffled > 0 else 0.0
    print(f"Shuffled H0 = {h0_shuffled:.4f} bits")
    print(f"Shuffled H3 = {h3_shuffled:.4f} bits")
    print(f"Shuffled structure score = {structure_score_shuffled:.6f}")

    # ── MI profile ────────────────────────────────────────────────────────────
    print("\n--- Computing MI decay profile ---")
    mi_profile: dict[int, float] = {}
    for lag in MI_LAGS:
        mi = compute_mutual_information(seq, lag)
        mi_profile[lag] = mi
        print(f"  MI(lag={lag:4d}) = {mi:.4f} bits")

    # ── Metadata ──────────────────────────────────────────────────────────────
    data_source = (
        "godotengine/godot-demo-projects (GitHub, shallow clone --depth 1, "
        "https://github.com/godotengine/godot-demo-projects). "
        f"All .tscn and .tres files concatenated. "
        f"{len(sources):,} files used to reach {n_chars:,} chars."
    )
    encoding_notes = (
        "Godot Engine scene format: INI-like text with [section] headers, "
        "key=value pairs, typed literals (Vector2, Color, PackedVector2Array, etc.), "
        "and resource references (ExtResource, SubResource). "
        "Files are UTF-8 text — highly structured, rule-governed, repetitive "
        "(every node has type, parent, transform, and resource refs in fixed patterns). "
        "Treated as raw character stream — character-level entropy measures the "
        "notation's compressibility."
    )

    # ── Assemble results ──────────────────────────────────────────────────────
    results = {
        "domain": "Game engine scene descriptions (Godot .tscn / .tres)",
        "data_source": data_source,
        "encoding_notes": encoding_notes,
        "n_chars": n_chars,
        "n_files": len(sources),
        "primary_encoding": "UTF-8 character stream (raw scene text)",
        "h0": round(h0, 6),
        "h2": round(h2, 6),
        "h3": round(h3, 6),
        "structure_score": round(structure_score, 6),
        "sequential_score": round(sequential_score, 6),
        "shuffled_control": {
            "h0": round(h0_shuffled, 6),
            "h3": round(h3_shuffled, 6),
            "structure_score": round(structure_score_shuffled, 6),
            "note": (
                "Shuffling destroys all sequential structure; structure_score "
                "collapses to ~0, confirming the signal is genuine sequential constraint."
            ),
        },
        "mi_profile": {str(k): round(v, 6) for k, v in mi_profile.items()},
        "alphabet_size": alphabet_size,
        "top_chars_by_freq": {
            k: int(v)
            for k, v in sorted(alphabet_counts.items(), key=lambda x: -x[1])[:30]
        },
    }

    # ── Save ──────────────────────────────────────────────────────────────────
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {RESULTS_FILE}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(
        f"GAME SCENES: structure_score={structure_score:.3f}, "
        f"sequential_score={sequential_score:.3f}"
    )
    print(
        f"  H0={h0:.2f}, H2={h2:.2f}, H3={h3:.2f}  "
        f"[alphabet={alphabet_size} chars, n={n_chars:,}]"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()
