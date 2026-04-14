"""WiFi CSI Shannon entropy analysis for the NBS survey.

Dataset: GeoTecINIT/stability-csi-har (GitHub)
"Temporal Stability on Human Activity Recognition based on Wi-Fi CSI"
Licensed for research use.

Each file: shape (n_windows, n_subcarriers=56, n_timesteps=50)
Activities: sit, stand, walk, etc. (6 activity types)
Datasets: D1 (20 recordings), D2, D3, D4 (4 recordings each) = 32 files

Encoding strategy (canonical for NBS survey):
  - Per-subcarrier: treat each of the 56 subcarriers as a separate time series
  - Each subcarrier amplitude sequence: 20 quantile bins -> chars A-T
  - Sequences concatenated (pooled counting, no cross-subcarrier joins)
  - This produces ~2.27M chars total (well above 1M target)
  - Truncate to 1M chars for consistency with other survey domains

Output: 1-research/nbs-survey/results/wifi_csi_entropy.json
Processed text: 1-research/nbs-survey/data/processed/wifi_csi_1M.txt
"""

import json
import sys
import os
import math
import numpy as np
from pathlib import Path
from collections import Counter

# Force UTF-8 output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent          # 1-research/
SCRIPT_DIR = Path(__file__).parent          # 1-research/nbs-survey/
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
DATA_DIR     = SCRIPT_DIR / "data" / "wifi_csi"
RESULTS_FILE = SCRIPT_DIR / "results" / "wifi_csi_entropy.json"
PROCESSED_FILE = SCRIPT_DIR / "data" / "processed" / "wifi_csi_1M.txt"

N_BINS   = 20
N_TARGET = 1_000_000      # target character count
MI_LAGS  = [1, 2, 5, 10, 50, 100, 500]

RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
PROCESSED_FILE.parent.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

CHARS = "ABCDEFGHIJKLMNOPQRST"   # 20 symbols


# ── Data loading ──────────────────────────────────────────────────────────────

def load_csi_files(data_dir: Path):
    """Load all D*_e*-x.npy files.

    Returns list of (n_windows, n_subcarriers, n_timesteps) arrays.
    """
    files = sorted(
        f for f in data_dir.iterdir()
        if f.suffix == ".npy" and f.stem.startswith("D")
    )
    arrays = []
    for fpath in files:
        try:
            arr = np.load(str(fpath))
            arrays.append(arr)
        except Exception as e:
            print(f"  Warning: could not load {fpath.name}: {e}")
    print(f"Loaded {len(arrays)} CSI recording files")
    return arrays


# ── Preprocessing ─────────────────────────────────────────────────────────────

def build_char_stream(arrays: list, n_target: int = N_TARGET) -> tuple:
    """Convert list of CSI arrays to a single character stream.

    Strategy:
    - For each recording: iterate over windows, then subcarriers
    - Each subcarrier's 50-timestep window -> a short char sequence
    - All sequences concatenated (pooled counting)
    - Truncate to n_target if needed

    Returns (stream: str, n_sequences: int, alphabet_size: int)
    """
    # First pass: collect all amplitude values for global percentile bins
    # (ensures consistent encoding across recordings and subcarriers)
    all_amplitudes = []
    for arr in arrays:
        all_amplitudes.append(arr.reshape(-1))
    all_amp = np.concatenate(all_amplitudes)

    # Global quantile edges
    global_edges = np.percentile(all_amp, np.linspace(0, 100, N_BINS + 1))
    global_edges = np.unique(global_edges)
    actual_bins = len(global_edges) - 1
    print(f"  Global amplitude range: {all_amp.min():.2f} – {all_amp.max():.2f}")
    print(f"  Global quantile bins: {actual_bins} (of {N_BINS} requested)")

    # Second pass: encode per-subcarrier sequences
    segments = []
    total_chars = 0
    n_sequences = 0

    for arr in arrays:
        n_windows, n_sub, n_time = arr.shape
        for s in range(n_sub):
            # Concatenate all windows for this subcarrier in this recording
            sub_amp = arr[:, s, :].reshape(-1)   # (n_windows * n_time,)
            # Encode using global edges
            bin_idx = np.digitize(sub_amp, global_edges[1:-1], right=True)
            bin_idx = np.clip(bin_idx, 0, actual_bins - 1)
            scale = (N_BINS - 1) / max(actual_bins - 1, 1)
            mapped = np.clip((bin_idx * scale).astype(int), 0, N_BINS - 1)
            seg = "".join(CHARS[i] for i in mapped)
            segments.append(seg)
            total_chars += len(seg)
            n_sequences += 1

        if total_chars >= n_target:
            break

    stream = "".join(segments)
    stream = stream[:n_target]  # truncate to target

    # Count actual alphabet used
    alphabet = set(stream)
    print(f"  Total sequences encoded: {n_sequences}")
    print(f"  Stream length: {len(stream)} chars")
    print(f"  Alphabet size: {len(alphabet)}")

    return stream, n_sequences, len(alphabet)


# ── Per-subcarrier structure (canonical measure) ───────────────────────────────

def compute_per_subcarrier_structure(arrays: list) -> dict:
    """Compute per-subcarrier structure scores (like bioreactor per-channel approach).

    For each subcarrier index (0-55), concatenate all windows' timeseries across
    all recordings and compute structure score. Returns mean and per-sub values.
    """
    # Pool all data along window axis
    all_arrs = np.concatenate(arrays, axis=0)  # (total_windows, n_sub, n_time)
    n_windows, n_sub, n_time = all_arrs.shape

    # Global percentile edges for consistent encoding
    all_amp = all_arrs.reshape(-1)
    global_edges = np.percentile(all_amp, np.linspace(0, 100, N_BINS + 1))
    global_edges = np.unique(global_edges)
    actual_bins = len(global_edges) - 1

    scores = []
    for s in range(n_sub):
        amp = all_arrs[:, s, :].reshape(-1)   # (n_windows * n_time,)
        # Encode
        bin_idx = np.digitize(amp, global_edges[1:-1], right=True)
        bin_idx = np.clip(bin_idx, 0, actual_bins - 1)
        scale = (N_BINS - 1) / max(actual_bins - 1, 1)
        mapped = np.clip((bin_idx * scale).astype(int), 0, N_BINS - 1)
        seg = "".join(CHARS[i] for i in mapped)

        if len(seg) < 100:
            scores.append(0.0)
            continue

        try:
            sc = compute_structure_score(seg)
            scores.append(sc)
        except Exception:
            scores.append(0.0)

    mean_sc = float(np.mean(scores))
    print(f"  Per-subcarrier structure scores: min={min(scores):.3f} max={max(scores):.3f} mean={mean_sc:.3f}")
    return {
        "mean": mean_sc,
        "min": float(min(scores)),
        "max": float(max(scores)),
        "per_subcarrier": [round(s, 4) for s in scores],
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("WiFi CSI Entropy Analysis — NBS Survey")
    print("=" * 60)

    # 1. Load data
    print("\n[1] Loading CSI data...")
    arrays = load_csi_files(DATA_DIR)
    if not arrays:
        raise RuntimeError(f"No CSI data found in {DATA_DIR}. Run download step first.")

    print(f"  Sample array shape: {arrays[0].shape}")
    total_windows = sum(a.shape[0] for a in arrays)
    total_subcarriers = arrays[0].shape[1]
    total_time = arrays[0].shape[2]
    print(f"  Total windows: {total_windows}, subcarriers: {total_subcarriers}, timesteps/window: {total_time}")

    # 2. Build character stream
    print("\n[2] Building character stream (per-subcarrier, global quantile bins)...")
    stream, n_sequences, alphabet_size = build_char_stream(arrays)

    # 3. Save processed text
    print(f"\n[3] Saving processed text to {PROCESSED_FILE}...")
    with open(PROCESSED_FILE, "w", encoding="utf-8") as f:
        f.write(stream)
    print(f"  Saved {len(stream)} chars")

    # 4. N-gram entropy (using sequence API, not pre-computed dicts)
    print("\n[4] Computing n-gram entropy...")
    uni_counts = compute_ngram_counts(stream, 1)
    h0_unigram = compute_entropy_miller_madow(uni_counts)
    h0_max = math.log2(alphabet_size) if alphabet_size > 1 else 0.0
    h2 = compute_conditional_entropy(stream, order=2)
    h3 = compute_conditional_entropy(stream, order=3)

    structure_score  = compute_structure_score(stream)
    sequential_score = compute_sequential_score(stream)

    print(f"  H0 (unigram MM): {h0_unigram:.4f} bits")
    print(f"  H0 (log2({alphabet_size})): {h0_max:.4f} bits")
    print(f"  H(X|bigram):     {h2:.4f} bits")
    print(f"  H(X|trigram):    {h3:.4f} bits")
    print(f"  Structure score:  {structure_score:.4f}")
    print(f"  Sequential score: {sequential_score:.4f}")

    # 5. Shuffled control
    print("\n[5] Computing shuffled control...")
    shuffled = shuffle_control(list(stream))
    shuffled_struct = compute_structure_score(shuffled)
    print(f"  Shuffled structure score = {shuffled_struct:.4f}")

    # 6. Mutual information profile
    print("\n[6] Computing MI at lags:", MI_LAGS)
    mi_profile = {}
    for lag in MI_LAGS:
        mi = compute_mutual_information(stream, lag)
        mi_profile[str(lag)] = round(mi, 6)
        print(f"  MI(lag={lag:4d}) = {mi:.4f} bits")

    # 7. Per-subcarrier structure (canonical measure)
    print("\n[7] Computing per-subcarrier structure scores...")
    per_sub = compute_per_subcarrier_structure(arrays)

    # 8. Assemble results (exact keys as specified)
    results = {
        "domain": "wifi_csi",
        "h0": round(h0_unigram, 6),
        "h2": round(h2, 6),
        "h3": round(h3, 6),
        "structure_score": round(structure_score, 6),
        "sequential_score": round(sequential_score, 6),
        "shuffled_structure_score": round(shuffled_struct, 6),
        "per_subcarrier_structure": per_sub,
        "mi_profile": mi_profile,
        "n_chars": len(stream),
        "alphabet_size": alphabet_size,
        "data_source": (
            "GeoTecINIT/stability-csi-har (GitHub) — "
            "'Temporal Stability on Human Activity Recognition based on Wi-Fi CSI'. "
            "D1-D4 labelled datasets, 32 recording files across 4 environments. "
            "Shape per file: (n_windows, 56 subcarriers, 50 timesteps). "
            "6 activity types: sit, stand, walk, pick up, wave, jump."
        ),
        "encoding_notes": (
            f"Per-subcarrier encoding: each of {total_subcarriers} subcarriers treated as "
            "independent time series. Global quantile bins (20 equal-occupancy) computed "
            "across all windows and subcarriers combined. Windows concatenated within each "
            "recording+subcarrier pair; sequences then pooled (no cross-subcarrier joins). "
            f"Stream truncated to {N_TARGET:,} chars. "
            "Canonical measure is per-subcarrier mean structure score (analogous to "
            "bioreactor per-channel). Pooled stream score also reported."
        ),
    }

    # 9. Save results
    print(f"\n[8] Saving results to {RESULTS_FILE}...")
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("  Done.")

    # 10. Summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"  Domain:              wifi_csi")
    print(f"  Characters:          {len(stream):,}")
    print(f"  Alphabet size:       {alphabet_size}")
    print(f"  H0 (unigram):        {h0_unigram:.4f} bits  (max {h0_max:.4f})")
    print(f"  H(X|bigram):         {h2:.4f} bits")
    print(f"  H(X|trigram):        {h3:.4f} bits")
    print(f"  Structure score:     {structure_score:.4f}  <- pooled stream")
    print(f"  Per-sub mean score:  {per_sub['mean']:.4f}  <- canonical")
    print(f"  Shuffled control:    {shuffled_struct:.4f}  (should be ~0)")
    print(f"  MI(lag=1):           {mi_profile['1']:.4f} bits")
    print(f"  MI(lag=10):          {mi_profile['10']:.4f} bits")
    print(f"  MI(lag=100):         {mi_profile['100']:.4f} bits")
    print(f"  MI(lag=500):         {mi_profile['500']:.4f} bits")

    canonical = per_sub["mean"]
    if canonical >= 0.7:
        verdict = "VERY HIGH (above tidal/ATC)"
    elif canonical >= 0.5:
        verdict = "HIGH (above expected range)"
    elif canonical >= 0.3:
        verdict = "MODERATE-HIGH (within expected range 0.3-0.6)"
    elif canonical >= 0.1:
        verdict = "LOW-MODERATE (below expected)"
    else:
        verdict = "NEAR-ZERO (unexpected)"

    print(f"\n  Verdict: {verdict}")
    print(f"  Expected: MODERATE-HIGH (0.3-0.6)")
    print("=" * 60)

    return results


if __name__ == "__main__":
    main()
