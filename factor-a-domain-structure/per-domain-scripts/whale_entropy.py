"""Humpback whale song Shannon entropy analysis.

Downloads: NOAA + MBARI humpback whale recordings (Internet Archive).
Audio -> mel spectrogram (80-3500 Hz, 20ms frames) -> dominant frequency bin
-> discretised to 20 bins -> character sequence -> NBS entropy pipeline.

Output: 1-research/nbs-survey/results/whale_entropy.json
"""

import json
import sys
import os
import warnings
from pathlib import Path
from collections import Counter

import numpy as np
import librosa

warnings.filterwarnings("ignore")

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
DATA_DIR = Path(__file__).parent / "data" / "whale"
RESULTS_FILE = Path(__file__).parent / "results" / "whale_entropy.json"
RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)

# Audio processing parameters
SR = 8000           # Downsample to 8kHz — sufficient for whale song (80-3500 Hz)
N_MELS = 32         # Mel bins
HOP_LENGTH = 160    # 20ms frames at 8kHz
FMIN = 80           # Humpback whale fundamental ~80 Hz
FMAX = 3500         # Upper limit of humpback song
N_BINS = 20         # Discretisation bins (as per task spec)
SILENCE_PERCENTILE = 15   # Bottom 15% of frames by RMS = silence

MI_LAGS = [1, 2, 5, 10, 50, 100, 500]
TARGET_CHARS = 1_000_000

# ── Audio files ───────────────────────────────────────────────────────────────
AUDIO_FILES = [
    ("NOAA_whale_song.mp3", "NOAA humpback whale song (13min, 44.1kHz, Public Domain)"),
    ("MBARI_humpback_whale.mp3", "MBARI humpback whale (120min, 16kHz, CC0)"),
    ("MBARI_humpback_2016.mp3", "MBARI humpback whale 2016 session (80min, 16kHz, CC0)"),
    ("MBARI_humpback_2017a.mp3", "MBARI humpback whale 2017 session-1 (155min, 16kHz, CC0)"),
]

DATA_SOURCE = (
    "NOAA/PMEL humpback whale song via Internet Archive (WhaleSong_928, Public Domain — "
    "https://archive.org/details/WhaleSong_928); "
    "Monterey Bay Aquarium Research Institute (MBARI) humpback whale recordings via "
    "Internet Archive (humpbackwhalesongs, CC0 — "
    "https://archive.org/details/humpbackwhalesongs). "
    "4 recordings: 13min (NOAA) + 120min + 80min + 155min (MBARI) = ~368min total. "
    "Downloaded 2026-04-11."
)

ENCODING_NOTES = (
    f"Audio resampled to {SR}Hz mono. Mel spectrogram: {N_MELS} mel bands, "
    f"{HOP_LENGTH/SR*1000:.0f}ms hop (= {HOP_LENGTH/SR:.3f}s frame resolution), "
    f"frequency range {FMIN}-{FMAX}Hz (humpback vocal range). "
    f"Per-frame energy (RMS): frames below {SILENCE_PERCENTILE}th percentile discarded as silence. "
    f"Active frames: dominant mel bin (argmax across {N_MELS} bands) extracted. "
    f"Dominant bin (0-{N_MELS-1}) linearly mapped to {N_BINS} discrete bins, "
    f"each encoded as a character A-T. Each character represents ~{HOP_LENGTH/SR*1000:.0f}ms "
    f"of audio (~{N_BINS} pitch regions from {FMIN}Hz to {FMAX}Hz). "
    f"HIGH structure score expected: humpback songs are hierarchically organised — "
    f"units -> phrases -> themes -> songs — with strong sequential repetition."
)


def load_audio_as_sequence(path: Path) -> tuple[list[int], float]:
    """Load audio file, compute mel spectrogram, return active-frame dominant bins.

    Returns:
        (dominant_bins, active_fraction)
        where dominant_bins is a list of ints in [0, N_BINS-1]
        and active_fraction is the proportion of frames retained.
    """
    fname = path.name
    print(f"  Loading {fname}...")
    y, sr = librosa.load(str(path), sr=SR, mono=True)
    duration = len(y) / sr
    print(f"    Duration: {duration:.0f}s ({duration/60:.1f}min), sr={sr}Hz")

    # Mel spectrogram
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=N_MELS, hop_length=HOP_LENGTH, fmin=FMIN, fmax=FMAX
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)  # shape (N_MELS, n_frames)
    n_frames = mel_db.shape[1]

    # Per-frame RMS energy for silence gating
    rms = librosa.feature.rms(y=y, hop_length=HOP_LENGTH)[0]
    rms_db = 20 * np.log10(rms + 1e-10)

    # Silence threshold
    threshold = np.percentile(rms_db, SILENCE_PERCENTILE)
    active_mask = rms_db > threshold

    # Align lengths (rms may differ by 1 frame from mel)
    min_len = min(n_frames, len(active_mask))
    active_mask = active_mask[:min_len]
    mel_db = mel_db[:, :min_len]

    # Dominant mel bin per frame, then keep active frames only
    dominant_bin = np.argmax(mel_db, axis=0)  # shape (n_frames,)
    active_dominant = dominant_bin[active_mask]

    # Discretise to N_BINS
    bins = (active_dominant * N_BINS / N_MELS).astype(int).clip(0, N_BINS - 1)
    active_frac = active_mask.sum() / min_len

    print(
        f"    Active frames: {active_mask.sum():,}/{min_len:,} ({100*active_frac:.1f}%), "
        f"bin range [{bins.min()}, {bins.max()}]"
    )
    return list(bins), float(active_frac)


def report_alphabet(sequence: list[int]) -> dict:
    """Characterise the alphabet distribution."""
    counts = Counter(sequence)
    total = len(sequence)
    alphabet = [chr(ord("A") + i) for i in range(N_BINS)]
    print("\nBin distribution (showing occupied bins):")
    for i in range(N_BINS):
        if i in counts:
            ch = alphabet[i]
            n = counts[i]
            print(f"  Bin {i:2d} ('{ch}'): {n:,} ({100*n/total:.1f}%)")
    return {i: counts.get(i, 0) for i in range(N_BINS)}


def main():
    print("=" * 60)
    print("Humpback Whale Song Entropy Analysis")
    print("=" * 60)

    # ── Load all audio files ──────────────────────────────────────────────────
    all_bins: list[int] = []
    file_info = []

    print("\nLoading audio files:")
    for fname, description in AUDIO_FILES:
        path = DATA_DIR / fname
        if not path.exists():
            print(f"  SKIPPING (not found): {fname}")
            continue
        bins, active_frac = load_audio_as_sequence(path)
        all_bins.extend(bins)
        file_info.append({
            "file": fname,
            "description": description,
            "n_frames": len(bins),
            "active_fraction": round(active_frac, 4),
        })
        print(f"    Cumulative sequence length: {len(all_bins):,} chars")

    if not all_bins:
        print("ERROR: No audio loaded. Check data directory.")
        sys.exit(1)

    # ── Truncate to TARGET_CHARS ──────────────────────────────────────────────
    if len(all_bins) > TARGET_CHARS:
        all_bins = all_bins[:TARGET_CHARS]
        print(f"\nTruncated to {len(all_bins):,} chars")

    # ── Encode as character sequence ──────────────────────────────────────────
    alphabet_chars = [chr(ord("A") + i) for i in range(N_BINS)]
    sequence = "".join(alphabet_chars[b] for b in all_bins)
    n_chars = len(sequence)

    # Count occupied bins
    bin_counts = report_alphabet(all_bins)
    occupied_bins = sum(1 for v in bin_counts.values() if v > 0)
    alphabet_used = [alphabet_chars[i] for i in range(N_BINS) if bin_counts.get(i, 0) > 0]

    print(f"\nSequence length: {n_chars:,} chars")
    print(f"Alphabet: {occupied_bins} bins occupied out of {N_BINS}")
    print(f"Alphabet chars: {''.join(alphabet_used)}")

    # ── H0 (unigram entropy) ──────────────────────────────────────────────────
    print("\n--- Computing H0 (unigram entropy) ---")
    unigram_counts = compute_ngram_counts(sequence, 1)
    h0 = compute_entropy_miller_madow(unigram_counts)
    h0_max = np.log2(occupied_bins)
    print(f"H0 = {h0:.4f} bits  (max for {occupied_bins} symbols = {h0_max:.4f} bits)")

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
    structure_score_shuffled = (
        max(0.0, 1 - h3_shuffled / h0_shuffled) if h0_shuffled > 0 else 0.0
    )
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

    # ── Assemble results ──────────────────────────────────────────────────────
    results = {
        "domain": "Humpback whale song (audio)",
        "data_source": DATA_SOURCE,
        "encoding_notes": ENCODING_NOTES,
        "audio_files": file_info,
        "audio_processing": {
            "sample_rate_hz": SR,
            "n_mels": N_MELS,
            "hop_length_samples": HOP_LENGTH,
            "frame_duration_ms": round(HOP_LENGTH / SR * 1000, 1),
            "fmin_hz": FMIN,
            "fmax_hz": FMAX,
            "n_bins": N_BINS,
            "silence_percentile_threshold": SILENCE_PERCENTILE,
            "encoding": "dominant mel bin (argmax) per active frame, linearly mapped to N_BINS",
        },
        "n_chars": n_chars,
        "alphabet_size": occupied_bins,
        "alphabet": alphabet_used,
        "bin_counts": {alphabet_chars[i]: bin_counts[i] for i in range(N_BINS) if bin_counts.get(i, 0) > 0},
        "h0": round(h0, 6),
        "h0_max_possible": round(h0_max, 6),
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
    }

    # ── Save ──────────────────────────────────────────────────────────────────
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {RESULTS_FILE}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("HUMPBACK WHALE SONG ENTROPY SUMMARY")
    print("=" * 60)
    print(f"  Sequence:        {n_chars:,} frames ({HOP_LENGTH/SR*1000:.0f}ms each)")
    print(f"  Alphabet:        {occupied_bins} bins (A-T, dominant freq per frame)")
    print(f"  H0:              {h0:.4f} bits (max {h0_max:.4f} bits)")
    print(f"  H2:              {h2:.4f} bits")
    print(f"  H3:              {h3:.4f} bits")
    print(f"  Structure score: {structure_score:.4f}  (1 - H3/H0)")
    print(f"  Sequential score:{sequential_score:.4f}  (1 - H3/H2_shuffled)")
    print(f"  Shuffled H3:     {h3_shuffled:.4f} (control)")
    print(f"  MI(lag=1):       {mi_profile[1]:.4f} bits")
    print(f"  MI(lag=10):      {mi_profile[10]:.4f} bits")
    print(f"  MI(lag=100):     {mi_profile[100]:.4f} bits")
    print("=" * 60)


if __name__ == "__main__":
    main()
