"""EEG voltage time-series Shannon entropy analysis.

Uses the PhysioNet EEG Motor Movement/Imagery Dataset (Schalk et al. 2004),
accessed via MNE-Python. Subjects 1-10, runs 1 (eyes-open baseline) and 2
(eyes-closed baseline) — resting state EEG.

Channel: Cz (central midline, standard reference for resting EEG)
Preprocessing:
  - Bandpass 0.5-40 Hz (removes DC drift and high-frequency muscle artefact)
  - Downsample 160 Hz → 100 Hz (matches signal bandwidth, avoids oversampling)
  - Discretise to 20 equal-width amplitude bins (p1-p99 range)
  - Encode as characters A-T

Output: 1-research/nbs-survey/results/eeg_entropy.json
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
DATA_DIR     = SCRIPT_DIR / "data" / "eeg"
RESULTS_FILE = Path(__file__).parent.parent.parent / "1-research" / "nbs-survey" / "results" / "eeg_entropy.json"

# Override: results go to the path specified in the task
RESULTS_FILE = Path("C:/Users/danp9/Claude vibe coding/speaking-opportunities/1-research/nbs-survey/results/eeg_entropy.json")

N_BINS       = 20
MI_LAGS      = [1, 2, 5, 10, 50, 100, 500]

# EEG parameters
CHANNEL      = "Cz.."          # Central midline electrode (MNE-EEGBCI label has trailing dots)
ORIG_SFREQ   = 160.0           # Original sampling rate in Hz
TARGET_SFREQ = 100.0           # Downsample target (matches ~40 Hz bandwidth with margin)
BANDPASS_LOW = 0.5             # Hz
BANDPASS_HIGH = 40.0           # Hz

# Subjects and runs for resting state
SUBJECTS     = list(range(1, 11))    # S001 – S010
RUNS         = [1, 2]                # 1 = eyes open, 2 = eyes closed baseline

RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)


# ── Data acquisition ──────────────────────────────────────────────────────────

def load_eeg_data(data_dir: Path) -> np.ndarray:
    """Load and preprocess EEG data from downloaded EDF files.

    Returns concatenated Cz channel data (in volts) at TARGET_SFREQ.
    """
    import mne

    edf_dir = data_dir / "MNE-eegbci-data" / "files" / "eegmmidb" / "1.0.0"

    all_segments = []
    files_loaded = []

    for subj in SUBJECTS:
        for run in RUNS:
            edf_path = edf_dir / f"S{subj:03d}" / f"S{subj:03d}R{run:02d}.edf"
            if not edf_path.exists():
                print(f"  Missing: {edf_path} — skipping")
                continue

            try:
                raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose=False)
            except Exception as e:
                print(f"  Error loading {edf_path.name}: {e} — skipping")
                continue

            # Standardise channel names (MNE-EEGBCI has trailing dots)
            ch_names = raw.ch_names

            # Find Cz channel (may be 'Cz..', 'Cz', 'CZ', etc.)
            cz_name = None
            for candidate in ["Cz..", "Cz", "CZ", "cz"]:
                if candidate in ch_names:
                    cz_name = candidate
                    break
            if cz_name is None:
                # Try partial match
                for name in ch_names:
                    if name.lower().startswith("cz"):
                        cz_name = name
                        break
            if cz_name is None:
                print(f"  No Cz channel found in {edf_path.name} — channels: {ch_names[:10]}")
                continue

            # Pick Cz only
            raw.pick([cz_name])

            # Bandpass filter 0.5–40 Hz (removes DC and high-freq artefacts)
            raw.filter(BANDPASS_LOW, BANDPASS_HIGH, fir_design="firwin", verbose=False)

            # Resample to TARGET_SFREQ
            if raw.info["sfreq"] != TARGET_SFREQ:
                raw.resample(TARGET_SFREQ, verbose=False)

            # Extract data (shape: [1, n_times])
            data, _ = raw[:, :]
            segment = data[0]  # 1D array

            all_segments.append(segment)
            files_loaded.append(edf_path.name)

            print(f"  Loaded S{subj:03d}R{run:02d}: {len(segment):,} samples "
                  f"at {TARGET_SFREQ:.0f} Hz ({len(segment)/TARGET_SFREQ:.1f}s)")

    if not all_segments:
        raise RuntimeError("No EEG segments loaded!")

    print(f"\nLoaded {len(files_loaded)} files: {files_loaded}")
    combined = np.concatenate(all_segments)
    print(f"Combined: {len(combined):,} samples "
          f"= {len(combined)/TARGET_SFREQ:.1f}s at {TARGET_SFREQ:.0f} Hz")
    return combined, files_loaded


# ── Discretisation ────────────────────────────────────────────────────────────

def discretise(data: np.ndarray, n_bins: int = N_BINS) -> str:
    """Map EEG voltage to n_bins equal-width bins, encode as chars.

    Equal-width bins over [p1, p99] of the data (robust to artefact spikes).
    Bins encoded as ASCII characters starting at 'A'.
    """
    p1, p99 = np.percentile(data, 1), np.percentile(data, 99)
    print(f"Amplitude range (p1–p99): [{p1*1e6:.2f}, {p99*1e6:.2f}] µV")

    # Clip to [p1, p99]
    clipped = np.clip(data, p1, p99)

    # Equal-width bins
    edges = np.linspace(p1, p99, n_bins + 1)
    bin_ids = np.digitize(clipped, edges[1:-1])   # 0 … n_bins-1

    chars = [chr(ord('A') + b) for b in bin_ids]
    return "".join(chars)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("EEG Voltage Time-Series Entropy Analysis")
    print("Dataset: PhysioNet EEGMMIDB (Schalk et al. 2004)")
    print(f"Channel: Cz (central midline)")
    print(f"Subjects: {SUBJECTS[0]}–{SUBJECTS[-1]}, Runs: {RUNS} (resting state)")
    print(f"Bandpass: {BANDPASS_LOW}–{BANDPASS_HIGH} Hz")
    print(f"Sampling rate: {ORIG_SFREQ:.0f} Hz -> {TARGET_SFREQ:.0f} Hz")
    print("=" * 60)

    # ── Load ─────────────────────────────────────────────────────────────────
    data, files_loaded = load_eeg_data(DATA_DIR)
    n_raw = len(data)
    print(f"\nTotal EEG samples: {n_raw:,} ({n_raw/TARGET_SFREQ:.1f}s)")

    # ── Discretise ───────────────────────────────────────────────────────────
    print(f"\nDiscretising {n_raw:,} samples into {N_BINS} amplitude bins...")
    sequence = discretise(data, N_BINS)
    n_chars = len(sequence)
    print(f"Sequence length: {n_chars:,} chars")

    # Alphabet report
    counts_raw: dict[str, int] = {}
    for ch in sequence:
        counts_raw[ch] = counts_raw.get(ch, 0) + 1
    alphabet = sorted(counts_raw.keys())
    alphabet_size = len(alphabet)
    print(f"\nAlphabet size: {alphabet_size} symbols")
    for ch in sorted(counts_raw, key=lambda x: -counts_raw[x]):
        print(f"  bin '{ch}': {counts_raw[ch]:,}  ({100*counts_raw[ch]/n_chars:.1f}%)")

    # ── H0 ───────────────────────────────────────────────────────────────────
    print("\n--- H0 (unigram entropy) ---")
    unigram_counts = compute_ngram_counts(sequence, 1)
    h0 = compute_entropy_miller_madow(unigram_counts)
    h_max = math.log2(alphabet_size)
    print(f"H0 = {h0:.4f} bits  (max possible = log2({alphabet_size}) = {h_max:.4f} bits)")

    # ── H2 ───────────────────────────────────────────────────────────────────
    print("\n--- H2 (bigram conditional entropy) ---")
    h2 = compute_conditional_entropy(sequence, order=2)
    print(f"H2 = {h2:.4f} bits")

    # ── H3 ───────────────────────────────────────────────────────────────────
    print("\n--- H3 (trigram conditional entropy) ---")
    h3 = compute_conditional_entropy(sequence, order=3)
    print(f"H3 = {h3:.4f} bits")

    # ── Structure score ───────────────────────────────────────────────────────
    print("\n--- Structure score = 1 - H3/H0 ---")
    structure_score = compute_structure_score(sequence)
    print(f"Structure score = {structure_score:.4f}")

    # ── Sequential score ──────────────────────────────────────────────────────
    print("\n--- Sequential score = 1 - H3/H2_shuffled ---")
    sequential_score = compute_sequential_score(sequence)
    print(f"Sequential score = {sequential_score:.4f}")

    # ── Shuffled control ──────────────────────────────────────────────────────
    print("\n--- Shuffled control ---")
    shuffled = shuffle_control(sequence, seed=42)
    h3_shuffled = compute_conditional_entropy(shuffled, order=3)
    h0_shuffled = compute_entropy_miller_madow(compute_ngram_counts(shuffled, 1))
    structure_score_shuffled = max(0.0, 1 - h3_shuffled / h0_shuffled) if h0_shuffled > 0 else 0.0
    print(f"Shuffled H0 = {h0_shuffled:.4f} bits")
    print(f"Shuffled H3 = {h3_shuffled:.4f} bits")
    print(f"Shuffled structure score = {structure_score_shuffled:.4f}")

    # ── MI profile ────────────────────────────────────────────────────────────
    print("\n--- MI decay profile ---")
    mi_profile: dict[int, float] = {}
    for lag in MI_LAGS:
        if lag < n_chars:
            mi = compute_mutual_information(sequence, lag)
            mi_profile[lag] = mi
            lag_ms = lag / TARGET_SFREQ * 1000
            print(f"  MI(lag={lag:4d} = {lag_ms:.0f}ms) = {mi:.4f} bits")

    # ── Assemble results ──────────────────────────────────────────────────────
    data_source = (
        "PhysioNet EEG Motor Movement/Imagery Dataset (Schalk et al. 2004, "
        "doi:10.1186/1743-0003-1-14; Goldberger et al. 2000, PhysioNet). "
        f"Subjects 1–{SUBJECTS[-1]}, runs 1 (eyes-open) and 2 (eyes-closed) "
        "resting-state baseline. 64-channel 160 Hz EEG. Accessed via MNE-Python "
        "mne.datasets.eegbci."
    )
    encoding_notes = (
        f"Single channel: Cz (central midline). Bandpass filtered {BANDPASS_LOW}–"
        f"{BANDPASS_HIGH} Hz (removes DC drift and muscle artefact above alpha/beta "
        f"range). Resampled from {ORIG_SFREQ:.0f} Hz to {TARGET_SFREQ:.0f} Hz to match "
        f"signal bandwidth (EEG meaningful content 0.5–40 Hz). Amplitude clipped to "
        f"[p1, p99] to exclude artefact spikes, then discretised into {N_BINS} "
        f"equal-width bins. Each sample encoded as ASCII character 'A'–'T'. "
        f"{len(files_loaded)} recordings concatenated: {len(SUBJECTS)} subjects × "
        f"{len(RUNS)} runs = {len(files_loaded)} files."
    )

    results = {
        "domain": "EEG voltage time-series (Cz channel, resting state)",
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
        "n_bins": N_BINS,
        "alphabet_size": alphabet_size,
        "alphabet": alphabet,
        "alphabet_counts": {k: int(v) for k, v in counts_raw.items()},
        "sampling_rate_hz": TARGET_SFREQ,
        "original_sampling_rate_hz": ORIG_SFREQ,
        "bandpass_hz": [BANDPASS_LOW, BANDPASS_HIGH],
        "channel": "Cz",
        "subjects": SUBJECTS,
        "runs": RUNS,
        "n_files": len(files_loaded),
        "data_source": data_source,
        "encoding_notes": encoding_notes,
    }

    # ── Save ──────────────────────────────────────────────────────────────────
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {RESULTS_FILE}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Domain:              EEG voltage time-series (Cz, resting)")
    print(f"  Channel:             Cz (central midline)")
    print(f"  Sampling rate:       {TARGET_SFREQ:.0f} Hz (downsampled from {ORIG_SFREQ:.0f} Hz)")
    print(f"  Bandpass:            {BANDPASS_LOW}–{BANDPASS_HIGH} Hz")
    print(f"  Recordings:          {len(files_loaded)} files ({len(SUBJECTS)} subjects × {len(RUNS)} runs)")
    print(f"  Samples:             {n_chars:,} chars ({n_chars/TARGET_SFREQ:.1f}s total)")
    print(f"  Bins:                {N_BINS}")
    print(f"  H0 (unigram):        {h0:.4f} bits  (max {h_max:.4f})")
    print(f"  H2 (bigram):         {h2:.4f} bits")
    print(f"  H3 (trigram):        {h3:.4f} bits")
    print(f"  Structure score:     {structure_score:.4f}")
    print(f"  Sequential score:    {sequential_score:.4f}")
    print(f"  Shuffled structure:  {structure_score_shuffled:.4f}")
    interpretation = (
        "HIGH structure" if structure_score > 0.5 else
        "MODERATE structure" if structure_score > 0.1 else
        "LOW structure (near-random)"
    )
    print(f"  Interpretation:      {interpretation}")
    print("=" * 60)


if __name__ == "__main__":
    main()
