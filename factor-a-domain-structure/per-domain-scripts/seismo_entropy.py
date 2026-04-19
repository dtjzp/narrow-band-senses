"""Seismogram waveform Shannon entropy analysis.

Downloads 2 days of broadband vertical (BHZ) data from station IU.ANMO
(Albuquerque NM reference station) via IRIS FDSN, discretises amplitudes
to 20 bins, and runs the NBS entropy pipeline.

Output: 1-research/nbs-survey/results/seismo_entropy.json
"""

import json
import sys
import os
import math
import numpy as np
from pathlib import Path
from collections import Counter

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
DATA_DIR    = SCRIPT_DIR / "data" / "seismo"
RESULTS_FILE = SCRIPT_DIR / "results" / "seismo_entropy.json"
N_BINS      = 20
MI_LAGS     = [1, 2, 5, 10, 50, 100, 500]

# Target ~1 million samples (BHZ = 20 sps → 1 day = 1,728,000 samples)
# 2 days gives us plenty; we'll cap at 1M after discretisation
TARGET_CHARS = 1_000_000

RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)


# ── Data acquisition ──────────────────────────────────────────────────────────

def download_waveforms(data_dir: Path) -> Path:
    """Download BHZ data from IRIS for IU.ANMO if not already cached."""
    cache_file = data_dir / "ANMO_BHZ_2020-01-01_2d.npy"
    if cache_file.exists():
        print(f"Using cached waveform data: {cache_file}")
        return cache_file

    print("Downloading seismogram data from IRIS FDSN...")
    print("  Network: IU  Station: ANMO  Location: 00  Channel: BHZ")
    print("  Period: 2020-01-01 to 2020-01-03 (2 days, ~3.5M samples at 20 sps)")

    from obspy.clients.fdsn import Client
    from obspy import UTCDateTime

    client = Client("IRIS")
    t_start = UTCDateTime("2020-01-01")
    t_end   = UTCDateTime("2020-01-03")

    st = client.get_waveforms("IU", "ANMO", "00", "BHZ", t_start, t_end)
    print(f"Downloaded {len(st)} trace(s)")
    for tr in st:
        print(f"  {tr.id}  {tr.stats.starttime} – {tr.stats.endtime}  "
              f"npts={tr.stats.npts}  sps={tr.stats.sampling_rate}")

    # Merge any gaps, fill with zeros (rare short gaps), convert to float64
    st.merge(method=1, fill_value=0)
    data = st[0].data.astype(np.float64)
    print(f"Merged trace: {len(data):,} samples")

    np.save(cache_file, data)
    print(f"Cached to {cache_file}")
    return cache_file


def fallback_download_ascii(data_dir: Path) -> Path | None:
    """Fallback: IRIS timeseries ASCII web service (if FDSN unavailable)."""
    import urllib.request, urllib.error

    cache_file = data_dir / "ANMO_BHZ_ascii.npy"
    if cache_file.exists():
        return cache_file

    url = (
        "http://service.iris.edu/irisws/timeseries/1/query"
        "?net=IU&sta=ANMO&loc=00&cha=BHZ"
        "&starttime=2020-01-01T00:00:00&endtime=2020-01-02T00:00:00"
        "&output=ascii1"
    )
    print(f"Fallback: fetching ASCII from IRIS timeseries WS...")
    try:
        with urllib.request.urlopen(url, timeout=60) as resp:
            text = resp.read().decode()
    except Exception as e:
        print(f"  ASCII fetch failed: {e}")
        return None

    samples = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("TIMESERIES") or line.startswith("#"):
            continue
        try:
            samples.append(float(line.split()[-1]))
        except ValueError:
            continue

    if not samples:
        print("  Parsed 0 samples from ASCII response")
        return None

    data = np.array(samples, dtype=np.float64)
    print(f"  Parsed {len(data):,} samples from ASCII")
    np.save(cache_file, data)
    return cache_file


# ── Discretisation ────────────────────────────────────────────────────────────

def discretise(data: np.ndarray, n_bins: int = N_BINS) -> str:
    """Map continuous amplitudes to n_bins equal-width bins, encode as chars.

    Equal-width bins over [p1, p99] of the data (robust to outlier spikes).
    Bins encoded as ASCII characters starting at 'A'.
    """
    p1, p99 = np.percentile(data, 1), np.percentile(data, 99)
    print(f"Amplitude range (p1–p99): [{p1:.2f}, {p99:.2f}]")

    # Clip to [p1, p99] so extreme spikes don't dominate
    clipped = np.clip(data, p1, p99)

    # Equal-width bins
    edges = np.linspace(p1, p99, n_bins + 1)
    bin_ids = np.digitize(clipped, edges[1:-1])   # 0 … n_bins-1

    # Encode as characters 'A'..'T' for n_bins=20
    chars = [chr(ord('A') + b) for b in bin_ids]
    return "".join(chars)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Seismogram Waveform Entropy Analysis")
    print("=" * 60)

    # ── Download / load ───────────────────────────────────────────────────────
    data = None
    download_source = "IRIS FDSN (obspy)"

    try:
        cache_file = download_waveforms(DATA_DIR)
        data = np.load(cache_file)
        print(f"Loaded {len(data):,} samples from cache")
    except Exception as e:
        print(f"FDSN download failed: {e}")
        print("Trying ASCII fallback...")
        fallback_file = fallback_download_ascii(DATA_DIR)
        if fallback_file is not None:
            data = np.load(fallback_file)
            download_source = "IRIS timeseries web service (ASCII)"
            print(f"Loaded {len(data):,} samples via ASCII fallback")
        else:
            print("ERROR: Both download methods failed. Exiting.")
            sys.exit(1)

    # ── Discretise ────────────────────────────────────────────────────────────
    print(f"\nDiscretising {len(data):,} samples into {N_BINS} amplitude bins...")
    sequence_full = discretise(data, N_BINS)

    # Cap at TARGET_CHARS
    if len(sequence_full) > TARGET_CHARS:
        sequence = sequence_full[:TARGET_CHARS]
        print(f"Capped to first {len(sequence):,} chars")
    else:
        sequence = sequence_full
    n_chars = len(sequence)
    print(f"Sequence length for analysis: {n_chars:,} chars")

    # Alphabet report
    counts_raw: dict[str, int] = {}
    for ch in sequence:
        counts_raw[ch] = counts_raw.get(ch, 0) + 1
    alphabet = sorted(counts_raw.keys())
    alphabet_size = len(alphabet)
    print(f"\nAlphabet size: {alphabet_size} symbols (bins)")
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
        mi = compute_mutual_information(sequence, lag)
        mi_profile[lag] = mi
        print(f"  MI(lag={lag:4d}) = {mi:.4f} bits")

    # ── Assemble results ──────────────────────────────────────────────────────
    data_source = (
        "IRIS FDSN waveform service via obspy — station IU.ANMO (Albuquerque NM, "
        "global reference station), channel BHZ (broadband vertical, 20 sps), "
        "2020-01-01 to 2020-01-03. Full-resolution continuous seismic noise + "
        "any regional/teleseismic events."
    )
    encoding_notes = (
        f"Raw 32-bit integer counts (nm/s), merged to single trace, clipped to "
        f"[p1, p99] amplitude range to remove spike outliers, then discretised "
        f"into {N_BINS} equal-width amplitude bins. Each sample encoded as a single "
        f"ASCII character 'A'–'T'. 20 sps × ~2 days ≈ 3.5M samples; capped at "
        f"{TARGET_CHARS:,} for entropy computation."
    )

    results = {
        "domain": "Seismogram waveform (IU.ANMO BHZ broadband vertical)",
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
        "data_source": data_source,
        "encoding_notes": encoding_notes,
        "download_source": download_source,
    }

    # ── Save ──────────────────────────────────────────────────────────────────
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {RESULTS_FILE}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Domain:           Seismogram waveform (IU.ANMO BHZ)")
    print(f"  Samples:          {n_chars:,} chars ({N_BINS} amplitude bins)")
    print(f"  H0 (unigram):     {h0:.4f} bits  (max {h_max:.4f})")
    print(f"  H2 (bigram):      {h2:.4f} bits")
    print(f"  H3 (trigram):     {h3:.4f} bits")
    print(f"  Structure score:  {structure_score:.4f}")
    print(f"  Sequential score: {sequential_score:.4f}")
    print(f"  Shuffled struct:  {structure_score_shuffled:.4f}")
    interpretation = (
        "HIGH structure" if structure_score > 0.5 else
        "MODERATE structure" if structure_score > 0.25 else
        "LOW structure (near-random)"
    )
    print(f"  Interpretation:   {interpretation}")
    print("=" * 60)


if __name__ == "__main__":
    main()
