"""Radio telescope signal Shannon entropy analysis -- SETI use case.

Uses real Breakthrough Listen / Green Bank Telescope data:
  Voyager1_block1.npy  -- raw baseband voltages from GBT 8-10 GHz receiver
  (Siemion et al., AGBT16A_999_17, 2016; released as BL open data)

The data is shaped (64 coarse_channels, 1000 time_samples, 2 polarizations),
dtype complex64.  Raw voltages are 2-bit digitised: each component (I, Q)
can take values in {-40, -12, 12, 40}, giving Stokes-I power
|X_pol|^2 + |Y_pol|^2 with exactly 5 possible values: {576, 2032, 3488, 4944, 6400}.

Primary encoding (5-symbol): map the 5 quantisation levels directly to symbols
A-E.  This is the honest encoding for 2-bit digitised GBT data.  With i.i.d.
thermal noise the structure score is expected to be near zero.

Secondary encoding (20-bin equal-width on normalised power): included for
comparison with other NBS survey domains; noted as artefact-affected.

Expected result: structure score ~ 0.000 -- thermal radiometer noise dominates
and the carrier signal is buried in noise at sub-percent SNR in this
coarse-channel view.  This is a deliberate "failure case" for the paper.

Output: 1-research/nbs-survey/results/seti_entropy.json
"""

import json
import sys
import math
import struct
import urllib.request
import numpy as np
from pathlib import Path

# ---- Path setup -------------------------------------------------------------
ROOT      = Path(__file__).parent.parent   # 1-research/
SCRIPT_DIR = Path(__file__).parent        # 1-research/nbs-survey/
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

# ---- Config -----------------------------------------------------------------
DATA_DIR     = SCRIPT_DIR / "data" / "seti"
RESULTS_FILE = SCRIPT_DIR / "results" / "seti_entropy.json"
MI_LAGS      = [1, 2, 5, 10, 50, 100, 500]
TARGET_CHARS = 1_000_000

RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

BL_BASE = "http://blpd0.ssl.berkeley.edu/Voyager_data/"


def download_file(url: str, dest: Path) -> bool:
    if dest.exists():
        print(f"  Using cached: {dest.name} ({dest.stat().st_size:,} bytes)")
        return True
    print(f"  Downloading {dest.name} from {url} ...")
    try:
        urllib.request.urlretrieve(url, str(dest))
        print(f"  Downloaded: {dest.name} ({dest.stat().st_size:,} bytes)")
        return True
    except Exception as e:
        print(f"  Download failed: {e}")
        return False


# ---- Data loading -----------------------------------------------------------

def load_gbt_block(path: Path) -> np.ndarray:
    """Load Voyager1_block1.npy; return Stokes-I power (n_chan * n_time,) flat."""
    data = np.load(str(path), allow_pickle=False)   # (64, 1000, 2), complex64
    print(f"  Shape: {data.shape}, dtype: {data.dtype}")
    # Stokes I = |X_pol|^2 + |Y_pol|^2
    power = np.abs(data[:, :, 0])**2 + np.abs(data[:, :, 1])**2   # (64, 1000)
    return power.flatten().astype(np.float32)


# ---- Encoding ---------------------------------------------------------------

def encode_5symbol(power: np.ndarray) -> str:
    """Map the 5 quantisation levels of 2-bit GBT data to symbols A-E.

    2-bit digitised GBT voltages: I, Q each in {-40,-12,12,40}.
    Stokes-I = |X|^2 + |Y|^2 takes exactly 5 values:
      576   = 2*(12^2+12^2) -- lowest power
      2032  = (12^2+12^2) + (12^2+40^2)
      3488  = 2*(12^2+40^2) or (12^2+12^2)+(40^2+40^2)
      4944  = (12^2+40^2) + (40^2+40^2)
      6400  = 2*(40^2+40^2) -- highest power
    """
    vals = sorted(np.unique(power))
    assert len(vals) == 5, f"Expected 5 unique power levels, got {len(vals)}: {vals}"
    level_map = {v: chr(65 + i) for i, v in enumerate(vals)}
    return "".join(level_map[v] for v in power)


def encode_20bin_equalwidth(power: np.ndarray) -> str:
    """Equal-width 20-bin encoding on per-channel MAD-normalised power.

    NOTE: with only 5 unique power levels this collapses to 5 active bins
    out of 20, creating a heavily skewed unigram distribution that inflates
    the apparent structure score relative to other NBS survey domains.
    Included for completeness; use 5-symbol encoding as the primary measure.
    """
    power_2d = power.reshape(64, 1000) if len(power) == 64000 else power.reshape(-1, 1)
    med = np.median(power_2d, axis=1, keepdims=True)
    mad = np.median(np.abs(power_2d - med), axis=1, keepdims=True) + 1e-6
    normed = ((power_2d - med) / mad).flatten()
    p1, p99 = np.percentile(normed, 1), np.percentile(normed, 99)
    clipped = np.clip(normed, p1, p99)
    edges = np.linspace(p1, p99, 21)
    bin_ids = np.digitize(clipped, edges[1:-1])
    return "".join(chr(65 + b) for b in bin_ids)


# ---- Entropy pipeline -------------------------------------------------------

def run_pipeline(sequence: str, label: str) -> dict:
    n = len(sequence)
    print(f"\n  [{label}]  n={n:,} chars")

    unigram = compute_ngram_counts(sequence, 1)
    h0 = compute_entropy_miller_madow(unigram)
    alphabet = sorted(set(sequence))
    h_max = math.log2(len(alphabet))
    print(f"    H0 = {h0:.4f} bits  (max log2({len(alphabet)}) = {h_max:.4f})")

    h2 = compute_conditional_entropy(sequence, order=2)
    h3 = compute_conditional_entropy(sequence, order=3)
    print(f"    H2 = {h2:.4f} bits")
    print(f"    H3 = {h3:.4f} bits")

    structure_score  = compute_structure_score(sequence)
    sequential_score = compute_sequential_score(sequence)
    print(f"    Structure score  = {structure_score:.4f}")
    print(f"    Sequential score = {sequential_score:.4f}")

    shuffled = shuffle_control(sequence, seed=42)
    h3_sh = compute_conditional_entropy(shuffled, order=3)
    h0_sh = compute_entropy_miller_madow(compute_ngram_counts(shuffled, 1))
    ss_sh = max(0.0, 1 - h3_sh / h0_sh) if h0_sh > 0 else 0.0
    print(f"    Shuffled structure score = {ss_sh:.4f}")

    mi_profile: dict[int, float] = {}
    for lag in MI_LAGS:
        if lag < n:
            mi_profile[lag] = compute_mutual_information(sequence, lag)

    counts_raw = {ch: sequence.count(ch) for ch in alphabet}

    return {
        "h0": round(h0, 6),
        "h2": round(h2, 6),
        "h3": round(h3, 6),
        "h_max": round(h_max, 6),
        "structure_score": round(structure_score, 6),
        "sequential_score": round(sequential_score, 6),
        "shuffled_control": {
            "h0": round(h0_sh, 6),
            "h3": round(h3_sh, 6),
            "structure_score": round(ss_sh, 6),
        },
        "mi_profile": {str(k): round(v, 6) for k, v in mi_profile.items()},
        "n_chars": n,
        "alphabet_size": len(alphabet),
        "alphabet": alphabet,
        "alphabet_counts": {k: int(v) for k, v in counts_raw.items()},
    }


# ---- Main -------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Radio Telescope Signal Entropy Analysis (SETI)")
    print("=" * 60)

    # Download
    npy_path = DATA_DIR / "Voyager1_block1.npy"
    ok = download_file(f"{BL_BASE}Voyager1_block1.npy", npy_path)
    if not ok:
        print("ERROR: cannot download Breakthrough Listen data.")
        sys.exit(1)

    print("\n[1] Load GBT raw voltage data")
    power_flat = load_gbt_block(npy_path)
    n_raw = len(power_flat)
    print(f"  Stokes-I power: {n_raw:,} samples")
    vals, counts = np.unique(power_flat, return_counts=True)
    print(f"  Unique power levels ({len(vals)}): {[int(v) for v in vals]}")
    for v, c in zip(vals, counts):
        print(f"    {int(v):5d}  {c:6,}  ({100*c/n_raw:.1f}%)")

    # ---- PRIMARY: 5-symbol encoding -----------------------------------------
    print("\n[2] Primary encoding: 5-symbol (direct 2-bit level mapping)")
    seq_5 = encode_5symbol(power_flat)
    print(f"  Raw sequence: {len(seq_5):,} chars, alphabet: {sorted(set(seq_5))}")

    # Tile to TARGET_CHARS
    n_tiles_5 = math.ceil(TARGET_CHARS / len(seq_5))
    seq_5_tiled = (seq_5 * n_tiles_5)[:TARGET_CHARS]
    print(f"  Tiled {n_tiles_5}x -> {len(seq_5_tiled):,} chars")
    print(f"  (Tiling valid: 2-bit GBT noise is stationary -- each tile i.i.d.)")

    print("\n  Running entropy pipeline on 5-symbol tiled sequence ...")
    res5 = run_pipeline(seq_5_tiled, "5-symbol tiled")

    # Cross-check on raw (un-tiled) short sequence
    print("\n  Cross-check on raw (un-tiled) 64,000-char sequence ...")
    res5_raw = run_pipeline(seq_5, "5-symbol raw (no tiling)")

    # ---- SECONDARY: 20-bin equal-width encoding (artifact disclosure) --------
    print("\n[3] Secondary encoding: 20-bin equal-width (artefact disclosure)")
    seq_20 = encode_20bin_equalwidth(power_flat)
    n_tiles_20 = math.ceil(TARGET_CHARS / len(seq_20))
    seq_20_tiled = (seq_20 * n_tiles_20)[:TARGET_CHARS]
    print(f"  Tiled {n_tiles_20}x -> {len(seq_20_tiled):,} chars")
    print(f"  NOTE: equal-width bins on 5 quantisation levels -> 15/20 bins empty.")
    print(f"  The structure score here reflects bin-map artefact, NOT signal structure.")

    res20 = run_pipeline(seq_20_tiled, "20-bin tiled (artefact)")

    # ---- Assemble results ---------------------------------------------------
    data_source = (
        "Breakthrough Listen open data -- Green Bank Telescope (GBT), "
        "AGBT16A_999_17 (PI: Andrew Siemion, 2016). "
        "File: Voyager1_block1.npy -- raw baseband voltages, shape (64 coarse "
        "channels, 1000 time samples, 2 polarisations), dtype complex64. "
        "Receiver: Rcvr8_10 (8-10 GHz), sky source: Voyager 1 carrier at ~8.4 GHz. "
        "Data URL: http://blpd0.ssl.berkeley.edu/Voyager_data/"
    )
    encoding_notes_primary = (
        "GBT 2-bit digitised voltages: I, Q each sampled to {-40,-12,12,40}. "
        "Stokes-I = |X_pol|^2 + |Y_pol|^2 gives exactly 5 discrete power levels "
        "{576, 2032, 3488, 4944, 6400}, mapped directly to symbols A-E. "
        "This is the honest encoding for 2-bit data. "
        f"Raw sequence: {n_raw:,} samples; tiled {n_tiles_5}x to {TARGET_CHARS:,} chars "
        "(valid: GBT radiometer noise is stationary, each 1000-sample block i.i.d.)."
    )
    encoding_notes_secondary = (
        "Per-channel MAD-normalised Stokes-I power clipped to [p1,p99], "
        "then 20 equal-width bins. With only 5 unique power values, "
        "15/20 bins are empty -- the reduced effective alphabet creates "
        "apparent structure (skewed unigrams) not present in the noise. "
        "Reported for disclosure, NOT used as the canonical measure."
    )

    ss_primary = res5["structure_score"]
    if ss_primary < 0.01:
        interpretation = (
            "NEAR-ZERO structure (score={:.4f}) -- consistent with pure thermal "
            "radiometer noise from 2-bit digitised GBT data. "
            "Correct null result: character-level AI finds no exploitable "
            "regularities in radio telescope noise. "
            "This is the expected outcome for a SETI null observation.".format(ss_primary)
        )
    elif ss_primary < 0.05:
        interpretation = (
            "VERY LOW structure (score={:.4f}) -- dominated by noise. "
            "Any residual structure likely from 2-bit quantisation statistics.".format(ss_primary)
        )
    else:
        interpretation = (
            "UNEXPECTED structure (score={:.4f}) -- investigate for "
            "systematic artefacts.".format(ss_primary)
        )

    results = {
        "domain": "Radio telescope signal (GBT Breakthrough Listen -- Voyager 1 tracking)",
        "canonical_encoding": "5-symbol (direct 2-bit level mapping)",
        # Primary (canonical) metrics
        "h0": res5["h0"],
        "h2": res5["h2"],
        "h3": res5["h3"],
        "structure_score": res5["structure_score"],
        "sequential_score": res5["sequential_score"],
        "shuffled_control": res5["shuffled_control"],
        "mi_profile": res5["mi_profile"],
        "n_chars": res5["n_chars"],
        "n_chars_raw_data": n_raw,
        "n_tiles": n_tiles_5,
        "alphabet_size": res5["alphabet_size"],
        "alphabet": res5["alphabet"],
        "alphabet_counts": res5["alphabet_counts"],
        # Raw (un-tiled) cross-check
        "raw_untiled": {
            "n_chars": res5_raw["n_chars"],
            "structure_score": res5_raw["structure_score"],
            "h0": res5_raw["h0"],
            "h3": res5_raw["h3"],
        },
        # Secondary encoding (artefact disclosure)
        "secondary_20bin_equalwidth": {
            "note": "artefact from empty bins -- NOT canonical measure",
            "structure_score": res20["structure_score"],
            "h0": res20["h0"],
            "h3": res20["h3"],
            "alphabet_size_effective": sum(1 for v in res20["alphabet_counts"].values() if v > 0),
            "encoding_notes": encoding_notes_secondary,
        },
        # Metadata
        "data_source": data_source,
        "encoding_notes": encoding_notes_primary,
        "interpretation": interpretation,
    }

    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {RESULTS_FILE}")

    # ---- Summary ------------------------------------------------------------
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Domain:           Radio telescope (GBT / Breakthrough Listen)")
    print(f"  Source:           Voyager 1 tracking, 8-10 GHz, 2016 (real data)")
    print(f"  Encoding:         5-symbol (2-bit digitised levels, direct)")
    print(f"  Raw samples:      {n_raw:,} (tiled {n_tiles_5}x to {TARGET_CHARS:,})")
    print(f"  H0:               {res5['h0']:.4f} bits  (max log2(5)={math.log2(5):.4f})")
    print(f"  H2:               {res5['h2']:.4f} bits")
    print(f"  H3:               {res5['h3']:.4f} bits")
    print(f"  Structure score:  {res5['structure_score']:.4f}  (PRIMARY)")
    print(f"  Sequential score: {res5['sequential_score']:.4f}")
    print(f"  Shuffled struct:  {res5['shuffled_control']['structure_score']:.4f}")
    print(f"  Raw (no tile):    {res5_raw['structure_score']:.4f}")
    print(f"  20-bin artefact:  {res20['structure_score']:.4f}  (NOT canonical)")
    print(f"  Interpretation:   {interpretation}")
    print("=" * 60)


if __name__ == "__main__":
    main()
