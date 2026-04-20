"""Satellite TLE (Two-Line Element) Shannon entropy analysis.

Downloads the full CelesTrak active satellite catalogue (~6,000+ satellites).
TLEs are treated as raw text — the TLE format IS the notation.
Each TLE pair (2 lines × 69 chars) encodes orbital parameters in a
fixed-width, physically constrained format.

Two encoding modes:
  1. RAW TEXT — entire TLE corpus as a character stream (the natural notation)
  2. FIELD-LEVEL — each orbital parameter discretised to 20 bins

Primary analysis: raw text (consistent with how language / notation entropy
is measured elsewhere in this survey).

Output: 1-research/nbs-survey/results/satellite_entropy.json
"""

import json
import sys
import re
import math
import urllib.request
import numpy as np
from pathlib import Path
from collections import Counter

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
DATA_DIR     = SCRIPT_DIR / "data" / "satellite"
RESULTS_FILE = SCRIPT_DIR / "results" / "satellite_entropy.json"
N_BINS       = 20
MI_LAGS      = [1, 2, 5, 10, 50, 100, 500]
TARGET_CHARS = 1_000_000

DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)

# CelesTrak TLE sources (try in order — full active catalogue preferred)
TLE_SOURCES = [
    ("active",    "https://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=tle",
     "CelesTrak active satellites TLE (~6000+ sats)"),
    ("stations",  "https://celestrak.org/NORAD/elements/gp.php?GROUP=stations&FORMAT=tle",
     "CelesTrak space stations TLE"),
]

DATA_SOURCE = (
    "CelesTrak NORAD Two-Line Element (TLE) sets — active satellite catalogue. "
    "Source: https://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=tle "
    "Downloaded 2026-04-11. "
    "Each TLE entry consists of 2 lines of 69 characters encoding orbital elements: "
    "epoch, eccentricity, inclination, RAAN, argument of perigee, mean motion, "
    "B* drag term, and satellite identifiers. "
    "Format is fixed-width ASCII with strict column assignments and checksums."
)

ENCODING_NOTES = (
    "TLE corpus treated as raw text (the TLE format IS the notation). "
    "Processing: line-1 and line-2 of each TLE concatenated with a separator character '|'. "
    "Each TLE pair → 69 + 1 + 69 + 1 = 140 characters. "
    "Character set: digits 0-9, letters A-Z, '.', '+', '-', ' '. "
    "HIGH structure expected: TLEs use fixed-width columns, limited character set "
    "(digits dominate), and physical constraints on orbital elements "
    "(e.g., eccentricity 0-1, inclination 0-180°, mean motion ~1-16 rev/day). "
    "Strong positional (column-level) structure expected to yield very low H3."
)


# ── TLE parsing helpers ───────────────────────────────────────────────────────

def is_tle_line1(line: str) -> bool:
    return len(line) >= 69 and line[0] == '1' and line[1] == ' '


def is_tle_line2(line: str) -> bool:
    return len(line) >= 69 and line[0] == '2' and line[1] == ' '


def parse_tle_file(raw_text: str) -> list[tuple[str, str]]:
    """Parse raw TLE text → list of (line1, line2) pairs.

    Handles both 2-line format (line1 + line2) and 3-line format
    (name + line1 + line2). Returns only the two orbital data lines.
    """
    lines = [l.strip() for l in raw_text.splitlines()]
    lines = [l for l in lines if l]  # drop blanks

    pairs = []
    i = 0
    while i < len(lines):
        # Skip name lines (not starting with 1 or 2)
        if is_tle_line1(lines[i]):
            l1 = lines[i]
            if i + 1 < len(lines) and is_tle_line2(lines[i + 1]):
                l2 = lines[i + 1]
                pairs.append((l1, l2))
                i += 2
                continue
        i += 1

    return pairs


def download_tle_data(data_dir: Path) -> tuple[str, list[tuple[str, str]], str]:
    """Download TLE data from CelesTrak. Returns (raw_text, tle_pairs, source_label)."""

    for name, url, label in TLE_SOURCES:
        cache_file = data_dir / f"celestrak_{name}.tle"

        if cache_file.exists() and cache_file.stat().st_size > 10_000:
            print(f"Using cached TLE data: {cache_file}")
            raw = cache_file.read_text(encoding="utf-8", errors="replace")
            pairs = parse_tle_file(raw)
            if pairs:
                print(f"  Parsed {len(pairs):,} TLE pairs from cache")
                return raw, pairs, label
            else:
                print(f"  Cache parse failed, re-downloading...")

        print(f"Downloading {label}...")
        print(f"  URL: {url}")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
            print(f"  Downloaded {len(raw):,} chars")
            cache_file.write_text(raw, encoding="utf-8")
            print(f"  Saved to {cache_file}")
            pairs = parse_tle_file(raw)
            print(f"  Parsed {len(pairs):,} TLE pairs")
            if pairs:
                return raw, pairs, label
        except Exception as e:
            print(f"  Failed ({e}), trying next source...")

    raise RuntimeError("All TLE sources failed. Check internet connection.")


def build_character_sequence(tle_pairs: list[tuple[str, str]]) -> str:
    """Build character sequence from TLE pairs.

    Each TLE pair → line1 (69 chars) + '|' + line2 (69 chars) + '|'
    = 140 chars per TLE. Pad/truncate lines to exactly 69 chars.

    The '|' separator marks TLE boundaries so positional structure is preserved.
    """
    parts = []
    for l1, l2 in tle_pairs:
        # Normalise to exactly 69 chars
        l1 = l1[:69].ljust(69)
        l2 = l2[:69].ljust(69)
        parts.append(l1 + "|" + l2 + "|")
    return "".join(parts)


def parse_orbital_elements(tle_pairs: list[tuple[str, str]]) -> dict[str, list[float]]:
    """Extract key orbital parameters as float lists for field-level encoding."""
    fields = {
        "inclination_deg": [],
        "eccentricity": [],
        "mean_motion_rev_per_day": [],
        "bstar_drag": [],
        "raan_deg": [],
        "arg_perigee_deg": [],
        "mean_anomaly_deg": [],
    }

    for l1, l2 in tle_pairs:
        try:
            # Line 1 fields
            bstar_raw = l1[53:61].strip()
            # B* in decimal point assumed format: ±NNNNN±NN → ±0.NNNNN e ±NN
            try:
                if len(bstar_raw) >= 6:
                    mantissa = float(bstar_raw[:-2]) * 1e-5
                    exp = int(bstar_raw[-2:])
                    bstar = mantissa * (10 ** exp)
                else:
                    bstar = float(bstar_raw) if bstar_raw else 0.0
            except Exception:
                bstar = 0.0

            # Line 2 fields (fixed columns per TLE standard)
            incl   = float(l2[8:16].strip())
            raan   = float(l2[17:25].strip())
            ecc    = float("0." + l2[26:33].strip())
            argp   = float(l2[34:42].strip())
            ma     = float(l2[43:51].strip())
            mm     = float(l2[52:63].strip())

            fields["inclination_deg"].append(incl)
            fields["eccentricity"].append(ecc)
            fields["mean_motion_rev_per_day"].append(mm)
            fields["bstar_drag"].append(bstar)
            fields["raan_deg"].append(raan)
            fields["arg_perigee_deg"].append(argp)
            fields["mean_anomaly_deg"].append(ma)

        except Exception:
            continue  # skip malformed TLEs

    return fields


def discretise_field(values: list[float], n_bins: int = N_BINS) -> str:
    """Discretise a float list to n_bins bins using percentile boundaries → char sequence."""
    arr = np.array(values, dtype=np.float64)
    # Use percentile bins (equal-frequency) to spread signal across bins
    boundaries = np.percentile(arr, np.linspace(0, 100, n_bins + 1))
    boundaries = np.unique(boundaries)  # collapse duplicates
    actual_bins = len(boundaries) - 1
    if actual_bins < 1:
        actual_bins = 1

    bins = np.digitize(arr, boundaries[1:], right=False).clip(0, actual_bins - 1)
    # Map to A-T
    chars = [chr(ord("A") + b) for b in bins]
    return "".join(chars)


def report_alphabet(sequence: str, label: str) -> dict:
    """Report character distribution."""
    counts = Counter(sequence)
    total = len(sequence)
    print(f"\n{label} alphabet ({len(counts)} distinct chars):")
    for ch, n in sorted(counts.most_common(20)):
        print(f"  '{ch}': {n:,} ({100*n/total:.2f}%)")
    if len(counts) > 20:
        print(f"  ... ({len(counts) - 20} more chars)")
    return dict(counts)


def run_entropy_pipeline(sequence: str, label: str) -> dict:
    """Run the full NBS entropy pipeline on a character sequence."""
    n_chars = len(sequence)
    counts_1 = Counter(sequence)
    alphabet_size = len(counts_1)

    print(f"\n{'='*60}")
    print(f"Entropy pipeline: {label}")
    print(f"{'='*60}")
    print(f"Sequence length: {n_chars:,} chars, alphabet: {alphabet_size}")

    # H0
    unigram_counts = compute_ngram_counts(sequence, 1)
    h0 = compute_entropy_miller_madow(unigram_counts)
    h0_max = math.log2(alphabet_size)
    print(f"H0 = {h0:.4f} bits  (max = {h0_max:.4f} bits for {alphabet_size} symbols)")

    # H2
    h2 = compute_conditional_entropy(sequence, order=2)
    print(f"H2 = {h2:.4f} bits")

    # H3
    h3 = compute_conditional_entropy(sequence, order=3)
    print(f"H3 = {h3:.4f} bits")

    # Structure + sequential scores
    structure_score = compute_structure_score(sequence)
    sequential_score = compute_sequential_score(sequence)
    print(f"Structure score  = {structure_score:.4f}  (1 - H3/H0)")
    print(f"Sequential score = {sequential_score:.4f}")

    # Shuffled control
    shuffled = shuffle_control(sequence, seed=42)
    h3_shuf = compute_conditional_entropy(shuffled, order=3)
    h0_shuf = compute_entropy_miller_madow(compute_ngram_counts(shuffled, 1))
    ss_shuf = max(0.0, 1 - h3_shuf / h0_shuf) if h0_shuf > 0 else 0.0
    print(f"Shuffled structure score = {ss_shuf:.4f}  (H3={h3_shuf:.4f})")

    # MI profile
    mi_profile = {}
    print("MI decay profile:")
    for lag in MI_LAGS:
        mi = compute_mutual_information(sequence, lag)
        mi_profile[lag] = mi
        print(f"  MI(lag={lag:4d}) = {mi:.4f} bits")

    return {
        "n_chars": n_chars,
        "alphabet_size": alphabet_size,
        "h0": round(h0, 6),
        "h0_max": round(h0_max, 6),
        "h2": round(h2, 6),
        "h3": round(h3, 6),
        "structure_score": round(structure_score, 6),
        "sequential_score": round(sequential_score, 6),
        "shuffled_control": {
            "h0": round(h0_shuf, 6),
            "h3": round(h3_shuf, 6),
            "structure_score": round(ss_shuf, 6),
        },
        "mi_profile": {str(k): round(v, 6) for k, v in mi_profile.items()},
    }


def main():
    print("=" * 60)
    print("Satellite TLE Entropy Analysis")
    print("=" * 60)

    # ── Download data ─────────────────────────────────────────────────────────
    raw_text, tle_pairs, source_label = download_tle_data(DATA_DIR)
    n_tles = len(tle_pairs)
    print(f"\nTotal TLE pairs: {n_tles:,}")

    # ── Build raw-text character sequence ────────────────────────────────────
    full_sequence = build_character_sequence(tle_pairs)
    print(f"Full sequence length: {len(full_sequence):,} chars  "
          f"({140} chars/TLE × {n_tles:,} TLEs)")

    # Report alphabet
    raw_counts = report_alphabet(full_sequence[:10_000], "Raw TLE text (first 10k chars)")

    # Truncate if very large
    if len(full_sequence) > TARGET_CHARS:
        sequence_raw = full_sequence[:TARGET_CHARS]
        print(f"\nTruncated to {len(sequence_raw):,} chars for entropy computation")
    else:
        sequence_raw = full_sequence

    chars_used = len(sequence_raw)

    # ── Raw-text entropy pipeline ─────────────────────────────────────────────
    raw_results = run_entropy_pipeline(sequence_raw, "Raw TLE text")

    # ── Field-level encoding ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Extracting orbital element fields...")
    fields = parse_orbital_elements(tle_pairs)
    n_parsed = len(fields["inclination_deg"])
    print(f"Successfully parsed {n_parsed:,} TLEs")

    field_results = {}
    interleaved_parts = []
    for fname, values in fields.items():
        if len(values) < 10:
            print(f"  Skipping {fname}: too few values ({len(values)})")
            continue
        fseq = discretise_field(values, N_BINS)
        print(f"\nField: {fname}  ({len(values):,} values -> {len(fseq):,} chars)")
        interleaved_parts.append(fseq)

        counts = Counter(fseq)
        alph = len(counts)
        print(f"  Alphabet size: {alph}, top bins: {counts.most_common(5)}")

        h0_f = compute_entropy_miller_madow(compute_ngram_counts(fseq, 1))
        h3_f = compute_conditional_entropy(fseq, order=3)
        ss_f = max(0.0, 1 - h3_f / h0_f) if h0_f > 0 else 0.0
        print(f"  H0={h0_f:.4f}  H3={h3_f:.4f}  structure_score={ss_f:.4f}")
        field_results[fname] = {
            "n_values": len(values),
            "alphabet_size": alph,
            "h0": round(h0_f, 6),
            "h3": round(h3_f, 6),
            "structure_score": round(ss_f, 6),
        }

    # Per-field mean structure score
    if field_results:
        per_field_scores = [v["structure_score"] for v in field_results.values()]
        mean_field_score = float(np.mean(per_field_scores))
        print(f"\nPer-field mean structure score: {mean_field_score:.4f}")

    # ── Assemble results ──────────────────────────────────────────────────────
    results = {
        "domain": "Satellite orbital telemetry (TLE format)",
        "tier": "B",
        "data_source": DATA_SOURCE,
        "encoding_notes": ENCODING_NOTES,
        "source_label": source_label,
        "n_tle_pairs": n_tles,
        "chars_per_tle": 140,
        "total_chars_available": len(full_sequence),
        "chars_analysed": chars_used,

        # Primary: raw TLE text analysis
        "raw_text": raw_results,

        # Secondary: per-field analysis
        "field_encoding": {
            "n_bins": N_BINS,
            "n_tles_parsed": n_parsed,
            "fields": field_results,
            "per_field_mean_structure_score": round(mean_field_score, 6) if field_results else None,
        },

        # Top-level summary keys (canonical for comparison table)
        "structure_score": raw_results["structure_score"],
        "sequential_score": raw_results["sequential_score"],
        "h0": raw_results["h0"],
        "h2": raw_results["h2"],
        "h3": raw_results["h3"],
        "n_chars": chars_used,
        "alphabet_size": raw_results["alphabet_size"],
    }

    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {RESULTS_FILE}")

    # ── Final summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SATELLITE TLE ENTROPY SUMMARY")
    print("=" * 60)
    print(f"  TLE pairs:       {n_tles:,}")
    print(f"  Chars analysed:  {chars_used:,}  ({140} chars/TLE)")
    print(f"  Alphabet:        {raw_results['alphabet_size']} chars "
          f"(digits + letters + special)")
    print(f"  H0:              {raw_results['h0']:.4f} bits  "
          f"(max {raw_results['h0_max']:.4f})")
    print(f"  H2:              {raw_results['h2']:.4f} bits")
    print(f"  H3:              {raw_results['h3']:.4f} bits")
    print(f"  Structure score: {raw_results['structure_score']:.4f}  (1 - H3/H0)")
    print(f"  Sequential score:{raw_results['sequential_score']:.4f}")
    if field_results:
        print(f"  Field mean score:{mean_field_score:.4f}  (per orbital param)")
    print("=" * 60)


if __name__ == "__main__":
    main()
