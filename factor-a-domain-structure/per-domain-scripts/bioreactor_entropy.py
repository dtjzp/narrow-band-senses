"""Bioreactor process log Shannon entropy analysis.

Data source: Zenodo record 15630354
"Real-time online monitoring of large-scale tubular photobioreactors
through gas-phase composition analysis"
DOI: 10.5281/zenodo.15630354

Sensors (1-minute resolution): DO, pH, FAir, FCO2, Pcomp, Pout, Tcomp,
Fg, Tlin, PAR, COG, CCG, RHrecirc, T_RHrecirc, FreshMediaFlow,
RecycleMediaFlow, Biomass

Strategy:
  - Load both raw data files (file 1: 23,243 rows; file 2: 4,321 rows)
  - Select key process channels: DO, pH, Tlin (temperature), COG (gas),
    CCG (gas), Biomass, PAR
  - Discretise each channel to 20 bins (matching our standard)
  - Encode each time-step as a multi-character token (one char per channel)
    and concatenate across time  → single stream of length
    n_timesteps × n_channels
  - Run full NBS entropy pipeline

Output: 1-research/nbs-survey/results/bioreactor_entropy.json
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]  # speaking-opportunities/
DATA_DIR = REPO_ROOT / "1-research" / "nbs-survey" / "data" / "bioreactor"
RESULTS_DIR = REPO_ROOT / "1-research" / "nbs-survey" / "results"
OUT_PATH = RESULTS_DIR / "bioreactor_entropy.json"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Entropy module
sys.path.insert(0, str(REPO_ROOT / "1-research" / "nbs-experiment"))
from entropy import (
    compute_ngram_counts,
    compute_entropy_miller_madow,
    compute_conditional_entropy,
    compute_structure_score,
    compute_sequential_score,
    shuffle_control,
)

# Try to import MI decay profile
try:
    from entropy import mi_decay_profile as _mi_decay_profile

    def mi_decay_profile(seq, lags):
        return _mi_decay_profile(seq, lags)
except ImportError:
    # Fallback: compute MI manually
    def mutual_information_lag(seq: str, lag: int) -> float:
        """I(X_t ; X_{t+lag}) via joint/marginal histograms."""
        n = len(seq)
        if n < lag + 10:
            return 0.0
        joint: dict[tuple, int] = {}
        for i in range(n - lag):
            pair = (seq[i], seq[i + lag])
            joint[pair] = joint.get(pair, 0) + 1
        # marginals
        px: dict[str, int] = {}
        py: dict[str, int] = {}
        total = sum(joint.values())
        for (x, y), c in joint.items():
            px[x] = px.get(x, 0) + c
            py[y] = py.get(y, 0) + c
        # I(X;Y) = H(X) + H(Y) - H(X,Y)
        def _h(d):
            n = sum(d.values())
            return sum(-c / n * math.log2(c / n) for c in d.values() if c > 0)
        return max(0.0, _h(px) + _h(py) - _h(joint))

    def mi_decay_profile(seq, lags):
        return {lag: mutual_information_lag(seq, lag) for lag in lags}

# ---------------------------------------------------------------------------
# Encoding helpers
# ---------------------------------------------------------------------------
N_BINS = 20
# ASCII printable range: 65-84 → 'A'..'T' (20 chars, safe for any text tool)
BIN_CHARS = "ABCDEFGHIJKLMNOPQRST"


def discretise(values: np.ndarray, n_bins: int = N_BINS) -> str:
    """Map a 1-D float array to a string using uniform quantile binning.

    Uses quantile (rank-based) bins so each bin has equal occupancy —
    consistent with the NBS survey standard approach.
    """
    valid = values[~np.isnan(values)]
    if len(valid) == 0:
        return ""
    # Quantile edges
    quantiles = np.linspace(0, 100, n_bins + 1)
    edges = np.percentile(valid, quantiles)
    # Ensure edges are strictly increasing (handles ties / constant signals)
    edges = np.unique(edges)
    n_actual = len(edges) - 1
    if n_actual < 1:
        # Constant signal — map all to 'A'
        return "A" * len(values)

    chars = []
    for v in values:
        if np.isnan(v):
            chars.append("A")  # NaN → first bin
        else:
            idx = int(np.searchsorted(edges, v, side="right")) - 1
            idx = max(0, min(n_actual - 1, idx))
            chars.append(BIN_CHARS[idx % len(BIN_CHARS)])
    return "".join(chars)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_csv(path: Path) -> dict[str, np.ndarray]:
    """Load CSV, return dict of column_name → np.ndarray."""
    import csv
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    if not rows:
        return {}
    cols = list(rows[0].keys())
    data = {}
    for col in cols:
        vals = []
        for row in rows:
            raw = row[col].strip()
            try:
                vals.append(float(raw))
            except ValueError:
                vals.append(float("nan"))
        data[col] = np.array(vals, dtype=float)
    return data


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    t0 = time.time()
    print("=" * 60)
    print("Bioreactor Process Log Entropy Analysis")
    print("=" * 60)

    # ── Load files ─────────────────────────────────────────────────────────
    print("\n[1] Loading data files ...")
    files = [
        DATA_DIR / "GA_RawData_1.txt",
        DATA_DIR / "GA_RawData_2.txt",
    ]

    all_data: list[dict[str, np.ndarray]] = []
    for fpath in files:
        d = load_csv(fpath)
        print(f"    {fpath.name}: {len(next(iter(d.values()))):,} rows, columns: {list(d.keys())}")
        all_data.append(d)

    # ── Select key sensor channels ─────────────────────────────────────────
    # Channels present in both files (intersection)
    common = set(all_data[0].keys()) & set(all_data[1].keys())
    # Prefer: DO, pH, Tlin, COG, CCG, Biomass  (all are core bioreactor signals)
    preferred = ["DO", "pH", "Tlin", "COG", "CCG", "Biomass"]
    channels = [c for c in preferred if c in common]
    # Add any remaining common channels that weren't preferred
    for c in sorted(common):
        if c not in channels and c != "Date":
            channels.append(c)

    print(f"\n    Common channels: {sorted(common)}")
    print(f"    Selected for analysis: {channels}")

    # ── Concatenate across files per channel ───────────────────────────────
    channel_data: dict[str, np.ndarray] = {}
    for ch in channels:
        parts = []
        for d in all_data:
            if ch in d:
                parts.append(d[ch])
        channel_data[ch] = np.concatenate(parts)

    n_timesteps = len(next(iter(channel_data.values())))
    print(f"\n    Total timesteps: {n_timesteps:,} (~{n_timesteps/60:.0f} hours = {n_timesteps/1440:.1f} days)")

    # ── Discretise and encode ──────────────────────────────────────────────
    print("\n[2] Discretising channels to 20 bins ...")
    channel_streams: dict[str, str] = {}
    for ch in channels:
        arr = channel_data[ch]
        s = discretise(arr, N_BINS)
        channel_streams[ch] = s
        unique_bins = len(set(s))
        print(f"    {ch:20s}: {unique_bins:2d} distinct bins, "
              f"range [{np.nanmin(arr):.3g}, {np.nanmax(arr):.3g}]")

    # ── Build interleaved stream ────────────────────────────────────────────
    # Strategy: interleave channels — each timestep contributes len(channels) chars
    # This preserves temporal order AND cross-channel correlations
    print(f"\n[3] Building interleaved stream ({n_timesteps:,} timesteps × {len(channels)} channels) ...")
    interleaved_parts = []
    for t in range(n_timesteps):
        for ch in channels:
            interleaved_parts.append(channel_streams[ch][t])
    stream = "".join(interleaved_parts)
    n_chars = len(stream)
    print(f"    Stream length: {n_chars:,} chars")

    # ── Alphabet report ────────────────────────────────────────────────────
    alphabet = sorted(set(stream))
    print(f"    Alphabet: {len(alphabet)} symbols: {''.join(alphabet)}")

    # ── Entropy metrics ────────────────────────────────────────────────────
    print("\n[4] Computing entropy metrics ...")

    uni_counts = compute_ngram_counts(stream, 1)
    h0 = compute_entropy_miller_madow(uni_counts)
    print(f"    H0 (unigram):    {h0:.4f} bits  (max log2({len(alphabet)}) = {math.log2(len(alphabet)):.4f})")

    h2 = compute_conditional_entropy(stream, order=2)
    print(f"    H(X|1-gram):     {h2:.4f} bits")

    h3 = compute_conditional_entropy(stream, order=3)
    print(f"    H(X|2-gram):     {h3:.4f} bits")

    h4 = compute_conditional_entropy(stream, order=4)
    print(f"    H(X|3-gram):     {h4:.4f} bits")

    struct_score = compute_structure_score(stream)
    print(f"    Structure score: {struct_score:.4f}  (1 - H3/H0)")

    seq_score = compute_sequential_score(stream)
    print(f"    Sequential score:{seq_score:.4f}  (1 - H3/H2_shuffled)")

    # ── Shuffled control ────────────────────────────────────────────────────
    print("\n[5] Computing shuffled control ...")
    shuffled = shuffle_control(stream, seed=42)
    uni_sh = compute_ngram_counts(shuffled, 1)
    h0_sh = compute_entropy_miller_madow(uni_sh)
    h3_sh = compute_conditional_entropy(shuffled, order=3)
    struct_sh = max(0.0, 1 - h3_sh / h0_sh) if h0_sh > 0 else 0.0
    print(f"    Shuffled H0:     {h0_sh:.4f} bits")
    print(f"    Shuffled H3:     {h3_sh:.4f} bits")
    print(f"    Shuffled struct: {struct_sh:.4f}")

    # ── MI decay ───────────────────────────────────────────────────────────
    print("\n[6] Computing MI decay profile ...")
    MI_LAGS = [1, 2, 5, 10, 50, 100, 500]
    mi_profile = mi_decay_profile(stream, MI_LAGS)
    for lag, mi in mi_profile.items():
        print(f"    MI(lag={lag:5d}): {mi:.4f} bits")

    # ── Per-channel breakdown ───────────────────────────────────────────────
    print("\n[7] Per-channel entropy breakdown ...")
    per_channel = {}
    for ch in channels:
        s = channel_streams[ch]
        uc = compute_ngram_counts(s, 1)
        ch_h0 = compute_entropy_miller_madow(uc)
        ch_h3 = compute_conditional_entropy(s, order=3) if len(s) >= 10 else 0.0
        ch_struct = max(0.0, 1 - ch_h3 / ch_h0) if ch_h0 > 0 else 0.0
        print(f"    {ch:20s}: H0={ch_h0:.3f}, H3={ch_h3:.3f}, struct={ch_struct:.3f}")
        per_channel[ch] = {
            "H0": round(ch_h0, 6),
            "H3": round(ch_h3, 6),
            "structure_score": round(ch_struct, 6),
        }

    # ── Per-channel mean structure score ───────────────────────────────────────
    ch_struct_scores = [per_channel[ch]["structure_score"] for ch in channels]
    mean_ch_struct = float(np.mean(ch_struct_scores))
    median_ch_struct = float(np.median(ch_struct_scores))
    print(f"\n    Per-channel mean structure score:   {mean_ch_struct:.4f}")
    print(f"    Per-channel median structure score: {median_ch_struct:.4f}")
    print(f"    Interleaved stream struct score:    {struct_score:.4f}")

    # ── Assemble results ────────────────────────────────────────────────────
    elapsed = time.time() - t0
    results = {
        "status": "ok",
        "domain": "bioreactor process log",
        "data_source": (
            "Zenodo record 15630354 — 'Real-time online monitoring of large-scale "
            "tubular photobioreactors through gas-phase composition analysis' "
            "(DOI: 10.5281/zenodo.15630354). Two campaigns: Oct 2024 (4321 min) and "
            "Dec 2024 (23243 min). Sensors: DO, pH, temperature, gas flow, CO2, "
            "biomass, PAR, humidity. 1-minute sampling interval."
        ),
        "encoding": (
            "Each timestep encoded as N-character token "
            "(one char per sensor channel, 20 quantile bins A-T). "
            "Channels interleaved across time. "
            "NOTE: interleaved stream structure score is lower than per-channel "
            "because cross-sensor character transitions break within-sensor autocorr. "
            "Per-channel scores are the physically meaningful primary result."
        ),
        "primary_structure_score": round(mean_ch_struct, 6),
        "primary_structure_score_note": (
            "Mean of per-channel structure scores (physically meaningful). "
            "Interleaved stream score is reduced by cross-sensor transitions."
        ),
        "channels": channels,
        "n_timesteps": n_timesteps,
        "n_chars": n_chars,
        "n_bins": N_BINS,
        "alphabet_size": len(alphabet),
        "alphabet": "".join(alphabet),
        "entropy": {
            "H0_unigram_bits": round(h0, 6),
            "H_cond_bigram_bits": round(h2, 6),
            "H_cond_trigram_bits": round(h3, 6),
            "H_cond_4gram_bits": round(h4, 6),
            "structure_score_interleaved": round(struct_score, 6),
            "sequential_score_interleaved": round(seq_score, 6),
            "structure_score_per_channel_mean": round(mean_ch_struct, 6),
            "structure_score_per_channel_median": round(median_ch_struct, 6),
            "max_entropy_bits": round(math.log2(len(alphabet)), 6),
            "H0_over_max": round(h0 / math.log2(len(alphabet)), 6),
        },
        "shuffled_control": {
            "H0_bits": round(h0_sh, 6),
            "H3_bits": round(h3_sh, 6),
            "structure_score": round(struct_sh, 6),
        },
        "mi_decay": {str(k): round(v, 6) for k, v in mi_profile.items()},
        "per_channel": per_channel,
        "elapsed_seconds": round(elapsed, 1),
    }

    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    # ── Final summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("BIOREACTOR ENTROPY SUMMARY")
    print("=" * 60)
    print(f"  Data:             {n_timesteps:,} timesteps x {len(channels)} channels")
    print(f"  Stream length:    {n_chars:,} chars")
    print(f"  Alphabet size:    {len(alphabet)}")
    print(f"  H0 (unigram):     {h0:.4f} bits  (max {math.log2(len(alphabet)):.4f})")
    print(f"  H(X|bigram):      {h2:.4f} bits")
    print(f"  H(X|trigram):     {h3:.4f} bits")
    print(f"  Structure score (interleaved): {struct_score:.4f}")
    print(f"  Structure score (ch mean):     {mean_ch_struct:.4f}  <<< primary result")
    print(f"  Sequential score (interleaved):{seq_score:.4f}")
    print()
    print("  COMPARISON (NBS survey benchmarks):")
    print(f"    Weather:  struct=0.201  (expect bioreactor > this)")
    print(f"    Bioreact: struct={mean_ch_struct:.3f}  (per-channel mean)")
    print(f"    Tidal:    struct=0.658  (expect bioreactor < this)")
    print()
    if mean_ch_struct > 0.658:
        verdict = "HIGHER than tidal (more constrained than expected)"
    elif mean_ch_struct > 0.201:
        verdict = "MODERATE-HIGH (as expected: weather < bioreactor < tidal)"
    elif mean_ch_struct > 0.10:
        verdict = "LOW-MODERATE (less structured than weather)"
    else:
        verdict = "LOW (near-random)"
    print(f"  VERDICT: {verdict}")
    print(f"  Results saved to: {OUT_PATH}")
    print(f"  Elapsed: {elapsed:.1f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
