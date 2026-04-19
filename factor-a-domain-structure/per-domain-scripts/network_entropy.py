"""Network security log entropy analysis — CICIDS2017 Monday-WorkingHours.

Steps:
1. Load CSV, extract key features.
2. Encode each flow as a short character string (protocol + port_bin + duration_bin + count_bin + flags).
3. Concatenate flows in order → continuous character stream.
4. Run full entropy analysis using the NBS entropy module.
5. Save results to results/network_entropy.json and print summary.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]  # speaking-opportunities/
DATA_DIR = REPO_ROOT / "1-research" / "nbs-survey" / "data" / "network"
RESULTS_DIR = REPO_ROOT / "1-research" / "nbs-survey" / "results"
CSV_PATH = DATA_DIR / "Monday-WorkingHours.pcap_ISCX.csv"
OUT_PATH = RESULTS_DIR / "network_entropy.json"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Add entropy module to path
sys.path.insert(0, str(REPO_ROOT / "1-research" / "nbs-experiment"))
from entropy import (
    compute_ngram_counts,
    compute_entropy_miller_madow,
    compute_conditional_entropy,
    compute_structure_score,
    compute_sequential_score,
    compute_structure_scores_bootstrap,
    mi_decay_profile,
    shuffle_control,
)


# ---------------------------------------------------------------------------
# Feature extraction helpers
# ---------------------------------------------------------------------------

# Protocol encoding: CICIDS2017 uses numeric protocol codes.
# Common codes: 6=TCP, 17=UDP, 1=ICMP, 0=HOPOPT, 58=IPv6-ICMP
PROTO_MAP = {6: "T", 17: "U", 1: "I", 0: "H", 58: "V"}

# Port bins (10 bins by well-known service ranges):
# 0: 0-79, 1: 80-443, 2: 444-1023, 3: 1024-4999, 4: 5000-9999,
# 5: 10000-19999, 6: 20000-29999, 7: 30000-39999, 8: 40000-49999, 9: 50000+
def port_bin(port: float) -> str:
    if pd.isna(port):
        return "9"
    p = int(port)
    if p <= 79:
        return "0"
    elif p <= 443:
        return "1"
    elif p <= 1023:
        return "2"
    elif p <= 4999:
        return "3"
    elif p <= 9999:
        return "4"
    elif p <= 19999:
        return "5"
    elif p <= 29999:
        return "6"
    elif p <= 39999:
        return "7"
    elif p <= 49999:
        return "8"
    else:
        return "9"


# Duration bins (microseconds, log-scale 10 bins):
# Uses log10 bins: <1, 1-10, 10-100, 100-1000, 1000-10000, ..., >1e9
def duration_bin(dur: float) -> str:
    if pd.isna(dur) or dur <= 0:
        return "0"
    log = math.log10(max(1, dur))
    # 0-9 mapped to log intervals 0..9 (0=<1us, 9=>1e9us=1000s)
    idx = int(min(9, log))
    return str(idx)


# Packet count bins (5 bins):
# 0: 1-2, 1: 3-5, 2: 6-20, 3: 21-100, 4: >100
def count_bin(cnt: float) -> str:
    if pd.isna(cnt) or cnt <= 0:
        return "0"
    c = int(cnt)
    if c <= 2:
        return "0"
    elif c <= 5:
        return "1"
    elif c <= 20:
        return "2"
    elif c <= 100:
        return "3"
    else:
        return "4"


# Flag encoding: SYN/ACK/FIN/RST presence → single char
# Encode the 4-bit pattern SYN|ACK|FIN|RST as a hex digit 0-F
def flag_char(syn: float, ack: float, fin: float, rst: float) -> str:
    bits = (
        (1 if (not pd.isna(syn) and syn > 0) else 0) << 3
        | (1 if (not pd.isna(ack) and ack > 0) else 0) << 2
        | (1 if (not pd.isna(fin) and fin > 0) else 0) << 1
        | (1 if (not pd.isna(rst) and rst > 0) else 0)
    )
    return format(bits, "X")  # '0'..'F'


def encode_flows(df: pd.DataFrame) -> str:
    """Encode each flow row as a 5-char string; return concatenated stream."""
    chars = []
    for _, row in df.iterrows():
        proto_raw = row.get("Protocol", 6)
        proto_c = PROTO_MAP.get(int(proto_raw) if not pd.isna(proto_raw) else 6, "X")
        pb = port_bin(row.get(" Destination Port", row.get("Destination Port", 0)))
        db = duration_bin(row.get(" Flow Duration", row.get("Flow Duration", 0)))
        cb = count_bin(
            (row.get(" Total Fwd Packets", row.get("Total Fwd Packets", 0)) or 0)
            + (row.get(" Total Backward Packets", row.get("Total Backward Packets", 0)) or 0)
        )
        fc = flag_char(
            row.get(" SYN Flag Count", row.get("SYN Flag Count", 0)),
            row.get(" ACK Flag Count", row.get("ACK Flag Count", 0)),
            row.get(" FIN Flag Count", row.get("FIN Flag Count", 0)),
            row.get(" RST Flag Count", row.get("RST Flag Count", 0)),
        )
        chars.append(proto_c + pb + db + cb + fc)
    return "".join(chars)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    t0 = time.time()
    print(f"[1] Loading {CSV_PATH.name} ...")
    df = pd.read_csv(CSV_PATH, low_memory=False)
    print(f"    Loaded {len(df):,} rows, {len(df.columns)} columns.")

    # Show label distribution
    label_col = " Label" if " Label" in df.columns else "Label"
    print(f"    Labels: {df[label_col].value_counts().to_dict()}")

    print("[2] Encoding flows as character stream ...")
    # Normalise column names (strip leading spaces for lookup)
    df.columns = [c.strip() for c in df.columns]

    stream_parts = []
    for _, row in df.iterrows():
        proto_raw = row.get("Protocol", 6)
        proto_c = PROTO_MAP.get(int(proto_raw) if not pd.isna(proto_raw) else 6, "X")
        pb = port_bin(row.get("Destination Port", 0))
        db = duration_bin(row.get("Flow Duration", 0))
        cb = count_bin(
            (row.get("Total Fwd Packets", 0) or 0)
            + (row.get("Total Backward Packets", 0) or 0)
        )
        fc = flag_char(
            row.get("SYN Flag Count", 0),
            row.get("ACK Flag Count", 0),
            row.get("FIN Flag Count", 0),
            row.get("RST Flag Count", 0),
        )
        stream_parts.append(proto_c + pb + db + cb + fc)

    stream = "".join(stream_parts)
    n_chars = len(stream)
    n_flows = len(df)
    chars_per_flow = n_chars // n_flows
    print(f"    Stream length: {n_chars:,} chars ({n_flows:,} flows × {chars_per_flow} chars/flow)")

    # Alphabet
    alphabet = sorted(set(stream))
    print(f"    Alphabet size: {len(alphabet)} unique chars")

    print("[3] Computing entropy metrics ...")

    # Unigram (H0)
    uni_counts = compute_ngram_counts(stream, 1)
    h0 = compute_entropy_miller_madow(uni_counts)
    print(f"    H0 (unigram): {h0:.4f} bits")

    # Bigram conditional entropy H(X|1-gram context)
    h2 = compute_conditional_entropy(stream, order=2)
    print(f"    H(X|1-gram): {h2:.4f} bits")

    # Trigram conditional entropy H(X|2-gram context)
    h3 = compute_conditional_entropy(stream, order=3)
    print(f"    H(X|2-gram): {h3:.4f} bits")

    # 4-gram
    h4 = compute_conditional_entropy(stream, order=4)
    print(f"    H(X|3-gram): {h4:.4f} bits")

    # Structure score
    struct_score = compute_structure_score(stream)
    print(f"    Structure score: {struct_score:.4f}")

    # Sequential score
    seq_score = compute_sequential_score(stream)
    print(f"    Sequential score: {seq_score:.4f}")

    # Shuffled baseline H0 (should equal H0 — sanity check)
    shuffled = shuffle_control(list(stream))
    shuffled_h0 = compute_entropy_miller_madow(compute_ngram_counts(shuffled, 1))
    shuffled_h2 = compute_conditional_entropy(shuffled, order=2)
    print(f"    Shuffled H0: {shuffled_h0:.4f} bits (should ~= H0)")
    print(f"    Shuffled H(X|1-gram): {shuffled_h2:.4f} bits")

    # MI decay
    print("    Computing MI decay profile ...")
    mi_profile = mi_decay_profile(stream, lags=[1, 2, 5, 10, 50, 100])
    for lag, mi in mi_profile.items():
        print(f"      MI(lag={lag}): {mi:.4f} bits")

    # Bootstrap CIs (reduced n for speed on long stream)
    print("    Bootstrap CIs (100 samples, 20% subsamples) ...")
    # Use 20% subsamples to keep computation fast on 2.6M chars
    boot = compute_structure_scores_bootstrap(
        stream,
        n_bootstrap=100,
        subsample_frac=0.20,
        seed=42,
    )
    print(f"    Structure score: {boot['mean']:.4f} [{boot['ci_low']:.4f}, {boot['ci_high']:.4f}]")
    print(f"    Sequential score: {boot['seq_mean']:.4f} [{boot['seq_ci_low']:.4f}, {boot['seq_ci_high']:.4f}]")

    # Per-protocol breakdown
    print("[4] Per-protocol entropy breakdown ...")
    proto_streams: dict[str, str] = {}
    for proto_char in ["T", "U", "I", "H", "V", "X"]:
        # Extract characters belonging to flows starting with this proto char
        proto_flow_indices = [i for i, p in enumerate(stream_parts) if p[0] == proto_char]
        if proto_flow_indices:
            sub = "".join(stream_parts[i] for i in proto_flow_indices)
            proto_streams[proto_char] = sub
            sub_h0 = compute_entropy_miller_madow(compute_ngram_counts(sub, 1))
            sub_struct = compute_structure_score(sub) if len(sub) >= 10 else 0.0
            print(f"    {proto_char}: {len(proto_flow_indices):,} flows, H0={sub_h0:.4f}, struct={sub_struct:.4f}")

    # Label-conditional: benign vs attack (if attacks present)
    label_counts = df["Label"].value_counts().to_dict()
    if len(label_counts) > 1:
        print("[5] Label-conditional entropy ...")
        for label, cnt in label_counts.items():
            label_indices = df.index[df["Label"] == label].tolist()
            sub = "".join(stream_parts[i] for i in label_indices if i < len(stream_parts))
            if len(sub) >= 10:
                sub_h0 = compute_entropy_miller_madow(compute_ngram_counts(sub, 1))
                sub_struct = compute_structure_score(sub)
                print(f"    {label}: n={cnt:,}, H0={sub_h0:.4f}, struct={sub_struct:.4f}")
    else:
        print("[5] All flows BENIGN (Monday is normal traffic day) — no per-label breakdown.")

    # NBS interpretation: bits-per-character at different model sizes
    # Approximate compressibility (lower = more structure)
    h_ratio_2 = h2 / h0 if h0 > 0 else 1.0
    h_ratio_3 = h3 / h0 if h0 > 0 else 1.0
    print(f"\n    Compression proxy H2/H0={h_ratio_2:.4f}, H3/H0={h_ratio_3:.4f}")

    elapsed = time.time() - t0

    # ---------------------------------------------------------------------------
    # Save results
    # ---------------------------------------------------------------------------
    results = {
        "dataset": "CICIDS2017",
        "file": "Monday-WorkingHours.pcap_ISCX.csv",
        "source": "https://huggingface.co/datasets/c01dsnap/CIC-IDS2017",
        "n_flows": n_flows,
        "stream_length": n_chars,
        "chars_per_flow": chars_per_flow,
        "alphabet_size": len(alphabet),
        "alphabet": "".join(alphabet),
        "label_distribution": {str(k): int(v) for k, v in label_counts.items()},
        "entropy": {
            "H0_unigram_bits": round(h0, 6),
            "H_cond_bigram_bits": round(h2, 6),
            "H_cond_trigram_bits": round(h3, 6),
            "H_cond_4gram_bits": round(h4, 6),
            "structure_score": round(struct_score, 6),
            "sequential_score": round(seq_score, 6),
            "shuffled_H0_bits": round(shuffled_h0, 6),
            "shuffled_H_cond_bigram_bits": round(shuffled_h2, 6),
            "compression_proxy_H2_over_H0": round(h_ratio_2, 6),
            "compression_proxy_H3_over_H0": round(h_ratio_3, 6),
        },
        "bootstrap": {
            "n_samples": 100,
            "subsample_frac": 0.20,
            "structure_score_mean": round(boot["mean"], 6),
            "structure_score_ci_low": round(boot["ci_low"], 6),
            "structure_score_ci_high": round(boot["ci_high"], 6),
            "sequential_score_mean": round(boot["seq_mean"], 6),
            "sequential_score_ci_low": round(boot["seq_ci_low"], 6),
            "sequential_score_ci_high": round(boot["seq_ci_high"], 6),
        },
        "mi_decay": {str(k): round(v, 6) for k, v in mi_profile.items()},
        "per_protocol": {},
        "elapsed_seconds": round(elapsed, 1),
    }

    for proto_char, sub in proto_streams.items():
        if len(sub) >= 10:
            sub_h0 = compute_entropy_miller_madow(compute_ngram_counts(sub, 1))
            sub_h2 = compute_conditional_entropy(sub, order=2) if len(sub) >= 4 else 0.0
            sub_struct = compute_structure_score(sub) if len(sub) >= 10 else 0.0
            results["per_protocol"][proto_char] = {
                "n_flows": len(proto_streams[proto_char]) // chars_per_flow,
                "H0_bits": round(sub_h0, 6),
                "H_cond_bigram_bits": round(sub_h2, 6),
                "structure_score": round(sub_struct, 6),
            }

    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n[6] Results saved to {OUT_PATH}")
    print(f"    Elapsed: {elapsed:.1f}s")

    # Final summary
    print("\n" + "=" * 60)
    print("NETWORK ENTROPY SUMMARY — CICIDS2017 Monday")
    print("=" * 60)
    print(f"  Flows:            {n_flows:,}")
    print(f"  Stream length:    {n_chars:,} chars")
    print(f"  Alphabet size:    {len(alphabet)}")
    print(f"  H0 (unigram):     {h0:.4f} bits")
    print(f"  H(X|bigram):      {h2:.4f} bits")
    print(f"  H(X|trigram):     {h3:.4f} bits")
    print(f"  Structure score:  {struct_score:.4f}  (1 = maximal structure)")
    print(f"  Sequential score: {seq_score:.4f}  (1 = maximal sequence order)")
    print(f"  Bootstrap struct: {boot['mean']:.4f} [{boot['ci_low']:.4f}, {boot['ci_high']:.4f}]")
    print(f"  Bootstrap seq:    {boot['seq_mean']:.4f} [{boot['seq_ci_low']:.4f}, {boot['seq_ci_high']:.4f}]")
    print()
    print("  INTERPRETATION:")
    print(f"    H0={h0:.2f} < log2({len(alphabet)})={math.log2(len(alphabet)):.2f}: alphabet not uniform (structured)")
    print(f"    Structure score {struct_score:.3f}: ", end="")
    if struct_score > 0.5:
        print("HIGH structure (grammar-like)")
    elif struct_score > 0.3:
        print("MODERATE structure")
    else:
        print("LOW structure (near-random)")
    print("=" * 60)


if __name__ == "__main__":
    main()
