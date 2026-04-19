"""Claim-verification scorer for MLP bridge — domain=financial, source=synthetic.

For each eval record, extract specific verifiable claims from the generated
description about bin-level patterns (counts, runs, alternation, half-to-half
drift), verify against the sequence+continuation digits, and compute a ceiling
score from ground-truth facts.

Outputs JSONL to G:/My Drive/nbs-bridge/results/mlp/financial_synthetic_scores.jsonl
"""

import json
import re
from collections import Counter
from pathlib import Path

IN_PATH = Path("G:/My Drive/nbs-bridge/results/mlp/financial_synthetic_eval_prepared.jsonl")
OUT_PATH = Path("G:/My Drive/nbs-bridge/results/mlp/financial_synthetic_scores.jsonl")

# Financial synthetic sequences are digit strings (0-9). Ground-truth facts
# are computed over sequence+continuation concatenated (the full window the
# model has access to when describing).


# --- ground-truth feature extractors --------------------------------------

def digits_only(tokens: str):
    return [int(c) for c in tokens if c.isdigit()]


def alternation_rate(digs) -> float:
    """Fraction of adjacent positions that differ."""
    if len(digs) < 2:
        return 0.0
    diffs = sum(1 for a, b in zip(digs[:-1], digs[1:]) if a != b)
    return diffs / (len(digs) - 1)


def bin_split_pct(digs):
    """(%low 0-2, %middle 3-5, %high 6-9) rounded to nearest integer."""
    n = len(digs)
    if n == 0:
        return (0, 0, 0)
    low = sum(1 for d in digs if d <= 2)
    mid = sum(1 for d in digs if 3 <= d <= 5)
    high = sum(1 for d in digs if d >= 6)
    return (round(100 * low / n), round(100 * mid / n), round(100 * high / n))


def top_two(digs):
    """Returns [(symbol, count), (symbol, count)] sorted by count desc then
    symbol asc. Useful for 'most frequent symbol is X with N, followed by Y
    with M' claims."""
    c = Counter(digs)
    ordered = sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))
    return ordered[:2]


def longest_run(digs):
    """Returns (symbol, length) for the longest unbroken run. Ties: first
    longest encountered."""
    if not digs:
        return (None, 0)
    best_sym, best_len = digs[0], 1
    cur_sym, cur_len = digs[0], 1
    for d in digs[1:]:
        if d == cur_sym:
            cur_len += 1
        else:
            cur_sym, cur_len = d, 1
        if cur_len > best_len:
            best_sym, best_len = cur_sym, cur_len
    return (best_sym, best_len)


def half_means(digs):
    """Returns (mean_first_half, mean_second_half) of the digit stream."""
    n = len(digs)
    if n < 2:
        return (0.0, 0.0)
    mid = n // 2
    a = digs[:mid]
    b = digs[mid:]
    return (sum(a) / len(a), sum(b) / len(b))


# --- claim extraction ------------------------------------------------------

# Approx tolerance for noisy human-style percentage / rate statements.
PCT_TOL = 3  # ±3 percentage points
RATE_TOL = 0.03
MEAN_TOL = 0.10


def extract_alternation_claim(desc: str):
    """Claim: 'Alternation rate is X' (e.g. 0.88)."""
    m = re.search(r"[Aa]lternation rate is\s+([0-9]+(?:\.[0-9]+)?)", desc)
    if not m:
        return None
    return ("alternation_rate", float(m.group(1)))


def extract_bin_split_claim(desc: str):
    """Claim: 'the split is roughly 36% / 24% / 36%' — may have extra values
    from hallucinations; we take the FIRST three."""
    m = re.search(
        r"split is roughly\s+(\d+)%\s*/\s*(\d+)%\s*/\s*(\d+)%",
        desc,
    )
    if not m:
        return None
    return ("bin_split", (int(m.group(1)), int(m.group(2)), int(m.group(3))))


def extract_top1_claim(desc: str):
    """Claim: "most frequent symbol is 'X' with N occurrences"."""
    m = re.search(
        r"most frequent symbol is\s+['\"]?(\d)['\"]?\s+with\s+(\d+)\s+occurrences",
        desc,
    )
    if not m:
        return None
    return ("top1", (int(m.group(1)), int(m.group(2))))


def extract_top2_claim(desc: str):
    """Claim: "followed by 'Y' with M"."""
    m = re.search(
        r"followed by\s+['\"]?(\d)['\"]?\s+with\s+(\d+)",
        desc,
    )
    if not m:
        return None
    return ("top2", (int(m.group(1)), int(m.group(2))))


def extract_longest_run_claim(desc: str):
    """Claim: "longest unbroken run is 'X' repeated N times"."""
    m = re.search(
        r"longest unbroken run is\s+['\"]?(\d)['\"]?\s+repeated\s+(\d+)\s+times?",
        desc,
    )
    if not m:
        return None
    return ("longest_run", (int(m.group(1)), int(m.group(2))))


def extract_half_drift_claim(desc: str):
    """Claim: "drift from A to B" or "First- and second-half means ... (A vs B)"."""
    m = re.search(r"drift from\s+([0-9]+\.[0-9]+)\s+to\s+([0-9]+\.[0-9]+)", desc)
    if m:
        return ("half_means", (float(m.group(1)), float(m.group(2))))
    m = re.search(
        r"half means[^(]*\(([0-9]+\.[0-9]+)\s+vs\s+([0-9]+\.[0-9]+)\)",
        desc,
    )
    if m:
        return ("half_means", (float(m.group(1)), float(m.group(2))))
    return None


CLAIM_EXTRACTORS = [
    extract_alternation_claim,
    extract_bin_split_claim,
    extract_top1_claim,
    extract_top2_claim,
    extract_longest_run_claim,
    extract_half_drift_claim,
]


def extract_claims(desc: str):
    out = []
    for fn in CLAIM_EXTRACTORS:
        c = fn(desc)
        if c is not None:
            out.append(c)
    return out


# --- verification ----------------------------------------------------------

def verify(label, claimed, digs):
    if label == "alternation_rate":
        actual = alternation_rate(digs)
        if abs(actual - claimed) <= RATE_TOL:
            return "CONFIRMED", actual
        return "REFUTED", actual
    if label == "bin_split":
        actual = bin_split_pct(digs)
        if all(abs(a - c) <= PCT_TOL for a, c in zip(actual, claimed)):
            return "CONFIRMED", actual
        return "REFUTED", actual
    if label == "top1":
        top = top_two(digs)
        if not top:
            return "UNVERIFIABLE", None
        actual_sym, actual_cnt = top[0]
        sym_c, cnt_c = claimed
        # Confirm if symbol is the actual top AND count is within 1
        if sym_c == actual_sym and abs(actual_cnt - cnt_c) <= 1:
            return "CONFIRMED", (actual_sym, actual_cnt)
        return "REFUTED", (actual_sym, actual_cnt)
    if label == "top2":
        top = top_two(digs)
        if len(top) < 2:
            return "UNVERIFIABLE", None
        actual_sym, actual_cnt = top[1]
        sym_c, cnt_c = claimed
        if sym_c == actual_sym and abs(actual_cnt - cnt_c) <= 1:
            return "CONFIRMED", (actual_sym, actual_cnt)
        return "REFUTED", (actual_sym, actual_cnt)
    if label == "longest_run":
        actual_sym, actual_len = longest_run(digs)
        sym_c, len_c = claimed
        if sym_c == actual_sym and len_c == actual_len:
            return "CONFIRMED", (actual_sym, actual_len)
        # Accept if the claimed length matches actual length (symbol might
        # tie) — length is the primary fact
        if len_c == actual_len:
            return "CONFIRMED", (actual_sym, actual_len)
        return "REFUTED", (actual_sym, actual_len)
    if label == "half_means":
        a_actual, b_actual = half_means(digs)
        a_c, b_c = claimed
        if abs(a_actual - a_c) <= MEAN_TOL and abs(b_actual - b_c) <= MEAN_TOL:
            return "CONFIRMED", (a_actual, b_actual)
        return "REFUTED", (a_actual, b_actual)
    return "UNVERIFIABLE", None


# --- ceiling facts ---------------------------------------------------------

def ceiling_facts(digs):
    """Six independent ground-truth facts derived directly from digs. These
    are tautologically confirmed against the same source, so ceiling_score
    should be 1.0 — this is the instrument's own upper bound."""
    top = top_two(digs)
    lr = longest_run(digs)
    hm = half_means(digs)
    bs = bin_split_pct(digs)
    facts = [
        ("alternation_rate", round(alternation_rate(digs), 2)),
        ("bin_split", bs),
        ("top1", top[0] if top else None),
        ("top2", top[1] if len(top) > 1 else None),
        ("longest_run", lr),
        ("half_means", (round(hm[0], 2), round(hm[1], 2))),
    ]
    return facts


# --- main ------------------------------------------------------------------

def main():
    records = []
    with IN_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as out:
        for rec in records:
            seq_idx = rec["seq_idx"]
            tokens = rec["sequence"] + rec["continuation"]
            digs = digits_only(tokens)
            gen = rec["generated_description"]

            claims = extract_claims(gen)
            # Clamp to 3-6 (already <=6 by construction)
            claims = claims[:6]
            n_claims = len(claims)
            n_conf = n_ref = n_unv = 0
            for label, claimed in claims:
                verdict, _actual = verify(label, claimed, digs)
                if verdict == "CONFIRMED":
                    n_conf += 1
                elif verdict == "REFUTED":
                    n_ref += 1
                else:
                    n_unv += 1

            bridge_score = (n_conf / n_claims) if n_claims > 0 else 0.0

            # Ceiling: 6 independent facts computed from ground truth
            facts = ceiling_facts(digs)
            ceiling_n = len(facts)
            ceiling_conf = ceiling_n  # by construction — all derived facts confirm
            ceiling_score = ceiling_conf / ceiling_n if ceiling_n else 0.0

            out_rec = {
                "seq_idx": seq_idx,
                "n_claims": n_claims,
                "n_confirmed": n_conf,
                "n_refuted": n_ref,
                "n_unverifiable": n_unv,
                "bridge_score": round(bridge_score, 4),
                "ceiling_n_claims": ceiling_n,
                "ceiling_n_confirmed": ceiling_conf,
                "ceiling_score": round(ceiling_score, 4),
            }
            out.write(json.dumps(out_rec) + "\n")

    # Summary
    with OUT_PATH.open("r", encoding="utf-8") as f:
        rows = [json.loads(l) for l in f if l.strip()]
    n = len(rows)
    avg_bridge = sum(r["bridge_score"] for r in rows) / n if n else 0
    avg_ceil = sum(r["ceiling_score"] for r in rows) / n if n else 0
    total_claims = sum(r["n_claims"] for r in rows)
    total_conf = sum(r["n_confirmed"] for r in rows)
    print(f"Wrote {n} records to {OUT_PATH}")
    print(f"Avg bridge_score  = {avg_bridge:.3f}  (total confirmed {total_conf}/{total_claims})")
    print(f"Avg ceiling_score = {avg_ceil:.3f}")


if __name__ == "__main__":
    main()
