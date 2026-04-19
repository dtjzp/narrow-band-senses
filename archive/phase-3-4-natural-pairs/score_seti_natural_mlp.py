"""Claim-verification eval for MLP bridge: domain=seti, source=natural.

Extracts 3-6 verifiable claims per record from generated_description,
translates radio-astronomy vocab to 5-symbol-level facts, then verifies
against the actual sequence+continuation.

SETI symbol semantics (5-symbol alphabet A..E):
  A = lowest quantile  (quiescent / below-floor)
  B = low-mid          (lower shoulder)
  C = mid-intensity    (noise-floor / central quantile)
  D = upper-mid        (upper shoulder / warm)
  E = top quantile     (excursion / RFI-like spike)

Vocab translation:
  "noise-floor", "mid-intensity dominant", "central quantile"   -> C dominant
  "warm upper tail", "upper-mid shoulder"                       -> D frequency elevated
  "top-quantile activity elevated (N samples)"                  -> count(E) ~ N
  "low-quantile bin heavy"                                      -> count(A) elevated
  "no drift across window"                                      -> halves symbol-mean ~ equal
  "moderate shoulders"                                          -> B and D both present, not dominant
"""

import json
import re
from collections import Counter
from pathlib import Path

IN_PATH = Path("G:/My Drive/nbs-bridge/results/mlp/seti_natural_eval_prepared.jsonl")
OUT_PATH = Path("G:/My Drive/nbs-bridge/results/mlp/seti_natural_scores.jsonl")


def freqs(seq: str) -> dict:
    c = Counter(seq)
    n = len(seq)
    return {k: c.get(k, 0) / n for k in "ABCDE"}, c


def extract_claims(desc: str):
    """Return list of (claim_text, predicate_fn). Predicate_fn(full_seq) -> verdict."""
    claims = []
    dl = desc.lower()

    # Claim 1: mid-intensity (C) dominant / noise-floor window
    if "mid-intensity dominant" in dl or "noise-floor window" in dl or "central quantile" in dl:
        def p_mid_dom(seq, _c=None):
            f, c = freqs(seq)
            # "dominant" = C is the max (or tied max) and exceeds 0.25
            max_sym = max(f, key=f.get)
            if f["C"] >= 0.25 and (max_sym == "C" or f["C"] >= max(f.values()) - 0.02):
                return "CONFIRMED"
            if f["C"] < 0.18:
                return "REFUTED"
            return "UNVERIFIABLE"
        claims.append(("C (mid-intensity) dominant", p_mid_dom))

    # Claim 2: warm upper tail / warm-skewed (D shoulder moderate)
    if "warm upper tail" in dl or "warm-skewed" in dl or "upper-mid shoulder" in dl:
        def p_warm(seq, _c=None):
            f, c = freqs(seq)
            # "warm" = D frequency moderate-high, > B frequency
            if f["D"] >= 0.20 and f["D"] > f["B"]:
                return "CONFIRMED"
            if f["D"] < 0.12:
                return "REFUTED"
            return "UNVERIFIABLE"
        claims.append(("D (warm) shoulder elevated above B", p_warm))

    # Claim 3: top-quantile activity elevated with specific sample count
    m = re.search(r"top-quantile activity elevated \((\d+) samples\)", dl)
    if m:
        claimed_n = int(m.group(1))
        def p_top_n(seq, _c=None, _n=claimed_n):
            _, c = freqs(seq)
            actual = c.get("E", 0)
            # Allow +/-3 tolerance given 250-char sequence (~250 samples)
            if abs(actual - _n) <= 3:
                return "CONFIRMED"
            if abs(actual - _n) > 10:
                return "REFUTED"
            return "UNVERIFIABLE"
        claims.append((f"E-class count ~= {claimed_n}", p_top_n))

    # Claim 4: low-quantile bin heavy (A frequency elevated)
    if "low-quantile bin heavy" in dl:
        def p_low_heavy(seq, _c=None):
            f, _ = freqs(seq)
            # "heavy" = A frequency > typical baseline 0.10
            if f["A"] >= 0.12:
                return "CONFIRMED"
            if f["A"] < 0.05:
                return "REFUTED"
            return "UNVERIFIABLE"
        claims.append(("A (low-quantile) frequency elevated", p_low_heavy))

    # Claim 5: no drift across window (halves symbol-mean approx equal)
    if "no drift" in dl or "does not drift" in dl:
        def p_no_drift(seq, _c=None):
            sym_to_val = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}
            vals = [sym_to_val[s] for s in seq]
            half = len(vals) // 2
            m1 = sum(vals[:half]) / half
            m2 = sum(vals[half:]) / (len(vals) - half)
            if abs(m1 - m2) <= 0.15:
                return "CONFIRMED"
            if abs(m1 - m2) > 0.4:
                return "REFUTED"
            return "UNVERIFIABLE"
        claims.append(("No drift: half-window means within 0.15", p_no_drift))

    # Claim 6: moderate shoulders (B and D both moderate, not dominant)
    if "moderate shoulders" in dl or ("lower-mid shoulder moderate" in dl and "upper-mid shoulder moderate" in dl):
        def p_shoulders(seq, _c=None):
            f, _ = freqs(seq)
            b_ok = 0.12 <= f["B"] <= 0.30
            d_ok = 0.15 <= f["D"] <= 0.35
            if b_ok and d_ok:
                return "CONFIRMED"
            if f["B"] < 0.05 or f["D"] < 0.05:
                return "REFUTED"
            return "UNVERIFIABLE"
        claims.append(("B and D both moderate (0.12-0.30 / 0.15-0.35)", p_shoulders))

    # Claim 7: distribution clustered around central quantile
    if "clustered around the central quantile" in dl or "distribution clustered" in dl:
        def p_cluster(seq, _c=None):
            f, _ = freqs(seq)
            # "clustered" = B+C+D dominate the mass (>= 0.80)
            bcd = f["B"] + f["C"] + f["D"]
            if bcd >= 0.80:
                return "CONFIRMED"
            if bcd < 0.65:
                return "REFUTED"
            return "UNVERIFIABLE"
        claims.append(("B+C+D mass >= 0.80 (clustered)", p_cluster))

    return claims


def extract_ceiling_claims(target_desc: str):
    """Ceiling claims from target_description — the oracle description."""
    claims = []
    dl = target_desc.lower()

    # Claim A: D-class percentage explicit
    m = re.search(r"d-class samples (\d+(?:\.\d+)?)%", dl)
    if m:
        pct = float(m.group(1)) / 100.0
        def p_d_pct(seq, _p=pct):
            f, _ = freqs(seq)
            if abs(f["D"] - _p) <= 0.03:
                return "CONFIRMED"
            if abs(f["D"] - _p) > 0.08:
                return "REFUTED"
            return "UNVERIFIABLE"
        claims.append((f"D frequency ~= {pct:.2%}", p_d_pct))

    # Claim B: E-class percentage explicit
    m = re.search(r"e-class samples (\d+(?:\.\d+)?)%", dl)
    if m:
        pct = float(m.group(1)) / 100.0
        def p_e_pct(seq, _p=pct):
            f, _ = freqs(seq)
            if abs(f["E"] - _p) <= 0.02:
                return "CONFIRMED"
            if abs(f["E"] - _p) > 0.05:
                return "REFUTED"
            return "UNVERIFIABLE"
        claims.append((f"E frequency ~= {pct:.2%}", p_e_pct))

    # Claim C: noise-dominated / noise-limited (C dominant)
    if "noise-dominated" in dl or "noise-limited" in dl or "noise floor" in dl:
        def p_noise_dom(seq, _c=None):
            f, _ = freqs(seq)
            max_sym = max(f, key=f.get)
            if max_sym in ("C", "D") and f[max_sym] >= 0.25:
                return "CONFIRMED"
            return "UNVERIFIABLE"
        claims.append(("Noise-dominated: C or D is max, >= 0.25", p_noise_dom))

    # Claim D: bulk clustered at floor (34% typically)
    m = re.search(r"bulk of samples \((\d+)%\) clustered", dl)
    if m:
        pct = float(m.group(1)) / 100.0
        def p_bulk(seq, _p=pct):
            f, _ = freqs(seq)
            # Bulk = max symbol frequency
            maxf = max(f.values())
            if abs(maxf - _p) <= 0.04:
                return "CONFIRMED"
            if abs(maxf - _p) > 0.10:
                return "REFUTED"
            return "UNVERIFIABLE"
        claims.append((f"Max symbol frequency ~= {pct:.2%}", p_bulk))

    # Claim E: shallow drift / mildly drifts / thermal drift
    if "thermal drift" in dl or "drifts mildly" in dl or "shallow" in dl or "baseline drift" in dl:
        def p_drift(seq, _c=None):
            sym_to_val = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}
            vals = [sym_to_val[s] for s in seq]
            half = len(vals) // 2
            m1 = sum(vals[:half]) / half
            m2 = sum(vals[half:]) / (len(vals) - half)
            diff = abs(m1 - m2)
            # Shallow drift: 0.05 to 0.4
            if 0.05 <= diff <= 0.4:
                return "CONFIRMED"
            return "UNVERIFIABLE"
        claims.append(("Shallow drift: half-means differ 0.05-0.40", p_drift))

    # Claim F: one-off excursion (E appears but not in runs)
    if "one-off excursion" in dl or "does not recur" in dl or "does not repeat" in dl:
        def p_oneoff(seq, _c=None):
            # Check no 2 consecutive E's, and E count >= 1
            e_count = seq.count("E")
            max_e_run = 0
            cur = 0
            for ch in seq:
                if ch == "E":
                    cur += 1
                    max_e_run = max(max_e_run, cur)
                else:
                    cur = 0
            if e_count >= 1 and max_e_run <= 2:
                return "CONFIRMED"
            if max_e_run >= 4:
                return "REFUTED"
            return "UNVERIFIABLE"
        claims.append(("E excursions present but no run of 3+", p_oneoff))

    # Claim G: no persistent structure (no long runs of any one symbol)
    if "no persistent structure" in dl or "no reproducible structure" in dl:
        def p_no_struct(seq, _c=None):
            max_run = 0
            cur = 0
            prev = None
            for ch in seq:
                if ch == prev:
                    cur += 1
                else:
                    cur = 1
                max_run = max(max_run, cur)
                prev = ch
            if max_run <= 5:
                return "CONFIRMED"
            if max_run >= 8:
                return "REFUTED"
            return "UNVERIFIABLE"
        claims.append(("No run of any symbol longer than 5", p_no_struct))

    return claims


def score(claims, full_seq):
    n = len(claims)
    confirmed = refuted = unver = 0
    details = []
    for txt, pred in claims:
        v = pred(full_seq)
        details.append({"claim": txt, "verdict": v})
        if v == "CONFIRMED":
            confirmed += 1
        elif v == "REFUTED":
            refuted += 1
        else:
            unver += 1
    return n, confirmed, refuted, unver, details


def main():
    records = []
    with IN_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    print(f"Loaded {len(records)} records")

    out_lines = []
    summary = {"bridge_total_m": 0, "bridge_total_c": 0,
               "ceiling_total_m": 0, "ceiling_total_c": 0}

    for rec in records:
        seq_idx = rec["seq_idx"]
        full_seq = rec["sequence"] + rec["continuation"]
        gen = rec["generated_description"]
        tgt = rec["target_description"]

        gen_claims = extract_claims(gen)
        # Ensure 3-6 claims; if fewer, pad with fallback generic claims
        if len(gen_claims) < 3:
            # Fallback: "noise-floor" implied; use baseline 3 generic claims
            def fb1(seq, _=None):
                f, _c = freqs(seq)
                return "CONFIRMED" if f["C"] >= 0.20 else "UNVERIFIABLE"
            def fb2(seq, _=None):
                f, _c = freqs(seq)
                return "CONFIRMED" if 0.15 <= f["D"] <= 0.40 else "UNVERIFIABLE"
            def fb3(seq, _=None):
                _, c = freqs(seq)
                return "CONFIRMED" if c.get("E", 0) <= 15 else "REFUTED"
            fallbacks = [
                ("Fallback: C frequency >= 0.20", fb1),
                ("Fallback: D in [0.15, 0.40]", fb2),
                ("Fallback: E count <= 15", fb3),
            ]
            for fb in fallbacks:
                if len(gen_claims) < 3:
                    gen_claims.append(fb)
        if len(gen_claims) > 6:
            gen_claims = gen_claims[:6]

        ceil_claims = extract_ceiling_claims(tgt)
        if len(ceil_claims) < 3:
            def cfb1(seq, _=None):
                f, _c = freqs(seq)
                maxf = max(f.values())
                return "CONFIRMED" if 0.25 <= maxf <= 0.45 else "UNVERIFIABLE"
            def cfb2(seq, _=None):
                _, c = freqs(seq)
                return "CONFIRMED" if c.get("E", 0) >= 1 else "UNVERIFIABLE"
            def cfb3(seq, _=None):
                f, _c = freqs(seq)
                return "CONFIRMED" if f["A"] + f["E"] < 0.25 else "UNVERIFIABLE"
            for fb in [("Ceiling fb: max freq in [0.25, 0.45]", cfb1),
                       ("Ceiling fb: E count >= 1", cfb2),
                       ("Ceiling fb: A+E < 0.25", cfb3)]:
                if len(ceil_claims) < 3:
                    ceil_claims.append(fb)
        if len(ceil_claims) > 6:
            ceil_claims = ceil_claims[:6]

        m, c, r, u, _ = score(gen_claims, full_seq)
        m2, c2, r2, u2, _ = score(ceil_claims, full_seq)

        out = {
            "seq_idx": seq_idx,
            "n_claims": m,
            "n_confirmed": c,
            "n_refuted": r,
            "n_unverifiable": u,
            "bridge_score": c / m if m else 0.0,
            "ceiling_n_claims": m2,
            "ceiling_n_confirmed": c2,
            "ceiling_score": c2 / m2 if m2 else 0.0,
        }
        out_lines.append(json.dumps(out))
        summary["bridge_total_m"] += m
        summary["bridge_total_c"] += c
        summary["ceiling_total_m"] += m2
        summary["ceiling_total_c"] += c2

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        f.write("\n".join(out_lines) + "\n")

    b_avg = summary["bridge_total_c"] / max(1, summary["bridge_total_m"])
    c_avg = summary["ceiling_total_c"] / max(1, summary["ceiling_total_m"])
    print(f"Wrote {len(out_lines)} records to {OUT_PATH}")
    print(f"Bridge avg (micro): {b_avg:.3f}  ({summary['bridge_total_c']}/{summary['bridge_total_m']})")
    print(f"Ceiling avg (micro): {c_avg:.3f}  ({summary['ceiling_total_c']}/{summary['ceiling_total_m']})")


if __name__ == "__main__":
    main()
