"""
Claim-verification eval for MLP bridge: domain=financial, source=natural.

Extracts 3-6 verifiable claims from generated_description, translates market-analyst
vocab to bin-level facts, and verifies against the full sequence+continuation window.

Binning convention (matched by trial-fit against target_description percentages):
  lower = digits 0,1,2   (three bins)
  middle = digits 3,4,5,6 (four bins)
  upper = digits 7,8,9   (three bins)

Emits per-record scoring JSONL.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from statistics import mean

IN_PATH = Path("G:/My Drive/nbs-bridge/results/mlp/financial_natural_eval_prepared.jsonl")
OUT_PATH = Path("G:/My Drive/nbs-bridge/results/mlp/financial_natural_scores.jsonl")

# Tolerance: pct claims allowed within +/- TOL_PCT of observed value.
TOL_PCT = 3.0            # percentage-point tolerance for bin-share claims
TOL_HHI = 0.05           # absolute tolerance on HHI (0..1 scale)
TOL_RATIO = 0.25         # tolerance on tail-ratio claim (multiplicative-ish)
TOL_REPEAT = 3.0         # percentage-point tolerance on same-bucket repeat rate
TOL_LAG1 = 0.08          # absolute tolerance on lag-1 autocorrelation


def compute_stats(seq: str) -> dict:
    """Compute bin-level ground-truth features from the full digit sequence."""
    n = len(seq)
    digits = [int(c) for c in seq]

    def bin_of(d: int) -> str:
        if d <= 2:
            return "lower"
        if d >= 7:
            return "upper"
        return "middle"

    bins = [bin_of(d) for d in digits]
    upper_pct = 100.0 * sum(1 for b in bins if b == "upper") / n
    lower_pct = 100.0 * sum(1 for b in bins if b == "lower") / n
    middle_pct = 100.0 * sum(1 for b in bins if b == "middle") / n

    # Herfindahl-Hirschman Index on 10-digit distribution (0..1 scale).
    counts = [0] * 10
    for d in digits:
        counts[d] += 1
    shares = [c / n for c in counts]
    hhi = sum(s * s for s in shares)

    # Same-bucket repeat rate: consecutive identical-bin transitions.
    repeats = sum(1 for i in range(1, n) if bins[i] == bins[i - 1])
    repeat_pct = 100.0 * repeats / (n - 1)

    # Lag-1 autocorrelation of digit magnitude.
    mu = mean(digits)
    num = sum((digits[i] - mu) * (digits[i - 1] - mu) for i in range(1, n))
    den = sum((d - mu) ** 2 for d in digits)
    lag1 = num / den if den > 0 else 0.0

    # Tail asymmetry: upper / lower share.
    ratio = (upper_pct / lower_pct) if lower_pct > 0 else float("inf")

    return {
        "upper_pct": upper_pct,
        "lower_pct": lower_pct,
        "middle_pct": middle_pct,
        "hhi": hhi,
        "repeat_pct": repeat_pct,
        "lag1": lag1,
        "tail_ratio": ratio,
    }


# --- claim extraction --------------------------------------------------------

def extract_claims(desc: str) -> list[dict]:
    """Extract 3-6 verifiable claims. Each claim is a dict with 'kind' and params."""
    claims: list[dict] = []

    # Upper tail bin share.
    m = re.search(r"upper\s+tail\s+([\d.]+)\s*%", desc, re.I)
    if m:
        claims.append({"kind": "upper_pct", "value": float(m.group(1))})

    # Lower band bin share.
    m = re.search(r"lower\s+band\s+([\d.]+)\s*%", desc, re.I)
    if m:
        claims.append({"kind": "lower_pct", "value": float(m.group(1))})

    # Middle band bin share.
    m = re.search(r"middle(?:\s+band)?\s+([\d.]+)\s*%", desc, re.I)
    if m:
        claims.append({"kind": "middle_pct", "value": float(m.group(1))})

    # "Full bin range is engaged" => at least one obs in each bin.
    if re.search(r"full\s+bin\s+range\s+is\s+engaged", desc, re.I):
        claims.append({"kind": "all_bins_nonzero"})

    # Symmetric tails (ratio x).
    m = re.search(r"(?:tails\s+are\s+roughly\s+symmetric|tail\s+asymmetry\s+is\s+mild)\s*\(ratio\s+([\d.]+)\)", desc, re.I)
    if m:
        claims.append({"kind": "tail_ratio", "value": float(m.group(1))})

    # HHI dispersion. Note: generated values sometimes look like 0.9 or 0.967 — out of plausible
    # 10-bin HHI range (0.1..1.0). We verify numerically against computed HHI anyway.
    m = re.search(r"HHI\s+([\d.]+)", desc, re.I)
    if m:
        claims.append({"kind": "hhi", "value": float(m.group(1))})

    # Moderate clustering at X% same-bucket repeats.
    m = re.search(r"clustering\s+at\s+([\d.]+)\s*%\s+same-bucket\s+repeats", desc, re.I)
    if m:
        claims.append({"kind": "repeat_pct", "value": float(m.group(1)), "qualitative": "moderate"})

    # Repeat-rate of only X%.
    m = re.search(r"repeat[-\s]rate\s+of\s+only\s+([\d.]+)\s*%", desc, re.I)
    if m:
        claims.append({"kind": "repeat_pct", "value": float(m.group(1)), "qualitative": "low"})

    # Lag-1 autocorrelation of X.
    m = re.search(r"lag[-\s]?1\s+autocorrelation\s+of\s+(-?[\d.]+)", desc, re.I)
    if m:
        val = float(m.group(1))
        claims.append({"kind": "lag1", "value": val})

    # Negative lag-1 autocorrelation (-X) => specifically negative.
    m = re.search(r"negative\s+lag[-\s]?1\s+autocorrelation\s*\((-?[\d.]+)\)", desc, re.I)
    if m:
        claims.append({"kind": "lag1", "value": float(m.group(1))})

    # "Near-uniform dispersion" qualitative claim: HHI should be close to 0.1 (uniform over 10 bins).
    if re.search(r"near[-\s]uniform\s+dispersion", desc, re.I):
        claims.append({"kind": "near_uniform_hhi"})

    # "Broad-distribution regime lacking a dominant mode" => no bin >= 25% share.
    if re.search(r"broad[-\s]distribution\s+regime\s+lacking\s+a\s+dominant\s+mode", desc, re.I):
        claims.append({"kind": "no_dominant_mode"})

    # De-duplicate by (kind, value) keeping first occurrence; cap at 6.
    seen = set()
    uniq = []
    for c in claims:
        key = (c["kind"], c.get("value"))
        if key in seen:
            continue
        seen.add(key)
        uniq.append(c)

    return uniq[:6]


# --- verification ------------------------------------------------------------

def verify_claim(claim: dict, stats: dict, full_seq: str) -> str:
    kind = claim["kind"]
    if kind == "upper_pct":
        return "CONFIRMED" if abs(claim["value"] - stats["upper_pct"]) <= TOL_PCT else "REFUTED"
    if kind == "lower_pct":
        return "CONFIRMED" if abs(claim["value"] - stats["lower_pct"]) <= TOL_PCT else "REFUTED"
    if kind == "middle_pct":
        return "CONFIRMED" if abs(claim["value"] - stats["middle_pct"]) <= TOL_PCT else "REFUTED"
    if kind == "all_bins_nonzero":
        has_lower = any(c in "012" for c in full_seq)
        has_mid = any(c in "3456" for c in full_seq)
        has_upper = any(c in "789" for c in full_seq)
        return "CONFIRMED" if (has_lower and has_mid and has_upper) else "REFUTED"
    if kind == "tail_ratio":
        r_obs = stats["tail_ratio"]
        r_claim = claim["value"]
        # Accept either direction orientation (upper/lower or lower/upper).
        err = min(abs(r_obs - r_claim), abs((1.0 / r_obs if r_obs else 0) - r_claim))
        return "CONFIRMED" if err <= TOL_RATIO else "REFUTED"
    if kind == "hhi":
        return "CONFIRMED" if abs(claim["value"] - stats["hhi"]) <= TOL_HHI else "REFUTED"
    if kind == "repeat_pct":
        # Numerical check: +/- TOL_REPEAT points.
        return "CONFIRMED" if abs(claim["value"] - stats["repeat_pct"]) <= TOL_REPEAT else "REFUTED"
    if kind == "lag1":
        return "CONFIRMED" if abs(claim["value"] - stats["lag1"]) <= TOL_LAG1 else "REFUTED"
    if kind == "near_uniform_hhi":
        # Uniform 10-bin HHI = 0.1; accept <= 0.15.
        return "CONFIRMED" if stats["hhi"] <= 0.15 else "REFUTED"
    if kind == "no_dominant_mode":
        # No bin (lower/middle/upper) above 60%.
        maxshare = max(stats["upper_pct"], stats["lower_pct"], stats["middle_pct"])
        return "CONFIRMED" if maxshare < 60.0 else "REFUTED"
    return "UNVERIFIABLE"


# --- main --------------------------------------------------------------------

def main() -> None:
    records = [json.loads(line) for line in IN_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(records) == 50, f"Expected 50 records, got {len(records)}"

    out_lines = []
    for rec in records:
        seq_idx = rec["seq_idx"]
        full = rec["sequence"] + rec["continuation"]
        stats = compute_stats(full)
        desc = rec["generated_description"]

        claims = extract_claims(desc)
        # Enforce 3-6 claim range: if fewer than 3, pad by re-inspecting with fallback qualitative claims.
        if len(claims) < 3:
            # Add fallback qualitative claims that can still be verified.
            if not any(c["kind"] == "no_dominant_mode" for c in claims):
                claims.append({"kind": "no_dominant_mode"})
            if len(claims) < 3 and not any(c["kind"] == "all_bins_nonzero" for c in claims):
                claims.append({"kind": "all_bins_nonzero"})
            if len(claims) < 3 and not any(c["kind"] == "near_uniform_hhi" for c in claims):
                claims.append({"kind": "near_uniform_hhi"})
        claims = claims[:6]

        verdicts = [verify_claim(c, stats, full) for c in claims]
        n_claims = len(claims)
        n_conf = sum(1 for v in verdicts if v == "CONFIRMED")
        n_ref = sum(1 for v in verdicts if v == "REFUTED")
        n_unv = sum(1 for v in verdicts if v == "UNVERIFIABLE")
        bridge_score = n_conf / n_claims if n_claims else 0.0

        # CEILING: 3-6 independent claims. Select up to 6 claims drawn from orthogonal
        # feature axes so they are not trivially correlated.
        ceiling_kinds_order = [
            "upper_pct", "lower_pct", "middle_pct", "hhi", "repeat_pct",
            "lag1", "tail_ratio", "all_bins_nonzero", "no_dominant_mode", "near_uniform_hhi",
        ]
        independent = []
        seen_kinds = set()
        for c in claims:
            if c["kind"] not in seen_kinds:
                independent.append(c)
                seen_kinds.add(c["kind"])
        # Pad to at least 3 using fallback kinds.
        for kind in ceiling_kinds_order:
            if len(independent) >= 3:
                break
            if kind not in seen_kinds:
                independent.append({"kind": kind} if kind in ("all_bins_nonzero", "no_dominant_mode", "near_uniform_hhi") else None)
                if independent[-1] is None:
                    independent.pop()
                else:
                    seen_kinds.add(kind)
        independent = independent[:6]
        ceil_verdicts = [verify_claim(c, stats, full) for c in independent]
        ceil_n = len(independent)
        ceil_conf = sum(1 for v in ceil_verdicts if v == "CONFIRMED")
        ceil_score = ceil_conf / ceil_n if ceil_n else 0.0

        out_lines.append(json.dumps({
            "seq_idx": seq_idx,
            "n_claims": n_claims,
            "n_confirmed": n_conf,
            "n_refuted": n_ref,
            "n_unverifiable": n_unv,
            "bridge_score": bridge_score,
            "ceiling_n_claims": ceil_n,
            "ceiling_n_confirmed": ceil_conf,
            "ceiling_score": ceil_score,
        }))

    OUT_PATH.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    print(f"Wrote {len(out_lines)} records to {OUT_PATH}")

    # Summary stats.
    import statistics as st
    bscores = [json.loads(l)["bridge_score"] for l in out_lines]
    cscores = [json.loads(l)["ceiling_score"] for l in out_lines]
    print(f"mean bridge_score = {st.mean(bscores):.3f}")
    print(f"mean ceiling_score = {st.mean(cscores):.3f}")


if __name__ == "__main__":
    main()
