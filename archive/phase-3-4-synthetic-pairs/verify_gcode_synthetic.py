"""
Claim verification for MLP bridge, domain=gcode, source=synthetic.

For each record we:
  1. Parse `generated_description` for SPECIFIC verifiable claims about
     character-level statistics of the `sequence`:
       - total length (e.g. "200-character stream")
       - decimal-point count ("24 decimal points", "24 dots")
       - digit percentage ("about 72% of positions are digits")
       - capital-letter counts ("G appears 8 times")
       - capital-letter absence ("Z, F, T, S are absent")
       - n-gram frequency claims ("'G1X' repeats 8 times", "'X10' appears N times")
       - whitespace / newline absence
  2. Verify each claim against the actual `sequence` string
     (CONFIRMED / REFUTED / UNVERIFIABLE).
  3. Compute a CEILING score by verifying the SAME claims against
     sequence + continuation (which is what the model was effectively
     predicting).  This gives an upper bound on how many claims *could*
     have been right if the model had perfectly described the full window.

Schema of each output record:
  {
    "seq_idx": N,
    "n_claims": M, "n_confirmed": C, "n_refuted": R,
    "n_unverifiable": U, "bridge_score": C/M,
    "ceiling_n_claims": M2, "ceiling_n_confirmed": C2,
    "ceiling_score": C2/M2
  }
"""

import json
import re
from collections import Counter
from pathlib import Path

IN_PATH = Path("G:/My Drive/nbs-bridge/results/mlp/gcode_synthetic_eval_prepared.jsonl")
OUT_PATH = Path("G:/My Drive/nbs-bridge/results/mlp/gcode_synthetic_scores.jsonl")

# tolerances
PCT_TOL = 3.0          # percentage-point tolerance for digit/decimal percentages
LEN_TOL = 5            # char tolerance for total length
COUNT_TOL = 2          # absolute tolerance for letter counts / n-gram counts
DOT_TOL = 3            # tolerance for decimal-point count


# ---------- helpers ----------

def digit_pct(s):
    return 100.0 * sum(c.isdigit() for c in s) / len(s) if s else 0.0


def dot_count(s):
    return s.count(".")


def letter_count(s, letter):
    return s.count(letter)


def ngram_count(s, ng):
    # overlapping count
    n = len(ng)
    if n == 0:
        return 0
    return sum(1 for i in range(len(s) - n + 1) if s[i:i + n] == ng)


def has_any_whitespace(s):
    return any(c.isspace() for c in s)


# ---------- claim extraction ----------

def extract_claims(text):
    """
    Return a list of dicts describing verifiable claims.
    Each claim dict has:
      type:  "length" | "dot_count" | "digit_pct" | "letter_count"
             | "letter_absent" | "ngram_count" | "no_whitespace"
      value: the asserted value
      raw:   the source phrase (for debugging)
    """
    claims = []
    t = text

    # --- length ---
    m = re.search(r"(\d{2,4})\s*-?\s*character\s+stream", t, re.I)
    if m:
        claims.append({"type": "length", "value": int(m.group(1)), "raw": m.group(0)})

    # --- "no newlines, no spaces" ---
    if re.search(r"no\s+newlines?", t, re.I) and re.search(r"no\s+spaces?", t, re.I):
        claims.append({"type": "no_whitespace", "value": True,
                       "raw": "no newlines, no spaces"})

    # --- decimal points / dots count: "24 decimal points", "24 dots total" ---
    m = re.search(r"(\d+)\s+decimal\s+points?", t, re.I)
    if m:
        claims.append({"type": "dot_count", "value": int(m.group(1)),
                       "raw": m.group(0)})
    else:
        m = re.search(r"(\d+)\s+dots?\s+total", t, re.I)
        if m:
            claims.append({"type": "dot_count", "value": int(m.group(1)),
                           "raw": m.group(0)})

    # --- digit percentage: "about 72% of positions are digits",
    #     "roughly 74% of the content", "72.5% of positions" ---
    m = re.search(
        r"(?:about|roughly|approximately|around)?\s*(\d{1,3}(?:\.\d+)?)\s*%\s*"
        r"(?:of\s+)?(?:positions?|characters?|content)?[^.]*?\bdigits?\b",
        t, re.I,
    )
    if m:
        claims.append({"type": "digit_pct", "value": float(m.group(1)),
                       "raw": m.group(0)})
    else:
        m = re.search(r"\bdigits?\b[^.]{0,40}?(\d{1,3}(?:\.\d+)?)\s*%", t, re.I)
        if m:
            claims.append({"type": "digit_pct", "value": float(m.group(1)),
                           "raw": m.group(0)})

    # --- decimal-point percentage: "decimal point making up about 12.00% of..." ---
    m = re.search(
        r"decimal\s+points?\b[^.]{0,60}?(\d{1,3}(?:\.\d+)?)\s*%",
        t, re.I,
    )
    if m:
        claims.append({"type": "dot_pct", "value": float(m.group(1)),
                       "raw": m.group(0)})

    # --- letter counts: "G appears 8 times", "X occurs 8 times", "Y 8 times" ---
    for m in re.finditer(
        r"\b([A-Z])\b\s*(?:appears|occurs|shows up|is)?\s*(\d+)\s*times?",
        t,
    ):
        letter = m.group(1)
        count = int(m.group(2))
        # guard against matching things like "G7 with 4 characters" — require
        # either the verb or the pattern "LETTER N times"
        if re.search(rf"\b{letter}\b\s*(?:appears|occurs|shows up|is)\s*{count}\s*times?", t) \
                or re.search(rf"\b{letter}\s+{count}\s*times?", t):
            claims.append({"type": "letter_count", "letter": letter,
                           "value": count, "raw": m.group(0)})

    # also catch "X is 8 times" style in lists: "X 8 times, Y 8 times, E 8 times"
    for m in re.finditer(r"\b([A-Z])\s+(\d+)\b(?!\s*[%.])", t):
        letter = m.group(1)
        count = int(m.group(2))
        # only accept if "times" follows within a few chars
        tail = t[m.end():m.end() + 10]
        if re.match(r"\s*times?\b", tail, re.I):
            claims.append({"type": "letter_count", "letter": letter,
                           "value": count, "raw": m.group(0)})

    # de-dup letter_count claims
    seen = set()
    dedup = []
    for c in claims:
        if c["type"] == "letter_count":
            key = (c["type"], c["letter"], c["value"])
            if key in seen:
                continue
            seen.add(key)
        dedup.append(c)
    claims = dedup

    # --- letters absent: "Z, F, T, S and M are absent" ---
    m = re.search(
        r"([A-Z](?:\s*,\s*[A-Z])*(?:\s*,?\s*and\s*[A-Z])?)\s+are\s+absent",
        t,
    )
    if m:
        group = m.group(1)
        letters = re.findall(r"\b([A-Z])\b", group)
        for L in letters:
            claims.append({"type": "letter_absent", "letter": L,
                           "raw": f"{L} absent"})

    # --- n-gram counts: "'G1X' (8)", "'G1X1' appears 8 times",
    #     "each repeating 8" ---
    for m in re.finditer(r"'([^']{2,6})'\s*\(\s*(\d+)\s*\)", t):
        claims.append({"type": "ngram_count", "ngram": m.group(1),
                       "value": int(m.group(2)), "raw": m.group(0)})
    for m in re.finditer(r"'([^']{2,6})'\s*(?:appears|occurs|repeats?|repeating)\s*(\d+)\s*times?", t):
        claims.append({"type": "ngram_count", "ngram": m.group(1),
                       "value": int(m.group(2)), "raw": m.group(0)})

    # cap at 6 claims (prefer first occurrences which are usually well-formed)
    return claims[:6]


# ---------- verification ----------

def verify_claim(claim, seq):
    """Return 'CONFIRMED' | 'REFUTED' | 'UNVERIFIABLE'."""
    t = claim["type"]
    if t == "length":
        return "CONFIRMED" if abs(len(seq) - claim["value"]) <= LEN_TOL else "REFUTED"
    if t == "no_whitespace":
        return "CONFIRMED" if not has_any_whitespace(seq) else "REFUTED"
    if t == "dot_count":
        return "CONFIRMED" if abs(dot_count(seq) - claim["value"]) <= DOT_TOL else "REFUTED"
    if t == "digit_pct":
        return "CONFIRMED" if abs(digit_pct(seq) - claim["value"]) <= PCT_TOL else "REFUTED"
    if t == "dot_pct":
        actual = 100.0 * dot_count(seq) / len(seq) if seq else 0.0
        return "CONFIRMED" if abs(actual - claim["value"]) <= PCT_TOL else "REFUTED"
    if t == "letter_count":
        actual = letter_count(seq, claim["letter"])
        return "CONFIRMED" if abs(actual - claim["value"]) <= COUNT_TOL else "REFUTED"
    if t == "letter_absent":
        return "CONFIRMED" if letter_count(seq, claim["letter"]) == 0 else "REFUTED"
    if t == "ngram_count":
        actual = ngram_count(seq, claim["ngram"])
        # for n-grams, allow a generous tolerance, but require non-zero
        if claim["value"] == 0:
            return "CONFIRMED" if actual == 0 else "REFUTED"
        # otherwise COUNT_TOL absolute, and also accept if the n-gram is
        # "reasonably frequent" (>=1 occurrence and within a factor of 2)
        if abs(actual - claim["value"]) <= COUNT_TOL:
            return "CONFIRMED"
        if actual > 0 and max(actual, claim["value"]) / max(1, min(actual, claim["value"])) <= 2.0:
            return "CONFIRMED"
        return "REFUTED"
    return "UNVERIFIABLE"


# ---------- main ----------

def main():
    rows = []
    with IN_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    out_lines = []
    for r in rows:
        seq = r["sequence"]
        full = seq + r.get("continuation", "")
        desc = r.get("generated_description", "")

        claims = extract_claims(desc)
        n = len(claims)

        # bridge_score: verify against seq
        c_c = sum(1 for cl in claims if verify_claim(cl, seq) == "CONFIRMED")
        c_r = sum(1 for cl in claims if verify_claim(cl, seq) == "REFUTED")
        c_u = n - c_c - c_r

        # ceiling: verify against seq + continuation, optionally re-extracting
        # (we keep the same claims — they describe what the generator *said*,
        # and we ask whether they're true of the fuller window).
        ceiling_claims = claims
        m2 = len(ceiling_claims)
        c2_c = sum(1 for cl in ceiling_claims if verify_claim(cl, full) == "CONFIRMED")

        bridge_score = c_c / n if n else 0.0
        ceiling_score = c2_c / m2 if m2 else 0.0

        out_lines.append({
            "seq_idx": r["seq_idx"],
            "n_claims": n,
            "n_confirmed": c_c,
            "n_refuted": c_r,
            "n_unverifiable": c_u,
            "bridge_score": bridge_score,
            "ceiling_n_claims": m2,
            "ceiling_n_confirmed": c2_c,
            "ceiling_score": ceiling_score,
        })

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for row in out_lines:
            f.write(json.dumps(row) + "\n")

    # console summary
    n_recs = len(out_lines)
    tot_claims = sum(r["n_claims"] for r in out_lines)
    tot_conf = sum(r["n_confirmed"] for r in out_lines)
    tot_ref = sum(r["n_refuted"] for r in out_lines)
    tot_unv = sum(r["n_unverifiable"] for r in out_lines)
    avg_bridge = sum(r["bridge_score"] for r in out_lines) / n_recs if n_recs else 0.0
    avg_ceil = sum(r["ceiling_score"] for r in out_lines) / n_recs if n_recs else 0.0
    print(f"records: {n_recs}")
    print(f"claims total: {tot_claims} (avg {tot_claims / n_recs:.2f}/rec)")
    print(f"  confirmed:    {tot_conf}")
    print(f"  refuted:      {tot_ref}")
    print(f"  unverifiable: {tot_unv}")
    print(f"avg bridge_score:  {avg_bridge:.3f}")
    print(f"avg ceiling_score: {avg_ceil:.3f}")
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
