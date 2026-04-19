"""
Claim-verification eval for MLP bridge on domain=english, source=synthetic.

Extracts specific verifiable character-level claims from generated_description,
verifies each against the sequence+continuation text, and writes scores.

Claims considered (with tolerances noted):
  - token count (exact)
  - mean token length (rounded to 1 decimal, tolerance 0.05)
  - min/max token length (exact)
  - vowel share / vowel count / letter count
  - top letter frequencies (letter + count, exact)
  - dominant bigrams (bigram + count, exact)

CONFIRMED if found in text statistics and matches within tolerance.
REFUTED if claim is specific and numeric but mismatches text.
UNVERIFIABLE if the claim is too vague or we cannot parse it.

Ceiling: up to 6 independent claim slots per record, drawn from the same
text statistics computed directly from sequence+continuation. The ceiling
represents an oracle that reports stats correctly.
"""

import json
import re
from collections import Counter
from pathlib import Path

IN_PATH = Path("G:/My Drive/nbs-bridge/results/mlp/english_synthetic_eval_prepared.jsonl")
OUT_PATH = Path("G:/My Drive/nbs-bridge/results/mlp/english_synthetic_scores.jsonl")

VOWELS = set("aeiou")


def compute_stats(text: str) -> dict:
    """Compute character-level statistics from a text string."""
    tokens = text.split()
    n_tokens = len(tokens)
    token_lens = [len(t) for t in tokens] if tokens else [0]
    mean_len = sum(token_lens) / n_tokens if n_tokens else 0.0
    min_len = min(token_lens) if tokens else 0
    max_len = max(token_lens) if tokens else 0

    letters = [c for c in text if c.isalpha()]
    n_letters = len(letters)
    n_vowels = sum(1 for c in letters if c in VOWELS)
    vowel_share = n_vowels / n_letters if n_letters else 0.0

    letter_counts = Counter(letters)
    # bigrams: consecutive letter pairs within the whole text including spaces? Target uses character bigrams of letters.
    # Looking at targets, bigrams are within-token character bigrams ('he' across "her"/"the") and appear letter-only.
    # We compute bigrams across adjacent characters where both are letters.
    bigrams = Counter()
    for i in range(len(text) - 1):
        a, b = text[i], text[i + 1]
        if a.isalpha() and b.isalpha():
            bigrams[a + b] += 1

    return {
        "n_tokens": n_tokens,
        "mean_len": mean_len,
        "min_len": min_len,
        "max_len": max_len,
        "n_letters": n_letters,
        "n_vowels": n_vowels,
        "vowel_share": vowel_share,
        "letter_counts": letter_counts,
        "bigrams": bigrams,
    }


# ----- Claim extraction from generated_description -----

def extract_claims(desc: str) -> list[dict]:
    """Extract up to 6 specific verifiable claims from the generated description.

    Each claim is a dict with a 'type' and parameters needed to verify.
    """
    claims = []
    d = desc.lower()

    # 1. Token count: "parses into N ... tokens" or "Counted N tokens" or "Across N tokens"
    m = re.search(r"(?:parses into|counted|across)\s+(\d+)\s+(?:whitespace[- ]delimited\s+)?tokens?", d)
    if m:
        claims.append({"type": "n_tokens", "value": int(m.group(1)), "raw": m.group(0)})

    # 2. Mean token length: "mean token length of X characters" or "mean X.XX chars"
    m = re.search(r"mean(?:\s+token\s+length\s+of|\s+)(\d+\.?\d*)\s*(?:chars?|characters?)", d)
    if m:
        claims.append({"type": "mean_len", "value": float(m.group(1)), "raw": m.group(0)})

    # 3. Range: "ranging from A to B" (min/max token length)
    m = re.search(r"ranging from\s+(\d+)\s+to\s+(\d+)", d)
    if m:
        claims.append({"type": "token_range", "min": int(m.group(1)), "max": int(m.group(2)), "raw": m.group(0)})

    # 4. Vowel share: "vowel share is X.XXX (N/M)" or "X% of M alphabetic"
    m = re.search(r"vowel share is\s+(\d+\.?\d*)\s*\((\d+)/(\d+)\)", d)
    if m:
        claims.append({
            "type": "vowel_share",
            "share": float(m.group(1)),
            "n_vowels": int(m.group(2)),
            "n_letters": int(m.group(3)),
            "raw": m.group(0),
        })
    else:
        m = re.search(r"vowels.*?account for\s+(\d+\.?\d*)%\s+of\s+(\d+)", d)
        if m:
            claims.append({
                "type": "vowel_share_pct",
                "share_pct": float(m.group(1)),
                "n_letters": int(m.group(2)),
                "raw": m.group(0),
            })

    # 5. Top letter frequencies: find all "'x' (N)" patterns after "letter" or "frequen" context
    # Look in the segment after "frequencies:" or "frequent letters are"
    letter_section = None
    m = re.search(r"(?:top letter frequencies|most (?:common|frequent) letters(?:\s+are)?)\s*:?\s*([^.]+)", d)
    if m:
        letter_section = m.group(1)
        letter_pairs = re.findall(r"'([a-z\-])'\s*\((\d+)\)", letter_section)
        # Only take concrete letters, skip placeholder '-'
        concrete = [(L, int(n)) for L, n in letter_pairs if L != "-"]
        if concrete:
            # Take up to 3 letter-frequency claims as a single aggregate claim per letter
            # But we want independent claims; take top few letters as separate claims only
            # to avoid blowing past 6 total. We take the first 2 here.
            for L, n in concrete[:2]:
                claims.append({"type": "letter_freq", "letter": L, "count": n, "raw": f"'{L}' ({n})"})

    # 6. Dominant bigrams: "'xy' xN"
    bg_section = None
    m = re.search(r"(?:dominant bigrams|leading character bigrams|bigrams)\s*(?:include)?\s*:?\s*([^.]+)", d)
    if m:
        bg_section = m.group(1)
        bg_pairs = re.findall(r"'([a-z]{2})'\s*x(\d+)", bg_section)
        if bg_pairs:
            for bg, n in bg_pairs[:2]:
                claims.append({"type": "bigram", "bigram": bg, "count": int(n), "raw": f"'{bg}' x{n}"})

    # Deduplicate (keep first occurrence of each (type, key))
    seen = set()
    unique = []
    for c in claims:
        key = (c["type"], c.get("letter"), c.get("bigram"))
        if key in seen:
            continue
        seen.add(key)
        unique.append(c)

    # Cap at 6
    return unique[:6]


# ----- Verification -----

def verify_claim(claim: dict, stats: dict) -> str:
    """Return CONFIRMED / REFUTED / UNVERIFIABLE."""
    t = claim["type"]

    if t == "n_tokens":
        return "CONFIRMED" if claim["value"] == stats["n_tokens"] else "REFUTED"

    if t == "mean_len":
        # tolerance 0.05 to handle rounding
        return "CONFIRMED" if abs(claim["value"] - stats["mean_len"]) <= 0.05 else "REFUTED"

    if t == "token_range":
        ok = claim["min"] == stats["min_len"] and claim["max"] == stats["max_len"]
        return "CONFIRMED" if ok else "REFUTED"

    if t == "vowel_share":
        # both share and counts; tolerate rounding on share (0.005), counts exact
        share_ok = abs(claim["share"] - stats["vowel_share"]) <= 0.005
        counts_ok = (claim["n_vowels"] == stats["n_vowels"] and claim["n_letters"] == stats["n_letters"])
        if share_ok and counts_ok:
            return "CONFIRMED"
        return "REFUTED"

    if t == "vowel_share_pct":
        share_ok = abs(claim["share_pct"] - stats["vowel_share"] * 100) <= 0.5
        letters_ok = claim["n_letters"] == stats["n_letters"]
        if share_ok and letters_ok:
            return "CONFIRMED"
        return "REFUTED"

    if t == "letter_freq":
        actual = stats["letter_counts"].get(claim["letter"], 0)
        return "CONFIRMED" if actual == claim["count"] else "REFUTED"

    if t == "bigram":
        actual = stats["bigrams"].get(claim["bigram"], 0)
        return "CONFIRMED" if actual == claim["count"] else "REFUTED"

    return "UNVERIFIABLE"


# ----- Ceiling: build 6 oracle claims from stats and verify (all should confirm) -----

def build_ceiling_claims(stats: dict) -> list[dict]:
    """Oracle: build up to 6 ground-truth claims from text statistics.
    These simulate what a perfect describer would report.
    """
    claims = []
    claims.append({"type": "n_tokens", "value": stats["n_tokens"]})
    claims.append({"type": "mean_len", "value": round(stats["mean_len"], 2)})
    claims.append({"type": "token_range", "min": stats["min_len"], "max": stats["max_len"]})
    claims.append({
        "type": "vowel_share",
        "share": round(stats["vowel_share"], 3),
        "n_vowels": stats["n_vowels"],
        "n_letters": stats["n_letters"],
    })
    # top letter
    if stats["letter_counts"]:
        top_letter, top_n = stats["letter_counts"].most_common(1)[0]
        claims.append({"type": "letter_freq", "letter": top_letter, "count": top_n})
    # top bigram
    if stats["bigrams"]:
        top_bg, top_bn = stats["bigrams"].most_common(1)[0]
        claims.append({"type": "bigram", "bigram": top_bg, "count": top_bn})
    return claims


# ----- Main -----

def main():
    records = []
    with IN_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    out_lines = []
    for rec in records:
        seq_idx = rec["seq_idx"]
        text = (rec.get("sequence", "") or "") + (rec.get("continuation", "") or "")
        desc = rec.get("generated_description", "") or ""

        stats = compute_stats(text)

        # Extracted claims from generated description
        claims = extract_claims(desc)
        n_claims = len(claims)
        verdicts = [verify_claim(c, stats) for c in claims]
        n_confirmed = sum(1 for v in verdicts if v == "CONFIRMED")
        n_refuted = sum(1 for v in verdicts if v == "REFUTED")
        n_unverifiable = sum(1 for v in verdicts if v == "UNVERIFIABLE")
        bridge_score = (n_confirmed / n_claims) if n_claims else 0.0

        # Ceiling: oracle
        ceiling_claims = build_ceiling_claims(stats)
        ceiling_verdicts = [verify_claim(c, stats) for c in ceiling_claims]
        ceiling_n_claims = len(ceiling_claims)
        ceiling_n_confirmed = sum(1 for v in ceiling_verdicts if v == "CONFIRMED")
        ceiling_score = (ceiling_n_confirmed / ceiling_n_claims) if ceiling_n_claims else 0.0

        out_lines.append({
            "seq_idx": seq_idx,
            "n_claims": n_claims,
            "n_confirmed": n_confirmed,
            "n_refuted": n_refuted,
            "n_unverifiable": n_unverifiable,
            "bridge_score": round(bridge_score, 4),
            "ceiling_n_claims": ceiling_n_claims,
            "ceiling_n_confirmed": ceiling_n_confirmed,
            "ceiling_score": round(ceiling_score, 4),
        })

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for row in out_lines:
            f.write(json.dumps(row) + "\n")

    # summary
    if out_lines:
        avg_bridge = sum(r["bridge_score"] for r in out_lines) / len(out_lines)
        avg_ceiling = sum(r["ceiling_score"] for r in out_lines) / len(out_lines)
        total_claims = sum(r["n_claims"] for r in out_lines)
        total_confirmed = sum(r["n_confirmed"] for r in out_lines)
        total_refuted = sum(r["n_refuted"] for r in out_lines)
        total_unver = sum(r["n_unverifiable"] for r in out_lines)
        print(f"Records: {len(out_lines)}")
        print(f"Total claims: {total_claims} (confirmed={total_confirmed}, refuted={total_refuted}, unverifiable={total_unver})")
        print(f"Avg bridge_score: {avg_bridge:.4f}")
        print(f"Avg ceiling_score: {avg_ceiling:.4f}")


if __name__ == "__main__":
    main()
