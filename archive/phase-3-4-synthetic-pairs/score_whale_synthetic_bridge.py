"""
Claim-verification eval for MLP bridge on domain=whale, source=synthetic.

For each record in whale_synthetic_eval_prepared.jsonl:
  1) Extract verifiable claims from `generated_description`
  2) Verify claims against actual sequence+continuation characters
  3) Compute CEILING: generate claims directly from truth, verify them

Writes whale_synthetic_scores.jsonl with per-record:
  seq_idx, n_claims, n_confirmed, n_refuted, n_unverifiable,
  bridge_score, ceiling_n_claims, ceiling_n_confirmed, ceiling_score
"""

import json
import re
from collections import Counter
from pathlib import Path

IN_PATH = Path("G:/My Drive/nbs-bridge/results/mlp/whale_synthetic_eval_prepared.jsonl")
OUT_PATH = Path("G:/My Drive/nbs-bridge/results/mlp/whale_synthetic_scores.jsonl")

NUM_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
}


# ---------- Sequence statistics ----------

def runs(text):
    """Return list of (char, run_length) for consecutive runs."""
    if not text:
        return []
    out = []
    prev = text[0]
    count = 1
    for ch in text[1:]:
        if ch == prev:
            count += 1
        else:
            out.append((prev, count))
            prev = ch
            count = 1
    out.append((prev, count))
    return out


def stats(sequence, continuation):
    full = sequence + continuation
    counter = Counter(sequence)
    n = len(sequence)
    distinct = sorted(counter.keys())
    rs = runs(sequence)
    longest_char, longest_len = max(rs, key=lambda x: x[1]) if rs else (None, 0)
    changes = sum(1 for i in range(1, n) if sequence[i] != sequence[i-1])
    change_rate = changes / max(1, n - 1)
    dominant_char, dominant_count = counter.most_common(1)[0] if counter else (None, 0)
    dominant_frac = dominant_count / n if n else 0
    # early vs late halves
    half = n // 2
    early = Counter(sequence[:half])
    late = Counter(sequence[half:])
    return {
        "sequence": sequence,
        "continuation": continuation,
        "full": full,
        "counter": counter,
        "n": n,
        "distinct": distinct,
        "n_distinct": len(distinct),
        "runs": rs,
        "n_runs": len(rs),
        "longest_char": longest_char,
        "longest_len": longest_len,
        "change_rate": change_rate,
        "dominant_char": dominant_char,
        "dominant_frac": dominant_frac,
        "early": early,
        "late": late,
    }


# ---------- Claim extraction ----------

def split_sentences(text):
    text = text.replace("\u2014", " -- ").replace("\u2013", " -- ")
    # Split on period/semicolon/em-dash separators
    parts = re.split(r"(?<=[.!?])\s+|;\s+|\s--\s", text)
    return [p.strip() for p in parts if p.strip()]


def parse_number(token):
    """Parse integer, float, percent, or number word."""
    token = token.strip().lower().rstrip(".,")
    if token in NUM_WORDS:
        return float(NUM_WORDS[token])
    token = token.replace("%", "")
    try:
        return float(token)
    except ValueError:
        return None


def find_number_near(sentence, keyword_regex):
    """Find a number that appears near a keyword."""
    m = re.search(keyword_regex, sentence, re.IGNORECASE)
    if not m:
        return None
    # look for first number-like token in sentence
    nums = re.findall(r"\d+\.?\d*", sentence)
    if nums:
        return float(nums[0])
    # word numbers
    for w, v in NUM_WORDS.items():
        if re.search(rf"\b{w}\b", sentence, re.IGNORECASE):
            return float(v)
    return None


def extract_claims(description, max_claims=6):
    """Extract 3-6 verifiable claims from description as dicts with type + params."""
    claims = []
    sentences = split_sentences(description)

    for sent in sentences:
        sent_l = sent.lower()

        # Claim: spreads across N symbols
        m = re.search(r"(\d+)\s*(?:distinct\s+)?symbols?", sent_l)
        if m:
            claims.append({"type": "n_distinct", "value": int(m.group(1)), "sent": sent})

        # Claim: list of specific symbols in parentheses (A, B, C, D) or similar
        m = re.search(r"\(([A-Z](?:\s*,\s*[A-Z])+)\)", sent)
        if m:
            syms = [s.strip() for s in m.group(1).split(",")]
            claims.append({"type": "symbol_set", "value": set(syms), "sent": sent})

        # Claim: N distinct runs / N runs
        m = re.search(r"(\d+)\s+(?:distinct\s+)?runs?\b", sent_l)
        if m:
            claims.append({"type": "n_runs", "value": int(m.group(1)), "sent": sent})

        # Claim: longest run is N consecutive X
        m = re.search(r"longest\s+run[^.]*?(\d+)\s+consecutive\s+([A-Z])", sent, re.IGNORECASE)
        if m:
            claims.append({
                "type": "longest_run",
                "length": int(m.group(1)),
                "char": m.group(2).upper(),
                "sent": sent,
            })

        # Claim: change rate / transition rate near X
        m = re.search(r"(?:per[-\s]step\s+change\s+rate|transition\s+rate)[^.]*?(\d+\.\d+)", sent_l)
        if m:
            claims.append({"type": "change_rate", "value": float(m.group(1)), "sent": sent})

        # Claim: led by X at about N%
        m = re.search(r"led\s+by\s+([A-Z])[^.]*?(\d+)\s*%", sent, re.IGNORECASE)
        if m:
            claims.append({
                "type": "dominant_pct",
                "char": m.group(1).upper(),
                "pct": float(m.group(2)),
                "sent": sent,
            })

        # Claim: dominant / most common symbol is X
        if "dominant" in sent_l or "majority" in sent_l or "most common" in sent_l:
            m = re.search(r"(?:dominant|most\s+common|majority)[^.]*?\b([A-Z])\b", sent)
            if m:
                claims.append({"type": "dominant_char", "char": m.group(1).upper(), "sent": sent})

        # Claim: current symbol is X / ends with X
        m = re.search(r"(?:current\s+symbol\s+is|ends?\s+(?:with|on))\s+([A-Z])", sent, re.IGNORECASE)
        if m:
            claims.append({"type": "last_char", "char": m.group(1).upper(), "sent": sent})

        # Claim: drift direction (D-rich early, B-dominated tail etc.)
        m = re.search(r"\b([A-Z])-rich\s+early", sent)
        if m:
            claims.append({"type": "early_dominant", "char": m.group(1).upper(), "sent": sent})
        m = re.search(r"\b([A-Z])-dominated\s+tail", sent)
        if m:
            claims.append({"type": "late_dominant", "char": m.group(1).upper(), "sent": sent})

        # Claim: from X toward Y (drift)
        m = re.search(r"from\s+([A-Z])\s+toward\s+([A-Z])", sent)
        if m:
            claims.append({
                "type": "drift",
                "from_char": m.group(1).upper(),
                "to_char": m.group(2).upper(),
                "sent": sent,
            })

        # Claim: "B-majority" or "X-majority"
        m = re.search(r"\b([A-Z])-majority", sent)
        if m:
            claims.append({"type": "dominant_char", "char": m.group(1).upper(), "sent": sent})

        # Claim: "over N steps"
        m = re.search(r"over\s+(\d+)\s+steps", sent_l)
        if m:
            claims.append({"type": "length", "value": int(m.group(1)), "sent": sent})

        # Claim: "variability is high / low"
        if re.search(r"variability\s+is\s+high", sent_l):
            claims.append({"type": "variability", "value": "high", "sent": sent})
        if re.search(r"variability\s+is\s+low", sent_l):
            claims.append({"type": "variability", "value": "low", "sent": sent})

        if len(claims) >= max_claims:
            break

    # Deduplicate by (type, params) footprint
    seen = set()
    unique = []
    for c in claims:
        key = (c["type"], tuple(sorted((k, str(v)) for k, v in c.items() if k != "sent")))
        if key not in seen:
            seen.add(key)
            unique.append(c)
    return unique[:max_claims]


# ---------- Verification ----------

def approx(a, b, tol_abs=None, tol_rel=None):
    if tol_abs is not None and abs(a - b) <= tol_abs:
        return True
    if tol_rel is not None and abs(a - b) <= tol_rel * max(1.0, abs(b)):
        return True
    return False


def verify_claim(claim, s):
    """Return 'CONFIRMED' / 'REFUTED' / 'UNVERIFIABLE'."""
    t = claim["type"]

    if t == "n_distinct":
        # tolerate ±1
        return "CONFIRMED" if abs(claim["value"] - s["n_distinct"]) <= 1 else "REFUTED"

    if t == "symbol_set":
        # all claimed symbols should appear; tolerate one miss
        actual = set(s["distinct"])
        claimed = claim["value"]
        missing = claimed - actual
        extra = actual - claimed
        if len(missing) == 0 and len(extra) <= 1:
            return "CONFIRMED"
        if len(missing) <= 1 and len(extra) <= 2:
            return "CONFIRMED"
        return "REFUTED"

    if t == "n_runs":
        # tolerate ±20% or ±3 whichever larger
        tol = max(3, int(0.20 * s["n_runs"]))
        return "CONFIRMED" if abs(claim["value"] - s["n_runs"]) <= tol else "REFUTED"

    if t == "longest_run":
        # character must match; length within 20%
        if claim["char"] != s["longest_char"]:
            return "REFUTED"
        tol = max(2, int(0.20 * s["longest_len"]))
        return "CONFIRMED" if abs(claim["length"] - s["longest_len"]) <= tol else "REFUTED"

    if t == "change_rate":
        return "CONFIRMED" if approx(claim["value"], s["change_rate"], tol_abs=0.03) else "REFUTED"

    if t == "dominant_pct":
        if claim["char"] != s["dominant_char"]:
            return "REFUTED"
        actual_pct = s["dominant_frac"] * 100
        return "CONFIRMED" if approx(claim["pct"], actual_pct, tol_abs=10) else "REFUTED"

    if t == "dominant_char":
        return "CONFIRMED" if claim["char"] == s["dominant_char"] else "REFUTED"

    if t == "last_char":
        return "CONFIRMED" if s["sequence"] and claim["char"] == s["sequence"][-1] else "REFUTED"

    if t == "early_dominant":
        if not s["early"]:
            return "UNVERIFIABLE"
        actual = s["early"].most_common(1)[0][0]
        return "CONFIRMED" if claim["char"] == actual else "REFUTED"

    if t == "late_dominant":
        if not s["late"]:
            return "UNVERIFIABLE"
        actual = s["late"].most_common(1)[0][0]
        return "CONFIRMED" if claim["char"] == actual else "REFUTED"

    if t == "drift":
        if not s["early"] or not s["late"]:
            return "UNVERIFIABLE"
        early_dom = s["early"].most_common(1)[0][0]
        late_dom = s["late"].most_common(1)[0][0]
        if early_dom == claim["from_char"] and late_dom == claim["to_char"]:
            return "CONFIRMED"
        return "REFUTED"

    if t == "length":
        return "CONFIRMED" if claim["value"] == s["n"] else "REFUTED"

    if t == "variability":
        # "high" ~ change_rate >= 0.05, "low" ~ < 0.05 (coarse)
        if claim["value"] == "high":
            return "CONFIRMED" if s["change_rate"] >= 0.05 else "REFUTED"
        if claim["value"] == "low":
            return "CONFIRMED" if s["change_rate"] < 0.05 else "REFUTED"
        return "UNVERIFIABLE"

    return "UNVERIFIABLE"


# ---------- Ceiling: auto-generate claims from truth ----------

def ceiling_claims(s):
    """Generate 3-6 ground-truth claims directly from the actual sequence."""
    claims = []
    # length
    claims.append({"type": "length", "value": s["n"], "sent": "(ceiling) length"})
    # n distinct
    claims.append({"type": "n_distinct", "value": s["n_distinct"], "sent": "(ceiling) distinct"})
    # symbol set
    claims.append({"type": "symbol_set", "value": set(s["distinct"]), "sent": "(ceiling) symbol set"})
    # n runs
    claims.append({"type": "n_runs", "value": s["n_runs"], "sent": "(ceiling) runs"})
    # longest run
    if s["longest_char"] is not None:
        claims.append({
            "type": "longest_run",
            "length": s["longest_len"],
            "char": s["longest_char"],
            "sent": "(ceiling) longest run",
        })
    # dominant char
    if s["dominant_char"] is not None:
        claims.append({"type": "dominant_char", "char": s["dominant_char"], "sent": "(ceiling) dominant"})
    return claims[:6]


# ---------- Main ----------

def main():
    with IN_PATH.open("r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    out_lines = []
    bridge_scores = []
    ceiling_scores = []

    for rec in records:
        seq = rec["sequence"]
        cont = rec.get("continuation", "")
        gen_desc = rec.get("generated_description", "") or ""
        s = stats(seq, cont)

        # ---- Bridge claims
        claims = extract_claims(gen_desc, max_claims=6)
        if len(claims) < 3:
            # pad with trivially derivable claims if extractor found too few
            # to avoid divide-by-zero; these are from the generated text so unverifiable
            pass

        n_c = n_r = n_u = 0
        for c in claims:
            v = verify_claim(c, s)
            if v == "CONFIRMED":
                n_c += 1
            elif v == "REFUTED":
                n_r += 1
            else:
                n_u += 1
        n_claims = len(claims)
        bridge_score = (n_c / n_claims) if n_claims else 0.0

        # ---- Ceiling claims
        cc = ceiling_claims(s)
        cc_c = 0
        for c in cc:
            if verify_claim(c, s) == "CONFIRMED":
                cc_c += 1
        ceiling_n_claims = len(cc)
        ceiling_score = (cc_c / ceiling_n_claims) if ceiling_n_claims else 0.0

        out = {
            "seq_idx": rec["seq_idx"],
            "n_claims": n_claims,
            "n_confirmed": n_c,
            "n_refuted": n_r,
            "n_unverifiable": n_u,
            "bridge_score": bridge_score,
            "ceiling_n_claims": ceiling_n_claims,
            "ceiling_n_confirmed": cc_c,
            "ceiling_score": ceiling_score,
        }
        out_lines.append(out)
        bridge_scores.append(bridge_score)
        ceiling_scores.append(ceiling_score)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for line in out_lines:
            f.write(json.dumps(line) + "\n")

    mean_bridge = sum(bridge_scores) / len(bridge_scores) if bridge_scores else 0.0
    mean_ceiling = sum(ceiling_scores) / len(ceiling_scores) if ceiling_scores else 0.0
    print(f"PATH: {OUT_PATH}")
    print(f"LINES: {len(out_lines)}")
    print(f"MEAN_BRIDGE_SCORE: {mean_bridge:.4f}")
    print(f"MEAN_CEILING_SCORE: {mean_ceiling:.4f}")


if __name__ == "__main__":
    main()
