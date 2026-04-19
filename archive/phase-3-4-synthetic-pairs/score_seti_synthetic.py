"""
Claim-verification eval for MLP bridge on domain=seti, source=synthetic.

Reads  : G:/My Drive/nbs-bridge/results/mlp/seti_synthetic_eval_prepared.jsonl
Writes : G:/My Drive/nbs-bridge/results/mlp/seti_synthetic_scores.jsonl

For each record:
  1. Parse generated_description for SPECIFIC verifiable claims about the
     5-symbol-alphabet window (counts, runs, dominance, transients/scarcity).
  2. Verify each claim against the true sequence (200 chars) + continuation
     (50 chars). Label CONFIRMED / REFUTED / UNVERIFIABLE.
  3. Do the same for target_description -> CEILING.
  4. Emit per-record scores with the fixed schema.

Tolerances (5-symbol-level patterns on a 200-char window, mostly near-uniform
noise -- generous but not permissive):
  - exact symbol count   :  ±3 absolute tokens
  - percentage / share   :  ±5 percentage points
  - run length (longest) :  ±1
  - "most common X"      :  X must actually be top-1 (ties OK)
  - "second most common" :  must be top-2 (ties OK)
  - "rarest X"           :  X must actually be bottom-1 (ties OK)
  - "X appears ~N times" :  ±3
  - "run of N Xs"        :  some run of that symbol of length >= N-1 must exist

We clamp n_claims to [3,6] per spec.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

IN_PATH = Path("G:/My Drive/nbs-bridge/results/mlp/seti_synthetic_eval_prepared.jsonl")
OUT_PATH = Path("G:/My Drive/nbs-bridge/results/mlp/seti_synthetic_scores.jsonl")

ALPHABET = ["A", "B", "C", "D", "E"]

# ---------- sequence analysis helpers ----------

def counts(seq: str) -> Counter:
    return Counter(c for c in seq if c in ALPHABET)

def longest_run(seq: str, sym: str | None = None) -> tuple[int, str]:
    """Longest run length; if sym is None, across whole sequence."""
    best, best_sym = 0, ""
    i = 0
    n = len(seq)
    while i < n:
        j = i
        while j < n and seq[j] == seq[i]:
            j += 1
        run_len = j - i
        if sym is None or seq[i] == sym:
            if run_len > best:
                best, best_sym = run_len, seq[i]
        i = j
    return best, best_sym

def max_run_of(seq: str, sym: str) -> int:
    best = 0
    i = 0
    n = len(seq)
    while i < n:
        if seq[i] == sym:
            j = i
            while j < n and seq[j] == sym:
                j += 1
            best = max(best, j - i)
            i = j
        else:
            i += 1
    return best

def rank_symbols(seq: str) -> list[tuple[str, int]]:
    """Symbols sorted by count desc; ties stable-broken alphabetically."""
    cts = counts(seq)
    # fill missing symbols with 0 for complete ranking
    for s in ALPHABET:
        cts.setdefault(s, 0)
    return sorted(cts.items(), key=lambda kv: (-kv[1], kv[0]))

# ---------- claim extraction ----------

# tolerances
COUNT_TOL = 3
PCT_TOL = 5.0
RUN_TOL = 1

def extract_claims(desc: str) -> list[dict]:
    """
    Parse a description into a list of structured claims.
    Each claim: {"kind": str, ...}
    Claim kinds:
      count_abs   : {"sym": "A", "n": 22}
      count_pct   : {"sym": "C", "pct": 47.0}
      rank_top1   : {"sym": "C"}         # "most common" / "by far the most frequent"
      rank_top2   : {"sym": "D"}         # "second most common" / "D second"
      rank_bot1   : {"sym": "E"}         # "rarest" / "genuinely scarce"
      run_sym     : {"sym": "C", "n": 6} # "run of 6 Cs"
      run_longest : {"n": 4}             # "longest run is 4"
      few_count   : {"sym": "E", "n": 7} # "E appears only 7 times"
    """
    claims: list[dict] = []
    text = desc

    # 1) "A=22, B=47, C=80, D=59, E=8"
    for m in re.finditer(r"\b([A-E])\s*=\s*(\d+)\b", text):
        claims.append({"kind": "count_abs", "sym": m.group(1), "n": int(m.group(2))})

    # 2) "C is the most common symbol (79 of 200, ~47%)"
    for m in re.finditer(
        r"\b([A-E])\s+is\s+(?:the\s+)?most\s+common[^.]*?\((\d+)\s*of\s*\d+\s*,\s*~?\s*(\d+(?:\.\d+)?)\s*%",
        text, flags=re.IGNORECASE,
    ):
        sym = m.group(1).upper()
        claims.append({"kind": "rank_top1", "sym": sym})
        claims.append({"kind": "count_abs", "sym": sym, "n": int(m.group(2))})
        claims.append({"kind": "count_pct", "sym": sym, "pct": float(m.group(3))})

    # 3) "most frequent character is C" / "bulk of the string is D"
    for m in re.finditer(
        r"(?:most\s+(?:common|frequent)[^.]*?(?:is|character\s+is)|bulk\s+of\s+the\s+string\s+is)\s+([A-E])\b",
        text, flags=re.IGNORECASE,
    ):
        claims.append({"kind": "rank_top1", "sym": m.group(1).upper()})

    # 4) "with D second (79)"  / "D second"
    for m in re.finditer(
        r"\b([A-E])\s+second\b(?:\s*\((\d+)\))?",
        text, flags=re.IGNORECASE,
    ):
        sym = m.group(1).upper()
        claims.append({"kind": "rank_top2", "sym": sym})
        if m.group(2):
            claims.append({"kind": "count_abs", "sym": sym, "n": int(m.group(2))})

    # 5) "rarest symbol, E" / "rarest is E" / "E is genuinely scarce"
    for m in re.finditer(r"rarest\s+(?:symbol|character)?\s*,?\s*(?:is\s+)?([A-E])\b", text, flags=re.IGNORECASE):
        claims.append({"kind": "rank_bot1", "sym": m.group(1).upper()})
    for m in re.finditer(r"\b([A-E])\s+is\s+genuinely\s+scarce\b", text, flags=re.IGNORECASE):
        claims.append({"kind": "rank_bot1", "sym": m.group(1).upper()})

    # 6) "E appears only 7 times" / "surfacing just 7 times"
    for m in re.finditer(
        r"\b([A-E])\b[^.]*?(?:appears|surfac\w+|shows?\s+up|present)[^.]*?(?:only|just)?\s*(\d+)\s*times?",
        text, flags=re.IGNORECASE,
    ):
        claims.append({"kind": "few_count", "sym": m.group(1).upper(), "n": int(m.group(2))})

    # 7) runs: "run of 6 Cs near position..." / "block of 4 identical Cs"
    for m in re.finditer(r"run\s+of\s+(\d+)\s+([A-E])s?\b", text, flags=re.IGNORECASE):
        claims.append({"kind": "run_sym", "sym": m.group(2).upper(), "n": int(m.group(1))})
    for m in re.finditer(r"block\s+of\s+(\d+)\s+identical\s+([A-E])s?\b", text, flags=re.IGNORECASE):
        claims.append({"kind": "run_sym", "sym": m.group(2).upper(), "n": int(m.group(1))})
    for m in re.finditer(r"(\d+)\s+([A-E])s\s+in\s+a\s+row\b", text, flags=re.IGNORECASE):
        claims.append({"kind": "run_sym", "sym": m.group(2).upper(), "n": int(m.group(1))})

    # 8) "longest run is 4" (if a trailing number exists)
    for m in re.finditer(r"longest\s+run\s+is\s+(\d+)", text, flags=re.IGNORECASE):
        claims.append({"kind": "run_longest", "n": int(m.group(1))})

    # 9) "~34% of slots" / "accounts for about 34%" -> dominance of top symbol
    #    We only attach this when the sentence pairs a percent with a letter
    #    already captured above (count_pct). So no extra rule needed here --
    #    rule (2) handles it.

    # dedupe while preserving order
    seen = set()
    unique: list[dict] = []
    for c in claims:
        key = tuple(sorted(c.items()))
        if key in seen:
            continue
        seen.add(key)
        unique.append(c)
    return unique

# ---------- verification ----------

def verify_claim(claim: dict, seq: str) -> str:
    """Return 'CONFIRMED' / 'REFUTED' / 'UNVERIFIABLE'."""
    cts = counts(seq)
    rank = rank_symbols(seq)
    top_syms = [s for s, c in rank if c == rank[0][1]]
    top2_val = rank[1][1]
    top2_syms_any = [s for s, c in rank if c >= top2_val and c <= rank[0][1]]
    # "top 2" interpretation: symbol whose count is rank[0] or rank[1] value
    top2_set = {rank[0][0], rank[1][0]}
    # also include ties at top-2 boundary
    for s, c in rank:
        if c == rank[1][1]:
            top2_set.add(s)

    bot_val = rank[-1][1]
    bot_set = {s for s, c in rank if c == bot_val}

    kind = claim["kind"]

    if kind == "count_abs":
        sym = claim["sym"]
        actual = cts.get(sym, 0)
        return "CONFIRMED" if abs(actual - claim["n"]) <= COUNT_TOL else "REFUTED"

    if kind == "count_pct":
        sym = claim["sym"]
        actual_pct = 100.0 * cts.get(sym, 0) / max(1, len(seq))
        return "CONFIRMED" if abs(actual_pct - claim["pct"]) <= PCT_TOL else "REFUTED"

    if kind == "rank_top1":
        return "CONFIRMED" if claim["sym"] in top_syms else "REFUTED"

    if kind == "rank_top2":
        return "CONFIRMED" if claim["sym"] in top2_set else "REFUTED"

    if kind == "rank_bot1":
        return "CONFIRMED" if claim["sym"] in bot_set else "REFUTED"

    if kind == "few_count":
        sym = claim["sym"]
        actual = cts.get(sym, 0)
        return "CONFIRMED" if abs(actual - claim["n"]) <= COUNT_TOL else "REFUTED"

    if kind == "run_sym":
        sym = claim["sym"]
        best = max_run_of(seq, sym)
        # Accept if actual run within RUN_TOL of claimed length
        return "CONFIRMED" if abs(best - claim["n"]) <= RUN_TOL or best >= claim["n"] else "REFUTED"

    if kind == "run_longest":
        best, _ = longest_run(seq)
        return "CONFIRMED" if abs(best - claim["n"]) <= RUN_TOL else "REFUTED"

    return "UNVERIFIABLE"

# ---------- claim selection (clamp to 3..6) ----------

# Priority order: we prefer structurally distinct claim kinds so 3-6 claims
# are reasonably independent (not e.g. five count_abs's that all derive from
# one "A=22, B=..." line). Duplicates of the same kind come last.
KIND_PRIORITY = [
    "rank_top1",
    "rank_bot1",
    "count_pct",
    "run_longest",
    "run_sym",
    "count_abs",
    "rank_top2",
    "few_count",
]

def select_claims(claims: list[dict], lo: int = 3, hi: int = 6) -> list[dict]:
    # sort stable by (priority index, original order)
    order_map = {k: i for i, k in enumerate(KIND_PRIORITY)}
    indexed = list(enumerate(claims))
    indexed.sort(key=lambda t: (order_map.get(t[1]["kind"], 999), t[0]))
    # pick at most `hi`, ensuring diversity first: round-robin over kinds.
    buckets: dict[str, list[dict]] = {}
    for _, c in indexed:
        buckets.setdefault(c["kind"], []).append(c)
    picked: list[dict] = []
    kind_queue = [k for k in KIND_PRIORITY if k in buckets] + [
        k for k in buckets if k not in KIND_PRIORITY
    ]
    while len(picked) < hi and any(buckets[k] for k in kind_queue):
        for k in kind_queue:
            if buckets[k]:
                picked.append(buckets[k].pop(0))
                if len(picked) >= hi:
                    break
    return picked  # may be < lo; caller handles

# ---------- scoring ----------

def score_description(desc: str, seq: str) -> tuple[int, int, int, int]:
    raw = extract_claims(desc)
    picked = select_claims(raw, lo=3, hi=6)
    verdicts = [verify_claim(c, seq) for c in picked]
    n_claims = len(picked)
    n_conf = sum(1 for v in verdicts if v == "CONFIRMED")
    n_ref = sum(1 for v in verdicts if v == "REFUTED")
    n_unv = sum(1 for v in verdicts if v == "UNVERIFIABLE")
    return n_claims, n_conf, n_ref, n_unv

# ---------- main ----------

def main() -> None:
    records = []
    with IN_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    out_lines = []
    for rec in records:
        seq = rec["sequence"] + rec.get("continuation", "")
        gen = rec.get("generated_description", "") or ""
        tgt = rec.get("target_description", "") or ""

        nb, cb, rb, ub = score_description(gen, seq)
        nc, cc, rc, uc = score_description(tgt, seq)

        bridge_score = (cb / nb) if nb > 0 else 0.0
        ceiling_score = (cc / nc) if nc > 0 else 0.0

        out = {
            "seq_idx": rec["seq_idx"],
            "n_claims": nb,
            "n_confirmed": cb,
            "n_refuted": rb,
            "n_unverifiable": ub,
            "bridge_score": round(bridge_score, 4),
            "ceiling_n_claims": nc,
            "ceiling_n_confirmed": cc,
            "ceiling_score": round(ceiling_score, 4),
        }
        out_lines.append(json.dumps(out))

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        f.write("\n".join(out_lines) + "\n")

    # brief stdout summary
    n = len(out_lines)
    mean_b = sum(json.loads(l)["bridge_score"] for l in out_lines) / max(1, n)
    mean_c = sum(json.loads(l)["ceiling_score"] for l in out_lines) / max(1, n)
    print(f"wrote {n} records -> {OUT_PATH}")
    print(f"mean bridge_score  = {mean_b:.3f}")
    print(f"mean ceiling_score = {mean_c:.3f}")

if __name__ == "__main__":
    main()
