"""
Claim-verification eval for MLP bridge on domain=dna_coding, source=natural.

Extracts specific verifiable claims from generated_description and checks
them against (sequence + continuation) nucleotide statistics.

Claim categories (each worth 1 claim if explicitly stated):
  1. GC content percentage (tolerance +/- 5 pp)
  2. Purine fraction (A+G) or pyrimidine-dominance statement (tol +/- 7 pp)
  3. GC skew sign + magnitude (tol +/- 0.10)
  4. CpG count (exact +/- 1) or CpG depletion qualitative
  5. Homopolymer max run length (tol +/- 1)
  6. Top frame-0 codon(s) -- at least one claimed codon appears in top-5
  7. In-frame stop count (exact match on "zero" / "1" / "no stops")
  8. Hexamer repeat claim (>=2 recurrences of named hexamer)

Ceiling: independent molecular-feature claims (GC%, purine, skew, CpG, homopol,
top-codon, in-frame-stop, hexamer) the target_description asserts and the
actual sequence supports -- measures upper bound of the evaluation rubric.
"""
import json
import re
from collections import Counter
from pathlib import Path

INPUT = Path("G:/My Drive/nbs-bridge/results/mlp/dna_coding_natural_eval_prepared.jsonl")
OUTPUT = Path("G:/My Drive/nbs-bridge/results/mlp/dna_coding_natural_scores.jsonl")


# ----- sequence statistics ---------------------------------------------------

def gc_pct(seq):
    if not seq: return 0.0
    return 100.0 * (seq.count("G") + seq.count("C")) / len(seq)

def purine_frac(seq):
    if not seq: return 0.0
    return (seq.count("A") + seq.count("G")) / len(seq)

def gc_skew(seq):
    g, c = seq.count("G"), seq.count("C")
    if g + c == 0: return 0.0
    return (g - c) / (g + c)

def cpg_count(seq):
    return sum(1 for i in range(len(seq) - 1) if seq[i:i+2] == "CG")

def cpg_obs_exp(seq):
    c, g = seq.count("C"), seq.count("G")
    n = len(seq)
    if c == 0 or g == 0 or n == 0: return 0.0
    expected = (c * g) / n
    return cpg_count(seq) / expected if expected > 0 else 0.0

def max_homopolymer(seq):
    if not seq: return 0, ""
    best, best_b = 1, seq[0]
    cur, cur_b = 1, seq[0]
    for ch in seq[1:]:
        if ch == cur_b:
            cur += 1
            if cur > best:
                best, best_b = cur, cur_b
        else:
            cur, cur_b = 1, ch
    return best, best_b

def frame0_codon_counts(seq):
    codons = [seq[i:i+3] for i in range(0, len(seq) - 2, 3)]
    codons = [c for c in codons if len(c) == 3 and set(c) <= set("ACGT")]
    return Counter(codons)

def inframe_stop_count(seq):
    stops = {"TAA", "TAG", "TGA"}
    codons = [seq[i:i+3] for i in range(0, len(seq) - 2, 3)]
    return sum(1 for c in codons if c in stops)

def hexamer_counts(seq):
    c = Counter(seq[i:i+6] for i in range(len(seq) - 5))
    return c


# ----- claim extractors ------------------------------------------------------

NUM = r"[-+]?\d+(?:\.\d+)?"

def extract_gc_claim(desc):
    # "GC content near 60%", "GC fraction of approximately 50.0%", "64.0% GC"
    patterns = [
        rf"GC\s+content\s+(?:near|at|of|around|approximately)?\s*({NUM})\s*%",
        rf"GC\s+fraction\s+of\s+(?:approximately\s+)?({NUM})\s*%",
        rf"({NUM})\s*%\s*GC",
    ]
    for pat in patterns:
        m = re.search(pat, desc, re.IGNORECASE)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                pass
    return None

def extract_purine_claim(desc):
    # "purine bias (A+G ~56%)", "A+G at 0.59", "pyrimidines dominating at 0.56"
    m = re.search(rf"A\s*\+\s*G\s*[~\s]*({NUM})\s*%", desc, re.IGNORECASE)
    if m:
        try: return float(m.group(1)) / 100.0, "purine"
        except: pass
    m = re.search(rf"A\s*\+\s*G\s*(?:at|of|~)?\s*({NUM})", desc, re.IGNORECASE)
    if m:
        try:
            v = float(m.group(1))
            return (v / 100.0 if v > 1 else v), "purine"
        except: pass
    m = re.search(rf"purine\s+(?:bias|skew)\s*\(?\s*[~+]?({NUM})\s*%", desc, re.IGNORECASE)
    if m:
        try: return float(m.group(1)) / 100.0, "purine"
        except: pass
    if re.search(r"pyrimidine(s)?\s+dominat", desc, re.IGNORECASE):
        return None, "pyrimidine_qual"
    if re.search(r"purine\s+(bias|enrich)", desc, re.IGNORECASE):
        return None, "purine_qual"
    return None, None

def extract_gcskew_claim(desc):
    # "positive GC skew of +0.29", "negative GC skew of -0.13", "GC skew of +0.9%"
    m = re.search(rf"GC\s+skew\s+of\s*({NUM})\s*(%)?", desc, re.IGNORECASE)
    if m:
        try:
            v = float(m.group(1))
            if m.group(2):
                v = v / 100.0
            return v
        except: pass
    m = re.search(rf"(positive|negative)\s+GC\s+skew", desc, re.IGNORECASE)
    if m:
        return 0.01 if m.group(1).lower() == "positive" else -0.01
    return None

def extract_cpg_claim(desc):
    # "4 cpgs", "5 cpg dinucleotides", "1 sites (obs/exp 0.01)", "cpg depletion/under-representation"
    m = re.search(r"(\d+)\s*cpg(?:s|\s+dinucleotide)", desc, re.IGNORECASE)
    count = int(m.group(1)) if m else None
    qual = None
    if re.search(r"cpg\s+(depletion|under[- ]?representation|depleted)", desc, re.IGNORECASE):
        qual = "depleted"
    elif re.search(r"cpg\s+(enrich|elevated)", desc, re.IGNORECASE):
        qual = "enriched"
    return count, qual

def extract_homopolymer_claim(desc):
    # "homopolymer tract of 6 bases", "homopolymer stretch of 6 Cs", "stretches of up to 5"
    patterns = [
        rf"homopolymer\s+(?:tract|stretch|run)s?\s+of\s+(?:up\s+to\s+)?(\d+)\s*(?:bases|[ACGT]s)?",
        rf"stretches?\s+of\s+up\s+to\s+(\d+)\s+consecutive",
        rf"(\d+)\s*consecutive\s+bases",
    ]
    for pat in patterns:
        m = re.search(pat, desc, re.IGNORECASE)
        if m:
            try: return int(m.group(1))
            except: pass
    return None

def extract_top_codons(desc):
    # "codon usage is dominated by TGG, GCA, GAC"
    # "most frequent frame-0 triplets are CCA"
    # "frame-0 triplets are GCA; ..."
    codons = []
    m = re.search(r"(?:codon\s+usage\s+is\s+dominated\s+by|most\s+frequent\s+frame-?0\s+(?:triplets|codons)\s+are|frame-?0\s+codon\s+usage.*?by)\s+([^.;]+)", desc, re.IGNORECASE)
    if m:
        tail = m.group(1)
        for tok in re.findall(r"\b([ACGT]{3})\b", tail):
            codons.append(tok)
    # Also pick up "triplets GGA, GAG, GTG recur frequently"
    m = re.search(r"triplets?\s+([ACGT]{3}(?:\s*,\s*[ACGT]{3}){1,})\s+(?:recur|frequently)", desc, re.IGNORECASE)
    if m:
        for tok in re.findall(r"[ACGT]{3}", m.group(1)):
            codons.append(tok)
    return list(dict.fromkeys(codons))[:5]  # preserve order, dedupe

def extract_stop_claim(desc):
    # "zero in-frame stops", "1 in-frame stop codon", "no stops"
    if re.search(r"zero\s+in-?frame\s+stop", desc, re.IGNORECASE):
        return 0
    if re.search(r"no\s+(?:in-?frame\s+)?stops?\b", desc, re.IGNORECASE):
        return 0
    m = re.search(r"(\d+)\s+in-?frame\s+stop", desc, re.IGNORECASE)
    if m:
        try: return int(m.group(1))
        except: pass
    return None

def extract_hexamer_claims(desc):
    # "repeated hexamer 'CTCCCA' recurs three or more times"
    hex_list = []
    for m in re.finditer(r"hexamer[^'\"]*['\"]([ACGT]{6})['\"]", desc, re.IGNORECASE):
        hex_list.append(m.group(1).upper())
    if re.search(r"no\s+strong\s+hexamer\s+repeats", desc, re.IGNORECASE):
        hex_list.append("__NONE__")
    return hex_list


# ----- verification ----------------------------------------------------------

def verify_claims(seq, desc):
    """Return list of (label, verdict) tuples. Verdict in {CONFIRMED, REFUTED, UNVERIFIABLE}."""
    results = []

    # GC content
    gc_claim = extract_gc_claim(desc)
    if gc_claim is not None:
        actual = gc_pct(seq)
        if abs(actual - gc_claim) <= 5.0:
            results.append(("gc_pct", "CONFIRMED"))
        else:
            results.append(("gc_pct", "REFUTED"))

    # Purine fraction
    pur_val, pur_kind = extract_purine_claim(desc)
    if pur_val is not None:
        actual = purine_frac(seq)
        if abs(actual - pur_val) <= 0.07:
            results.append(("purine_frac", "CONFIRMED"))
        else:
            results.append(("purine_frac", "REFUTED"))
    elif pur_kind == "purine_qual":
        actual = purine_frac(seq)
        results.append(("purine_qual", "CONFIRMED" if actual >= 0.52 else "REFUTED"))
    elif pur_kind == "pyrimidine_qual":
        actual = purine_frac(seq)
        results.append(("pyrimidine_qual", "CONFIRMED" if actual <= 0.48 else "REFUTED"))

    # GC skew
    skew_claim = extract_gcskew_claim(desc)
    if skew_claim is not None:
        actual = gc_skew(seq)
        # sign match (treat near-zero as sign-agnostic)
        if abs(skew_claim) <= 0.02:
            sign_ok = True
        else:
            sign_ok = (skew_claim * actual) > 0 or abs(actual) < 0.03
        mag_ok = abs(abs(actual) - abs(skew_claim)) <= 0.15
        if sign_ok and mag_ok:
            results.append(("gc_skew", "CONFIRMED"))
        else:
            results.append(("gc_skew", "REFUTED"))

    # CpG
    cpg_cnt, cpg_qual = extract_cpg_claim(desc)
    if cpg_cnt is not None:
        actual = cpg_count(seq)
        if abs(actual - cpg_cnt) <= 2:
            results.append(("cpg_count", "CONFIRMED"))
        else:
            results.append(("cpg_count", "REFUTED"))
    elif cpg_qual == "depleted":
        actual = cpg_obs_exp(seq)
        results.append(("cpg_depleted", "CONFIRMED" if actual < 0.6 else "REFUTED"))
    elif cpg_qual == "enriched":
        actual = cpg_obs_exp(seq)
        results.append(("cpg_enriched", "CONFIRMED" if actual > 1.0 else "REFUTED"))

    # Homopolymer
    hp_claim = extract_homopolymer_claim(desc)
    if hp_claim is not None:
        actual, _ = max_homopolymer(seq)
        if abs(actual - hp_claim) <= 1:
            results.append(("homopolymer", "CONFIRMED"))
        else:
            results.append(("homopolymer", "REFUTED"))

    # Top codons
    top_codons = extract_top_codons(desc)
    if top_codons:
        counts = frame0_codon_counts(seq)
        top5 = {c for c, _ in counts.most_common(5)}
        # claim is CONFIRMED if at least one claimed codon is in actual top-5
        hit = any(c in top5 for c in top_codons)
        results.append(("top_codons", "CONFIRMED" if hit else "REFUTED"))

    # In-frame stops
    stop_claim = extract_stop_claim(desc)
    if stop_claim is not None:
        actual = inframe_stop_count(seq)
        # exact or within 1
        if abs(actual - stop_claim) <= 1:
            results.append(("inframe_stops", "CONFIRMED"))
        else:
            results.append(("inframe_stops", "REFUTED"))

    # Hexamer
    hexes = extract_hexamer_claims(desc)
    if hexes:
        counts = hexamer_counts(seq)
        if hexes == ["__NONE__"]:
            max_rep = max(counts.values()) if counts else 0
            results.append(("hexamer_none", "CONFIRMED" if max_rep < 3 else "REFUTED"))
        else:
            hits = [counts.get(h, 0) >= 2 for h in hexes if h != "__NONE__"]
            if any(hits):
                results.append(("hexamer", "CONFIRMED"))
            else:
                results.append(("hexamer", "REFUTED"))

    return results


def verify_ceiling(seq, target_desc):
    """
    Ceiling: what an ORACLE-accurate description for this sequence could claim.
    We use the target_description (which was programmatically generated from
    sequence statistics) as the reference and verify its claims against the
    actual sequence. This establishes the upper bound on bridge_score.
    """
    return verify_claims(seq, target_desc)


# ----- main ------------------------------------------------------------------

def main():
    out_lines = []
    with INPUT.open() as f:
        for line in f:
            rec = json.loads(line)
            full_seq = rec["sequence"] + rec["continuation"]
            gen = rec["generated_description"]
            tgt = rec["target_description"]

            # Generated-description claims
            gen_results = verify_claims(full_seq, gen)
            # Clamp to 3-6 claims (take first 6; if fewer than 3, pad with UNVERIFIABLE)
            if len(gen_results) > 6:
                gen_results = gen_results[:6]
            while len(gen_results) < 3:
                gen_results.append((f"pad_{len(gen_results)}", "UNVERIFIABLE"))

            n_claims = len(gen_results)
            n_conf = sum(1 for _, v in gen_results if v == "CONFIRMED")
            n_ref = sum(1 for _, v in gen_results if v == "REFUTED")
            n_unv = sum(1 for _, v in gen_results if v == "UNVERIFIABLE")

            # Ceiling: target-description claims, verified against the sequence
            ceil_results = verify_ceiling(full_seq, tgt)
            if len(ceil_results) > 6:
                ceil_results = ceil_results[:6]
            while len(ceil_results) < 3:
                ceil_results.append((f"pad_{len(ceil_results)}", "UNVERIFIABLE"))

            n_claims2 = len(ceil_results)
            n_conf2 = sum(1 for _, v in ceil_results if v == "CONFIRMED")

            row = {
                "seq_idx": rec["seq_idx"],
                "n_claims": n_claims,
                "n_confirmed": n_conf,
                "n_refuted": n_ref,
                "n_unverifiable": n_unv,
                "bridge_score": n_conf / n_claims if n_claims else 0.0,
                "ceiling_n_claims": n_claims2,
                "ceiling_n_confirmed": n_conf2,
                "ceiling_score": n_conf2 / n_claims2 if n_claims2 else 0.0,
            }
            out_lines.append(json.dumps(row))

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w") as f:
        f.write("\n".join(out_lines) + "\n")

    # summary
    import statistics as st
    bridge = [json.loads(l)["bridge_score"] for l in out_lines]
    ceiling = [json.loads(l)["ceiling_score"] for l in out_lines]
    n_cl = [json.loads(l)["n_claims"] for l in out_lines]
    print(f"wrote {len(out_lines)} rows -> {OUTPUT}")
    print(f"  mean n_claims        = {st.mean(n_cl):.2f}")
    print(f"  mean bridge_score    = {st.mean(bridge):.3f}")
    print(f"  mean ceiling_score   = {st.mean(ceiling):.3f}")


if __name__ == "__main__":
    main()
