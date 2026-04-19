"""
Claim-verification scorer for MLP bridge, domain=english, source=natural.

For each record in english_natural_eval_prepared.jsonl:
  1. Extract 3-6 specific verifiable claims from generated_description.
  2. Verify each against (sequence + continuation) — the raw 250 chars.
  3. Emit bridge_score (all claims) and ceiling_score (3-6 independent claims).

Verifiable claim types (per user spec):
  - register: narrative / dialogue / exposition / descriptive
  - sentence structure: short vs long sentences
  - presence of dialogue markers
  - common words mentioned in description
  - topic
  - plus: literally quoted phrases are specific factual claims

Key scoring principles:
  * A quoted phrase in the generation is a claim that the phrase is in (or
    paraphrases content of) the raw passage. CONFIRMED iff substring match.
  * "dialogue" register claim verified by reporting-verb markers (said, cried).
  * "descriptive" / "exposition" register claim REFUTED if the passage is
    dominated by dialogue; else CONFIRMED.
  * "narrative voice" claim CONFIRMED generically (the corpus IS narrative
    fiction), but only once per description.
  * Common-word / topic-word claims: the description picks out content words;
    we check each against the raw passage tokens.
  * Named-work claims (Pride and Prejudice, Austen, Duchess of Cambridge)
    checked against distinctive character names in the raw passage.
"""

import json
import re
from pathlib import Path


IN_PATH  = Path("G:/My Drive/nbs-bridge/results/mlp/english_natural_eval_prepared.jsonl")
OUT_PATH = Path("G:/My Drive/nbs-bridge/results/mlp/english_natural_scores.jsonl")


# ---------- helpers ---------------------------------------------------------

STOPWORDS = {
    "the","and","a","an","of","in","to","is","was","were","be","been","being",
    "that","this","it","its","with","for","on","at","by","from","as","or","but",
    "not","no","so","if","then","than","into","out","up","down","over","under",
    "i","me","my","you","your","he","him","his","she","her","we","us","our",
    "they","them","their","one","two","would","could","should","can","will",
    "have","has","had","do","did","does","am","are","what","when","where","who",
    "which","how","there","here","about","any","all","some","more","most","very",
    "just","like","said","only","such","too","much","many","also","again",
    "after","before","still","same","both","each","other","another",
}


def normalise(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9' ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokens(text: str) -> list[str]:
    return [t for t in normalise(text).split() if t]


def contains_phrase(haystack_norm: str, phrase: str, min_words: int = 3) -> bool:
    p = normalise(phrase)
    if len(p.split()) < min_words:
        return False
    return p in haystack_norm


def dialogue_marker_count(raw: str) -> int:
    low = raw.lower()
    markers = [" said ", " cried ", " shouted ", " replied ", " asked ",
               " answered ", " whispered ", " added ", " exclaimed ",
               " remarked ", " inquired "]
    return sum(low.count(m) for m in markers)


def has_dialogue_markers(raw: str) -> bool:
    return dialogue_marker_count(raw) > 0


# ---------- claim extraction ------------------------------------------------

# Register labels the generator uses
REGISTER_LABELS = {
    "narrative":   r"\bnarrative\s+(voice|cadence)\b|\bnarrative\b",
    "descriptive": r"\bdescriptive\s+passage\b|\bdescriptive\b",
    "measured":    r"\bmeasured\s+narrative\b|\bmeasured\b",
    "reflective":  r"\breflective\s+exposition\b|\breflective\b",
    "exposition":  r"\bexposition\b|\bexpository\b",
    "scene":       r"\bscene\b",
    "fragment":    r"\bfragment\b",
    "dialogue":    r"\bdialogue\b|\bconversation\b",
}

NAMED_WORKS = [
    ("pride and prejudice", "pride and prejudice"),
    ("princess of the forest", "princess of the forest"),
    ("duchess of cambridge", "duchess of cambridge"),
    ("austenian", "austen"),
    ("austen's", "austen"),
    ("austen", "austen"),
    ("laurie", "laurie"),
    ("little women", "little women"),
]


def extract_quoted_phrases(desc: str) -> list[str]:
    """Pull all quoted substrings. Handles:
      - "..." (straight double quotes)
      - '...' (straight single quotes)
      - unicode curly quotes
      - UNCLOSED opening quote at end-of-string (common with truncated LLM
        generations): everything after the unmatched opening quote counts.
    """
    out = []
    # Closed double quotes
    out += re.findall(r'"([^"]{2,})"', desc)
    out += re.findall(r"\u201c([^\u201d]{2,})\u201d", desc)
    # Closed single quotes (conservative: require >=3 chars and space
    # boundaries to avoid matching apostrophes inside contractions).
    out += re.findall(r"(?:^|\s)'([^']{3,})'(?:\s|$|[.,;:!?])", desc)

    # UNCLOSED straight double quote: if count is odd, grab tail after last one
    if desc.count('"') % 2 == 1:
        last = desc.rfind('"')
        tail = desc[last+1:].strip()
        if len(tail.split()) >= 3:
            out.append(tail)

    return out


def extract_common_words(desc: str) -> list[str]:
    """Content words the description uses as if they were from the passage.
    Excludes stopwords and meta-critical jargon. Only content-bearing words
    that could plausibly be in a narrative passage.
    """
    meta_jargon = {
        "register","syntax","diction","stylistic","narrative","descriptive",
        "exposition","expository","passage","sentence","clause","continuation",
        "marker","markers","reader","cadence","voice","measured","reflective",
        "fragment","scene","fiction","line","lines","paragraph","paragraphs",
        "prose","discourse","pause","followed","opens","unfolds","begins",
        "noticeable","turn","phrase","device","colours","texture","overwhelming",
        "rich","complex","intimate","long","short","parallel","parallels",
        "compound","comma","colon","semi","period","lexicon","tone","latinate",
        "archaic","archaism","anglo","saxon","em","dash","parenthetical",
        "chiastic","rhetorical","single","double","spectrum","analyse","analysis",
        "code","ending","word","words","likely","probably","plausibly",
        "world","since","lauda","hyperbolic","stylistic","laudable","ostraon",
        "ostra","lexaic","gesnes","verb","self","paragraph","ita","ita",
        "addressing","characterized","commentary","observations","situation",
        "appears","appears","apparent","presence","structure","form",
        "formal","formal","work","works","speaker","written","reader",
        "contained","contains","foregrounds","foreground","follows","following",
        "quoted","quote","directly","indirectly","indirect","direct","character",
        "characters","tradition","british","british","english","sentence",
        "subordinate","subordinate","incorporate","incorporates",
    }
    out = []
    for w in tokens(desc):
        if w in STOPWORDS: continue
        if w in meta_jargon: continue
        if len(w) <= 2: continue
        if w.isdigit(): continue
        out.append(w)
    # dedupe preserving order
    seen = set(); uniq = []
    for w in out:
        if w not in seen:
            seen.add(w); uniq.append(w)
    return uniq


def extract_claims(generated: str, raw: str) -> list[dict]:
    claims = []
    desc_low = generated.lower()

    # --- 1. Register claims -------------------------------------------------
    registers_seen = set()
    for label, pat in REGISTER_LABELS.items():
        if re.search(pat, desc_low):
            if label not in registers_seen:
                claims.append({
                    "type": "register",
                    "text": f"passage register is '{label}'",
                    "label": label,
                })
                registers_seen.add(label)

    # --- 2. Named works / authors ------------------------------------------
    seen_ne = set()
    for surface, canon in NAMED_WORKS:
        if surface in desc_low and canon not in seen_ne:
            claims.append({
                "type": "named_entity",
                "text": f"passage is from/about '{canon}'",
                "label": canon,
            })
            seen_ne.add(canon)

    # --- 3. Quoted phrases (each is a factual claim) -----------------------
    quoted = extract_quoted_phrases(generated)
    seen_q = set()
    for q in quoted:
        qn = normalise(q)
        if not qn: continue
        # Require at least 3 content words to be worth claiming.
        if len(qn.split()) < 3: continue
        # Dedup by first 30 chars of normalised form (handles repetitive tails)
        key = qn[:30]
        if key in seen_q: continue
        seen_q.add(key)
        claims.append({
            "type": "quoted_phrase",
            "text": f'description quotes "{q[:60]}..." as present in passage',
            "label": q,
        })

    # --- 4. Common / topic words -------------------------------------------
    common = extract_common_words(generated)
    # Skip words already used in named entities (canon names etc.)
    skip = {w for c in claims if c["type"]=="named_entity"
                for w in tokens(c["label"])}
    # Skip words already used inside quoted phrases (we don't want to
    # double-count e.g. "girl" from "I am a little girl").
    for c in claims:
        if c["type"] == "quoted_phrase":
            for w in tokens(c["label"])[:10]:
                skip.add(w)

    n_word_claims = 0
    for w in common:
        if w in skip: continue
        claims.append({
            "type": "common_word",
            "text": f"content word '{w}' present in passage",
            "label": w,
        })
        n_word_claims += 1
        if n_word_claims >= 5: break

    # --- 5. Dialogue presence claim ----------------------------------------
    if re.search(r"\bdialogue\b|\bconversation\b|\bspoken\b|\bdiscourse\b",
                 desc_low) and "dialogue" not in registers_seen:
        claims.append({
            "type": "dialogue",
            "text": "passage contains dialogue",
            "label": "dialogue",
        })

    # --- 6. Narrator / identity claims from first-person quote -------------
    # If the generator quotes "I am X" / "I was X" / "I have X", it is claiming
    # the passage uses first-person narration AND that the narrator has that
    # identity. These are two distinct verifiable claims.
    for q in quoted:
        qlow = q.lower().strip()
        # First-person narration claim (only add once)
        if re.match(r"^(i am|i was|i have|i had|i will|i'm|my |me )", qlow):
            if not any(c["type"]=="first_person" for c in claims):
                claims.append({
                    "type": "first_person",
                    "text": "passage is in first-person (narrator uses 'I')",
                    "label": "first_person",
                })
            # Identity claim: "I am a little girl" -> identity = "little girl"
            m = re.match(r"^i\s+(?:am|was)\s+(?:a\s+|an\s+|the\s+)?([a-z ]{3,25}?)(?:,|\.|$| and |\s\s)",
                         qlow)
            if m:
                ident = m.group(1).strip()
                # Keep only nominal identity words, strip common trailing junk
                ident = re.sub(r"\s+(and|or|but)$", "", ident).strip()
                if ident and not any(c.get("label")==ident and c["type"]=="narrator_identity" for c in claims):
                    claims.append({
                        "type": "narrator_identity",
                        "text": f"passage narrator self-describes as '{ident}'",
                        "label": ident,
                    })
            break  # one first-person claim is enough

    # --- 7. Repetition / structural claim ----------------------------------
    # Generators that emit "X, and X, and X..." are implicitly claiming the
    # passage has that repetition structure. Verifiable: the raw passage does
    # NOT have such repetition (natural prose varies).
    for q in quoted:
        # Detect repetition: same 3+ word span repeating 3+ times
        qn = normalise(q)
        words = qn.split()
        if len(words) < 12: continue
        # Find a repeated 4-word span
        repeated = False
        for i in range(len(words)-8):
            span = " ".join(words[i:i+4])
            if qn.count(span) >= 3:
                repeated = True
                break
        if repeated:
            if not any(c["type"]=="repetition_structure" for c in claims):
                claims.append({
                    "type": "repetition_structure",
                    "text": "passage has near-verbatim repetition of a 4-word span",
                    "label": "repetition",
                })
            break

    return claims


# ---------- verification ----------------------------------------------------

def verify_claim(claim: dict, raw: str) -> str:
    raw_norm = normalise(raw)
    raw_tok_set = set(tokens(raw))
    low = raw.lower()

    t = claim["type"]
    lbl = claim["label"]

    if t == "register":
        has_dlg = has_dialogue_markers(raw)
        # Check whether the raw passage is dialogue-heavy: count dialogue
        # markers. With 2+ markers in 250 chars it's mixed dialogue; we call
        # a passage "descriptive/exposition" REFUTED in that case and
        # "dialogue" CONFIRMED.
        heavy_dlg = dialogue_marker_count(raw) >= 2
        if lbl == "dialogue":
            return "CONFIRMED" if has_dlg else "REFUTED"
        if lbl in {"descriptive","exposition","reflective"}:
            # These explicitly contrast with dialogue. Refute if passage is
            # dialogue-heavy.
            return "REFUTED" if heavy_dlg else "CONFIRMED"
        if lbl in {"narrative","measured","scene","fragment"}:
            # Generic narrative/scene labels: CONFIRMED (this IS narrative
            # fiction from Austen/Carroll).
            return "CONFIRMED"
        return "UNVERIFIABLE"

    if t == "named_entity":
        is_austen = any(name in low for name in
                        ["elizabeth","darcy","bennet","bingley","collins",
                         "wickham","jane","lydia","longbourn","netherfield",
                         "pemberley","charlotte","hertfordshire","gardiner",
                         "fitzwilliam","catherine","rosings","hurst","caroline"])
        is_carroll = any(name in low for name in
                         ["alice","hatter","queen","rabbit","mock turtle",
                          "gryphon","cheshire","duchess"])
        # If NEITHER Austen nor Carroll names are present, we can't verify the
        # source from character names alone — the passage is narrative prose
        # without proper nouns. Return UNVERIFIABLE in that case.
        if not is_austen and not is_carroll:
            if lbl in ("pride and prejudice","austen","austenian","little women"):
                return "UNVERIFIABLE"
            if lbl in ("duchess of cambridge","princess of the forest"):
                # These are very specific false claims — still refuted.
                return "REFUTED"
            return "UNVERIFIABLE"

        if lbl == "pride and prejudice":
            return "CONFIRMED" if is_austen else "REFUTED"
        if lbl in ("austen", "austenian"):
            return "CONFIRMED" if is_austen else "REFUTED"
        if lbl == "duchess of cambridge":
            return "REFUTED"
        if lbl == "princess of the forest":
            return "REFUTED"
        if lbl == "laurie":
            return "CONFIRMED" if "laurie" in low else "REFUTED"
        if lbl == "little women":
            return "REFUTED"
        return "UNVERIFIABLE"

    if t == "quoted_phrase":
        return "CONFIRMED" if contains_phrase(raw_norm, lbl) else "REFUTED"

    if t == "common_word":
        w = normalise(lbl)
        if not w: return "UNVERIFIABLE"
        return "CONFIRMED" if w in raw_tok_set else "REFUTED"

    if t == "dialogue":
        return "CONFIRMED" if has_dialogue_markers(raw) else "REFUTED"

    if t == "first_person":
        # Is the raw passage first-person? Check for explicit "i " or " i "
        # as a standalone pronoun (the corpus is lowercased, no apostrophes).
        toks = tokens(raw)
        if "i" in toks or " i " in (" " + " ".join(toks) + " "):
            return "CONFIRMED"
        return "REFUTED"

    if t == "narrator_identity":
        # e.g. "little girl", "man", "woman". Check the phrase appears as a
        # substring of the raw passage (the narrator or a character would be
        # so described).
        ident_norm = normalise(lbl)
        if not ident_norm: return "UNVERIFIABLE"
        return "CONFIRMED" if ident_norm in raw_norm else "REFUTED"

    if t == "repetition_structure":
        # Does the raw passage have a 4-word span repeating 3+ times?
        raw_words = raw_norm.split()
        for i in range(len(raw_words)-8):
            span = " ".join(raw_words[i:i+4])
            if raw_norm.count(span) >= 3:
                return "CONFIRMED"
        return "REFUTED"

    return "UNVERIFIABLE"


# ---------- ceiling selection ----------------------------------------------

def pick_ceiling(claims: list[dict]) -> list[dict]:
    """
    Pick 3-6 INDEPENDENT claims. Priority:
      1. Up to 2 quoted_phrase (strongest, literal content).
      2. Up to 2 named_entity (specific work claim).
      3. Up to 1 dialogue claim.
      4. Up to 2 register claims (distinct labels).
      5. Common words fill remaining slots.
    """
    by_type = {"quoted_phrase": [], "named_entity": [], "dialogue": [],
               "register": [], "common_word": [], "first_person": [],
               "narrator_identity": [], "repetition_structure": []}
    for c in claims:
        by_type.setdefault(c["type"], []).append(c)

    ceiling = []
    ceiling += by_type["quoted_phrase"][:1]       # 1 literal content claim
    ceiling += by_type["named_entity"][:2]         # specific work/author
    ceiling += by_type["first_person"][:1]         # narration person
    ceiling += by_type["narrator_identity"][:1]    # character identity
    ceiling += by_type["repetition_structure"][:1] # structural
    ceiling += by_type["dialogue"][:1]
    ceiling += by_type["register"][:1]             # register (one)

    # Fill with common words if below 6
    for c in by_type["common_word"]:
        if len(ceiling) >= 6: break
        ceiling.append(c)

    ceiling = ceiling[:6]

    # Floor at 3: pad from leftovers
    if len(ceiling) < 3:
        ids = {id(c) for c in ceiling}
        for c in claims:
            if id(c) in ids: continue
            ceiling.append(c)
            if len(ceiling) >= 3: break

    return ceiling


# ---------- main ------------------------------------------------------------

def score_record(rec: dict) -> dict:
    raw = (rec["sequence"] or "") + (rec["continuation"] or "")
    claims = extract_claims(rec["generated_description"], raw)

    for c in claims:
        c["verdict"] = verify_claim(c, raw)

    n_claims = len(claims)
    n_c = sum(1 for c in claims if c["verdict"] == "CONFIRMED")
    n_r = sum(1 for c in claims if c["verdict"] == "REFUTED")
    n_u = sum(1 for c in claims if c["verdict"] == "UNVERIFIABLE")

    bridge_score = (n_c / n_claims) if n_claims else 0.0

    ceiling = pick_ceiling(claims)
    cn = len(ceiling)
    cc = sum(1 for c in ceiling if c["verdict"] == "CONFIRMED")
    ceiling_score = (cc / cn) if cn else 0.0

    return {
        "seq_idx": rec["seq_idx"],
        "n_claims": n_claims,
        "n_confirmed": n_c,
        "n_refuted": n_r,
        "n_unverifiable": n_u,
        "bridge_score": round(bridge_score, 4),
        "ceiling_n_claims": cn,
        "ceiling_n_confirmed": cc,
        "ceiling_score": round(ceiling_score, 4),
    }


def main():
    out_lines = []
    with IN_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            rec = json.loads(line)
            score = score_record(rec)
            out_lines.append(json.dumps(score, ensure_ascii=False))

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text("\n".join(out_lines) + "\n", encoding="utf-8")

    scores = [json.loads(l) for l in out_lines]
    n = len(scores)
    mean_bridge  = sum(s["bridge_score"]  for s in scores) / n
    mean_ceiling = sum(s["ceiling_score"] for s in scores) / n
    mean_nclaims = sum(s["n_claims"]       for s in scores) / n
    mean_cclaims = sum(s["ceiling_n_claims"] for s in scores) / n
    print(f"Wrote {n} records to {OUT_PATH}")
    print(f"Mean n_claims:         {mean_nclaims:.2f}")
    print(f"Mean bridge_score:     {mean_bridge:.4f}")
    print(f"Mean ceiling_n_claims: {mean_cclaims:.2f}")
    print(f"Mean ceiling_score:    {mean_ceiling:.4f}")


if __name__ == "__main__":
    main()
