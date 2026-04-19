"""Claim-verification eval for MLP bridge on domain=whale, source=natural.

Reads whale_natural_eval_prepared.jsonl, extracts factual claims from
generated_description, verifies them against sequence+continuation character
data, and writes whale_natural_scores.jsonl. Also computes a ceiling from
independently-generated claims.
"""

import json
import re
from collections import Counter
from pathlib import Path

IN_PATH = Path("G:/My Drive/nbs-bridge/results/mlp/whale_natural_eval_prepared.jsonl")
OUT_PATH = Path("G:/My Drive/nbs-bridge/results/mlp/whale_natural_scores.jsonl")

# Whale letter bins: A..T => 19 quantile bins from sub-bass to whistle.
# Interpret descriptor terms in the generated text:
#   "sub-bass"        -> letter in {A,B}
#   "low" / "low moan"-> letter in {A,B,C,D}
#   "mid-low"         -> letter in {C,D,E,F}
#   "mid"             -> letter in {F,G,H,I,J,K}
#   "mid-high"        -> letter in {K,L,M,N}
#   "high" / "whistle"-> letter >= L
# These ranges overlap intentionally — the descriptions use soft bands.
BAND_SETS = {
    "sub-bass":  set("AB"),
    "low moan":  set("ABCD"),
    "low":       set("ABCD"),
    "mid-low":   set("CDEF"),
    "mid":       set("FGHIJK"),
    "mid-high":  set("KLMN"),
    "high":      set("LMNOPQRSTU"),
    "whistle":   set("LMNOPQRSTU"),
    "whistle-band": set("LMNOPQRSTU"),
}


def letter_idx(c: str) -> int:
    return ord(c) - ord("A")


def band_of(c: str) -> str:
    i = letter_idx(c)
    if i <= 1:
        return "sub-bass"
    if i <= 3:
        return "low moan"
    if i <= 5:
        return "mid-low"
    if i <= 10:
        return "mid"
    if i <= 13:
        return "mid-high"
    return "high"


def split_sentences(text: str):
    # Split on sentence-ish punctuation; also split on semicolons.
    parts = re.split(r"(?<=[.!?])\s+|;\s*", text.strip())
    return [p.strip() for p in parts if p.strip()]


# ---------------------- claim verification primitives ----------------------

def dominant_letter(seq: str):
    c = Counter(seq)
    return c.most_common(1)[0]  # (letter, count)


def dominant_band(seq: str):
    bc = Counter(band_of(ch) for ch in seq)
    return bc.most_common(1)[0]


def longest_run(seq: str, letters: set):
    best = cur = 0
    best_start = cur_start = 0
    start = 0
    for i, ch in enumerate(seq):
        if ch in letters:
            if cur == 0:
                cur_start = i
            cur += 1
            if cur > best:
                best = cur
                best_start = cur_start
        else:
            cur = 0
    return best, best_start


def plateau_info(seq: str, letters: set):
    """Return (longest_plateau_len, position_label) where position_label is
    opening/first-third/centre/middle/end/second-half based on where the
    longest run starts."""
    run_len, run_start = longest_run(seq, letters)
    n = len(seq)
    mid = run_start + run_len / 2
    frac = mid / max(n, 1)
    if frac < 0.25:
        pos = "opening"
    elif frac < 0.42:
        pos = "first-third"
    elif frac < 0.58:
        pos = "centre"
    elif frac < 0.75:
        pos = "second-half"
    else:
        pos = "end"
    return run_len, pos, run_start


def span_range(seq: str):
    idxs = [letter_idx(c) for c in seq]
    return min(idxs), max(idxs)


# ---------------------- claim extraction ----------------------

# Phrases we recognize as factual claims. Each is a regex with a handler.

def extract_claims(text: str):
    """Return list of (claim_text, verifier_fn) tuples.
    verifier_fn takes (full_seq_str) and returns a verdict string.
    """
    claims = []
    lowered = text.lower()
    sentences = split_sentences(text)

    # 1. "carries the theme for more than half of the phrase" / "more than half"
    for band_name, band_set in BAND_SETS.items():
        pat = rf"{re.escape(band_name)}[^.]*?(carries the theme for more than half|more than half of the phrase|dominant tonal centre|recurs as the dominant)"
        if re.search(pat, lowered):
            claims.append(
                (
                    f"{band_name} is dominant band (>= ~40% of chars)",
                    _check_band_dominant(band_set, 0.40),
                )
            )
            break  # only take one dominant-band claim

    # 2. plateau position claim
    #   "the plateau is set at the opening"
    #   "the plateau is centred" / "centered, occupying the middle"
    #   "sits in the first third" / "in the position of a leading theme"
    #   "the plateau sits at the end" / "second half"
    plateau_band = None
    for band_name, band_set in BAND_SETS.items():
        if re.search(rf"plateau on the {re.escape(band_name)}", lowered):
            plateau_band = (band_name, band_set)
            break
    if plateau_band is not None:
        band_name, band_set = plateau_band
        pos_claim = None
        if "set at the opening" in lowered or "at the opening" in lowered:
            pos_claim = "opening"
        elif "centred" in lowered or "centered" in lowered or "occupying the middle" in lowered or "middle of the phrase" in lowered:
            pos_claim = "centre"
        elif "first third" in lowered or "leading theme" in lowered:
            pos_claim = "first-third"
        elif "second half" in lowered or "sits in the second" in lowered:
            pos_claim = "second-half"
        elif "at the end" in lowered or "closing" in lowered:
            pos_claim = "end"
        if pos_claim is not None:
            claims.append(
                (
                    f"plateau on {band_name} is positioned at {pos_claim}",
                    _check_plateau_position(band_set, pos_claim),
                )
            )

    # 3. plateau length claim: "long moan-plateau" or "dominant moan-plateau" or "brief"
    if plateau_band is not None:
        band_name, band_set = plateau_band
        if "long moan-plateau" in lowered or "dominant moan-plateau" in lowered or "held nearly without interruption" in lowered:
            claims.append(
                (
                    f"plateau on {band_name} is long (run >= 30 chars)",
                    _check_plateau_length(band_set, 30),
                )
            )
        elif "brief" in lowered and "plateau" in lowered:
            claims.append(
                (
                    f"plateau on {band_name} is brief (run < 15 chars)",
                    _check_plateau_length_max(band_set, 15),
                )
            )

    # 4. spectral span claims
    #   "wide spectral span" / "narrow spectral span" / "moderate spectral span"
    #   "reaching from X up into Y"
    span_match = re.search(r"(wide|narrow|moderate) spectral span", lowered)
    if span_match:
        width = span_match.group(1)
        claims.append(
            (
                f"spectral span is {width}",
                _check_span_width(width),
            )
        )

    # 5. span endpoints: "from sub-bass to mid-low" or "from sub-bass up into the whistle-band"
    endpts = re.search(r"from ([a-z\- ]+?) (?:to|up into) the ([a-z\- ]+?)(?:\.|,|$)", lowered)
    if endpts:
        low_term = endpts.group(1).strip()
        high_term = endpts.group(2).strip()
        low_set = BAND_SETS.get(low_term)
        high_set = BAND_SETS.get(high_term)
        if low_set and high_set:
            claims.append(
                (
                    f"span reaches from {low_term} to {high_term}",
                    _check_span_endpoints(low_set, high_set),
                )
            )

    # 6. contour shape: "undulating", "arching (rises and falls)", "rising", "falling", "flat"
    if "gently undulating" in lowered or "undulating contour" in lowered:
        claims.append(("contour is undulating (many direction changes)", _check_contour_undulating()))
    elif "arching contour" in lowered or ("rises and then falls" in lowered):
        claims.append(("contour arches (rise then fall)", _check_contour_arch()))
    elif "rising contour" in lowered or "steadily rising" in lowered:
        claims.append(("contour is rising", _check_contour_rising()))
    elif "falling contour" in lowered or "steadily falling" in lowered:
        claims.append(("contour is falling", _check_contour_falling()))
    elif "flat contour" in lowered or "mostly flat" in lowered:
        claims.append(("contour is flat", _check_contour_flat()))

    # 7. cadence / unit count: "several discrete units" / "many" / "few"
    if "several discrete units" in lowered or "moderate cadence" in lowered:
        claims.append(("moderate number of distinct runs (5-40)", _check_cadence(5, 40)))
    elif "many discrete units" in lowered or "rapid cadence" in lowered:
        claims.append(("many distinct runs (>= 20)", _check_cadence(20, 10**6)))
    elif "few discrete units" in lowered or "slow cadence" in lowered:
        claims.append(("few distinct runs (<= 6)", _check_cadence(0, 6)))

    # 8. transitions: "alternate between upward and downward steps across neighbouring bands"
    if "alternate between upward and downward" in lowered or "alternating" in lowered:
        claims.append(("transitions alternate direction", _check_alternating()))
    if "neighbouring bands" in lowered or "neighboring bands" in lowered:
        claims.append(("steps are mostly to neighbouring bands (|delta|<=2)", _check_neighbour_steps()))

    return claims


# ---------------------- verifier closures ----------------------

def _check_band_dominant(band_set, thresh):
    def f(seq):
        frac = sum(1 for c in seq if c in band_set) / len(seq)
        if frac >= thresh:
            return "CONFIRMED"
        if frac >= thresh - 0.10:
            return "UNVERIFIABLE"  # borderline
        return "REFUTED"
    return f


def _check_plateau_position(band_set, expected_pos):
    def f(seq):
        run_len, pos, _ = plateau_info(seq, band_set)
        if run_len < 5:
            return "REFUTED"  # no meaningful plateau on that band
        if pos == expected_pos:
            return "CONFIRMED"
        # allow adjacency for partial credit = UNVERIFIABLE
        order = ["opening", "first-third", "centre", "second-half", "end"]
        if expected_pos in order and pos in order:
            if abs(order.index(expected_pos) - order.index(pos)) <= 1:
                return "UNVERIFIABLE"
        return "REFUTED"
    return f


def _check_plateau_length(band_set, min_len):
    def f(seq):
        run_len, _ = longest_run(seq, band_set)
        return "CONFIRMED" if run_len >= min_len else "REFUTED"
    return f


def _check_plateau_length_max(band_set, max_len):
    def f(seq):
        run_len, _ = longest_run(seq, band_set)
        return "CONFIRMED" if run_len < max_len else "REFUTED"
    return f


def _check_span_width(width):
    def f(seq):
        lo, hi = span_range(seq)
        span = hi - lo
        if width == "wide":
            if span >= 10:
                return "CONFIRMED"
            return "REFUTED"
        if width == "narrow":
            if span <= 5:
                return "CONFIRMED"
            return "REFUTED"
        # moderate
        if 4 <= span <= 11:
            return "CONFIRMED"
        return "REFUTED"
    return f


def _check_span_endpoints(low_set, high_set):
    def f(seq):
        # sequence must contain at least one char in low_set and in high_set
        has_low = any(c in low_set for c in seq)
        has_high = any(c in high_set for c in seq)
        if has_low and has_high:
            return "CONFIRMED"
        if has_low or has_high:
            return "UNVERIFIABLE"
        return "REFUTED"
    return f


def _direction_changes(seq):
    idxs = [letter_idx(c) for c in seq]
    changes = 0
    last_sign = 0
    for a, b in zip(idxs, idxs[1:]):
        d = b - a
        sign = (d > 0) - (d < 0)
        if sign != 0 and last_sign != 0 and sign != last_sign:
            changes += 1
        if sign != 0:
            last_sign = sign
    return changes


def _check_contour_undulating():
    def f(seq):
        n = len(seq)
        changes = _direction_changes(seq)
        # undulating = many direction changes relative to length
        if changes >= max(4, n // 40):
            return "CONFIRMED"
        if changes >= 2:
            return "UNVERIFIABLE"
        return "REFUTED"
    return f


def _check_contour_arch():
    def f(seq):
        # max should be roughly in the middle third
        idxs = [letter_idx(c) for c in seq]
        peak = idxs.index(max(idxs))
        n = len(idxs)
        # also: first half rises on average and second half falls
        half = n // 2
        first_mean = sum(idxs[:half]) / max(half, 1)
        second_mean = sum(idxs[half:]) / max(n - half, 1)
        peak_central = 0.20 < peak / n < 0.80
        # arch means peak in middle and broadly first rises / second falls is not required — just peak central
        if peak_central:
            return "CONFIRMED"
        return "REFUTED"
    return f


def _check_contour_rising():
    def f(seq):
        idxs = [letter_idx(c) for c in seq]
        half = len(idxs) // 2
        first = sum(idxs[:half]) / max(half, 1)
        second = sum(idxs[half:]) / max(len(idxs) - half, 1)
        if second - first >= 1.5:
            return "CONFIRMED"
        if second - first >= 0.5:
            return "UNVERIFIABLE"
        return "REFUTED"
    return f


def _check_contour_falling():
    def f(seq):
        idxs = [letter_idx(c) for c in seq]
        half = len(idxs) // 2
        first = sum(idxs[:half]) / max(half, 1)
        second = sum(idxs[half:]) / max(len(idxs) - half, 1)
        if first - second >= 1.5:
            return "CONFIRMED"
        if first - second >= 0.5:
            return "UNVERIFIABLE"
        return "REFUTED"
    return f


def _check_contour_flat():
    def f(seq):
        idxs = [letter_idx(c) for c in seq]
        mean = sum(idxs) / len(idxs)
        var = sum((x - mean) ** 2 for x in idxs) / len(idxs)
        return "CONFIRMED" if var <= 2.0 else "REFUTED"
    return f


def _count_runs(seq):
    runs = 1
    for a, b in zip(seq, seq[1:]):
        if a != b:
            runs += 1
    return runs


def _count_distinct_runs(seq):
    # distinct maximal runs
    return _count_runs(seq)


def _check_cadence(lo, hi):
    def f(seq):
        r = _count_distinct_runs(seq)
        return "CONFIRMED" if lo <= r <= hi else "REFUTED"
    return f


def _check_alternating():
    def f(seq):
        idxs = [letter_idx(c) for c in seq]
        signs = []
        for a, b in zip(idxs, idxs[1:]):
            d = b - a
            if d > 0:
                signs.append(1)
            elif d < 0:
                signs.append(-1)
        if len(signs) < 2:
            return "UNVERIFIABLE"
        alt = sum(1 for a, b in zip(signs, signs[1:]) if a != b)
        ratio = alt / (len(signs) - 1)
        if ratio >= 0.40:
            return "CONFIRMED"
        if ratio >= 0.20:
            return "UNVERIFIABLE"
        return "REFUTED"
    return f


def _check_neighbour_steps():
    def f(seq):
        idxs = [letter_idx(c) for c in seq]
        diffs = [abs(b - a) for a, b in zip(idxs, idxs[1:]) if b != a]
        if not diffs:
            return "UNVERIFIABLE"
        neigh = sum(1 for d in diffs if d <= 2) / len(diffs)
        if neigh >= 0.70:
            return "CONFIRMED"
        if neigh >= 0.40:
            return "UNVERIFIABLE"
        return "REFUTED"
    return f


# ---------------------- ceiling claims ----------------------

def ceiling_claims(seq: str):
    """Independently generate 3-6 factual claims from the data, then verify.
    Because they're data-derived, they should all CONFIRM (modulo threshold
    edge cases) — giving a ceiling."""
    claims = []
    n = len(seq)
    # C1: dominant letter claim
    dl, dc = dominant_letter(seq)
    frac = dc / n
    claims.append(
        (
            f"dominant letter is {dl} with frequency ~{frac:.2f}",
            _check_dominant_letter(dl, frac),
        )
    )
    # C2: dominant band claim
    db, dbc = dominant_band(seq)
    bfrac = dbc / n
    claims.append(
        (
            f"dominant band is {db} (~{bfrac:.2f})",
            _check_band_dominant_exact(BAND_SETS.get(db, set()), bfrac),
        )
    )
    # C3: span
    lo, hi = span_range(seq)
    claims.append(
        (
            f"span covers letters {chr(lo+65)}..{chr(hi+65)}",
            _check_exact_span(lo, hi),
        )
    )
    # C4: longest run of dominant letter
    rl, _ = longest_run(seq, {dl})
    claims.append(
        (
            f"longest run of {dl} is {rl}",
            _check_exact_longest_run(dl, rl),
        )
    )
    # C5: number of distinct runs
    runs = _count_runs(seq)
    claims.append(
        (
            f"there are {runs} distinct maximal runs",
            _check_exact_runs(runs),
        )
    )
    # C6: presence of high-band characters (>= L)
    has_high = any(letter_idx(c) >= 11 for c in seq)
    claims.append(
        (
            f"{'contains' if has_high else 'lacks'} whistle-band characters (>= L)",
            _check_high_presence(has_high),
        )
    )
    return claims


def _check_dominant_letter(expected_letter, expected_frac):
    def f(seq):
        dl, dc = dominant_letter(seq)
        frac = dc / len(seq)
        if dl == expected_letter and abs(frac - expected_frac) < 0.05:
            return "CONFIRMED"
        return "REFUTED"
    return f


def _check_band_dominant_exact(band_set, expected_frac):
    def f(seq):
        if not band_set:
            return "UNVERIFIABLE"
        frac = sum(1 for c in seq if c in band_set) / len(seq)
        return "CONFIRMED" if abs(frac - expected_frac) < 0.05 else "REFUTED"
    return f


def _check_exact_span(expected_lo, expected_hi):
    def f(seq):
        lo, hi = span_range(seq)
        return "CONFIRMED" if lo == expected_lo and hi == expected_hi else "REFUTED"
    return f


def _check_exact_longest_run(letter, expected_rl):
    def f(seq):
        rl, _ = longest_run(seq, {letter})
        return "CONFIRMED" if rl == expected_rl else "REFUTED"
    return f


def _check_exact_runs(expected):
    def f(seq):
        return "CONFIRMED" if _count_runs(seq) == expected else "REFUTED"
    return f


def _check_high_presence(expected):
    def f(seq):
        actual = any(letter_idx(c) >= 11 for c in seq)
        return "CONFIRMED" if actual == expected else "REFUTED"
    return f


# ---------------------- main ----------------------

def main():
    out_lines = []
    bridge_scores = []
    ceiling_scores = []

    with IN_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            seq_full = rec["sequence"] + rec["continuation"]
            gen = rec.get("generated_description", "") or ""

            # Bridge claims
            claims = extract_claims(gen)
            # Clamp to 3-6
            if len(claims) > 6:
                claims = claims[:6]
            n_confirmed = n_refuted = n_unverifiable = 0
            for _text, fn in claims:
                v = fn(seq_full)
                if v == "CONFIRMED":
                    n_confirmed += 1
                elif v == "REFUTED":
                    n_refuted += 1
                else:
                    n_unverifiable += 1
            m = len(claims)
            bridge_score = n_confirmed / m if m > 0 else 0.0

            # Ceiling claims
            cc = ceiling_claims(seq_full)
            if len(cc) > 6:
                cc = cc[:6]
            c_conf = 0
            for _text, fn in cc:
                if fn(seq_full) == "CONFIRMED":
                    c_conf += 1
            m2 = len(cc)
            ceiling_score = c_conf / m2 if m2 > 0 else 0.0

            out = {
                "seq_idx": rec["seq_idx"],
                "n_claims": m,
                "n_confirmed": n_confirmed,
                "n_refuted": n_refuted,
                "n_unverifiable": n_unverifiable,
                "bridge_score": bridge_score,
                "ceiling_n_claims": m2,
                "ceiling_n_confirmed": c_conf,
                "ceiling_score": ceiling_score,
            }
            out_lines.append(json.dumps(out))
            bridge_scores.append(bridge_score)
            ceiling_scores.append(ceiling_score)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text("\n".join(out_lines) + "\n", encoding="utf-8")

    mean_bridge = sum(bridge_scores) / len(bridge_scores) if bridge_scores else 0.0
    mean_ceiling = sum(ceiling_scores) / len(ceiling_scores) if ceiling_scores else 0.0
    print(f"OUT: {OUT_PATH}")
    print(f"LINES: {len(out_lines)}")
    print(f"MEAN_BRIDGE_SCORE: {mean_bridge:.4f}")
    print(f"MEAN_CEILING_SCORE: {mean_ceiling:.4f}")


if __name__ == "__main__":
    main()
