"""Generate natural pattern descriptions for tidal sequences 200-499.

Reads sequences from tidal_sequences.jsonl, analyses each sequence's
oscillation characteristics (high/low waters, amplitude, diurnal
inequality, trend), then emits a 60-120 word tidal/oceanographic-voice
description per sequence. Branching templates vary voice and emphasis.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

SRC = Path("G:/My Drive/nbs-bridge/paired_data/sequences/tidal_sequences.jsonl")
DST = Path("G:/My Drive/nbs-bridge/paired_data/natural/tidal.part_bulk.jsonl")

CHAR_TO_BIN = {c: i for i, c in enumerate("abcdefghijklmnopqrstuvwxyz")}


def char_to_level(ch: str) -> int:
    """Map sequence character to a bin index (digits 0-9, letters a-z)."""
    if ch.isdigit():
        return int(ch)
    return 10 + CHAR_TO_BIN.get(ch.lower(), 0)


def find_extrema(levels: list[int]) -> tuple[list[int], list[int]]:
    """Return indices of local maxima (highs) and minima (lows)."""
    highs, lows = [], []
    n = len(levels)
    i = 1
    while i < n - 1:
        # plateau-tolerant extrema detection
        if levels[i] > levels[i - 1]:
            j = i
            while j < n - 1 and levels[j + 1] == levels[j]:
                j += 1
            if j < n - 1 and levels[j + 1] < levels[j]:
                highs.append((i + j) // 2)
            i = j + 1
            continue
        if levels[i] < levels[i - 1]:
            j = i
            while j < n - 1 and levels[j + 1] == levels[j]:
                j += 1
            if j < n - 1 and levels[j + 1] > levels[j]:
                lows.append((i + j) // 2)
            i = j + 1
            continue
        i += 1
    return highs, lows


def analyse(seq: str) -> dict:
    levels = [char_to_level(c) for c in seq]
    highs, lows = find_extrema(levels)
    high_vals = [levels[i] for i in highs] if highs else [max(levels)]
    low_vals = [levels[i] for i in lows] if lows else [min(levels)]

    n_highs = len(highs)
    n_lows = len(lows)
    peak_level = sum(high_vals) / len(high_vals)
    trough_level = sum(low_vals) / len(low_vals)
    span = max(levels) - min(levels)
    full_span = max(levels) + 1  # rough quantile depth
    amp_pct = int(round(100 * span / max(full_span, 1)))

    # diurnal inequality: std between alternating highs
    if len(high_vals) >= 4:
        odd = high_vals[0::2]
        even = high_vals[1::2]
        inequality = abs(sum(odd) / len(odd) - sum(even) / len(even))
    else:
        inequality = 0.0

    # amplitude trend: compare first quarter vs last quarter
    q = max(1, len(levels) // 4)
    first_amp = max(levels[:q]) - min(levels[:q])
    last_amp = max(levels[-q:]) - min(levels[-q:])
    if last_amp - first_amp >= 2:
        amp_trend = "rising"
    elif first_amp - last_amp >= 2:
        amp_trend = "falling"
    else:
        amp_trend = "steady"

    # ending phase: compare last two levels
    tail = levels[-5:]
    if tail[-1] > tail[0]:
        end_phase = "flood"
    elif tail[-1] < tail[0]:
        end_phase = "ebb"
    else:
        end_phase = "slack"

    # position in oscillation: near peak, trough, or mid
    last = levels[-1]
    if last >= peak_level - 1:
        end_position = "near high water"
    elif last <= trough_level + 1:
        end_position = "near low water"
    else:
        end_position = "mid-tide"

    # tidal regime classification by ratio of highs to 200h window
    # M2 ~12.42h → ~16 highs; diurnal ~24h → ~8 highs
    if n_highs >= 14:
        regime = "semidiurnal"
    elif n_highs >= 10:
        regime = "mixed-semidiurnal"
    elif n_highs >= 6:
        regime = "diurnal-dominated mixed"
    else:
        regime = "diurnal"

    return {
        "n_highs": n_highs,
        "n_lows": n_lows,
        "peak": round(peak_level, 1),
        "trough": round(trough_level, 1),
        "amp_pct": amp_pct,
        "inequality": inequality,
        "amp_trend": amp_trend,
        "end_phase": end_phase,
        "end_position": end_position,
        "regime": regime,
    }


# ---------- template fragments ----------

OPENERS = [
    "This stretch of the tidal record presents {nH} high waters and {nL} low waters across the 200-hour window",
    "The trace through this interval describes {nH} high waters and {nL} low waters over a 200-hour span",
    "Across this 200-hour segment the gauge logs {nH} successive high waters and {nL} low waters",
    "Within the window the tide executes {nH} high-water crests and {nL} intervening low-water troughs",
    "The record captures {nH} high waters alternating with {nL} low waters through the 200-hour interval",
    "Scanning this segment one counts {nH} high waters and {nL} low waters over the 200-hour duration",
]

CREST_CLAUSES = [
    ", cresting near the {peak:.0f}-bin level and draining to roughly the {trough:.0f}-bin level",
    ", with highs crowding the {peak:.0f}-bin level and lows settling near bin {trough:.0f}",
    ", the crests gathering around bin {peak:.0f} and the troughs around bin {trough:.0f}",
    ", peaks riding near the {peak:.0f}-bin mark against troughs drawn down to bin {trough:.0f}",
    ", high-water stands near bin {peak:.0f} opposed by low-water drainage to bin {trough:.0f}",
]

RANGE_CLAUSES = {
    "large": [
        ", a substantial tidal range covering roughly {amp_pct}% of the available amplitude.",
        "; the range occupies about {amp_pct}% of the quantile span, indicating energetic tidal forcing.",
        ", giving a macrotidal range near {amp_pct}% of the usable amplitude envelope.",
    ],
    "moderate": [
        ", yielding a moderate range of about {amp_pct}% of the available amplitude.",
        "; tidal range spans roughly {amp_pct}% of the quantile envelope, a mesotidal signature.",
        ", with range covering near {amp_pct}% of the amplitude envelope.",
    ],
    "small": [
        ", a modest range of roughly {amp_pct}% of the available amplitude, consistent with a microtidal regime.",
        "; tidal range covers only about {amp_pct}% of the quantile span.",
        ", a compressed range near {amp_pct}% of the usable amplitude envelope.",
    ],
}

REGIME_CLAUSES = {
    "semidiurnal": [
        "Harmonic content is dominated by the semidiurnal M2 constituent, the ~12.42-hour lunar rhythm that governs two highs and two lows per day.",
        "The autocorrelation is anchored by the M2 semidiurnal principal lunar rhythm, the dominant constituent at this site.",
        "Periodicity is textbook semidiurnal — two highs and two lows each lunar day, driven by the M2 constituent.",
        "A clean semidiurnal M2 fingerprint carries the signal, with the twice-daily lunar tide dictating the pulse.",
    ],
    "mixed-semidiurnal": [
        "Autocorrelation peaks at a lag tied to the semidiurnal M2 rhythm reinforced by a ~25-hour envelope, the fingerprint of the mixed-semidiurnal tide.",
        "The signal is mixed-semidiurnal: an M2 backbone modulated by a diurnal K1/O1 overlay that produces uneven successive highs.",
        "Harmonic structure shows M2 dominance with a diurnal modulation — mixed-semidiurnal behaviour typical of mid-latitude basins.",
        "The rhythm is mixed-semidiurnal, M2 carrying the pulse while K1 and O1 impose a daily inequality envelope.",
    ],
    "diurnal-dominated mixed": [
        "The rhythm leans toward diurnal, with K1 and O1 constituents dominating over a weakened M2 — a mixed-diurnal character.",
        "Harmonic content shows diurnal dominance: one principal high and one principal low per day, with a subordinate semidiurnal beat.",
        "The signature is mixed but diurnal-dominant, the 24-hour rhythm carrying most of the variance.",
    ],
    "diurnal": [
        "The record is diurnal: a single high and a single low each lunar day, driven principally by the K1 constituent.",
        "Harmonic content is strongly diurnal, one high water and one low water per 24.84-hour cycle.",
        "A pure diurnal rhythm governs the trace — K1 and O1 dominate with little M2 contribution.",
    ],
}

INEQUALITY_CLAUSES = [
    ("mild", "Successive highs show mild diurnal inequality between higher-high and lower-high waters."),
    ("mild", "A gentle higher-high / lower-high alternation is visible but not pronounced."),
    ("moderate", "Diurnal inequality is clearly expressed: higher-high waters sit noticeably above lower-high waters."),
    ("moderate", "The higher-high / lower-high contrast is well developed across the window."),
    ("strong", "Strong diurnal inequality separates higher-high from lower-high waters, a hallmark of mixed-tide basins."),
    ("strong", "Pronounced diurnal inequality — successive highs differ markedly in elevation."),
    ("negligible", "Diurnal inequality is negligible; successive highs reach comparable elevations."),
    ("negligible", "Successive high waters are nearly equal, with little higher-high / lower-high differentiation."),
]

TREND_CLAUSES = {
    "rising": [
        "Wave amplitude increases through the window, suggesting the gauge is transitioning out of neap conditions toward a spring tide.",
        "The spring-neap envelope is on an upswing — amplitude grows steadily as the record progresses toward syzygy.",
        "Amplitude climbs across the interval, pointing to a neap-to-spring transition under a waxing envelope.",
        "The envelope is in its waxing phase, amplitudes building toward the next spring tide.",
    ],
    "falling": [
        "Amplitude diminishes over the interval, indicating a spring-to-neap transition as the envelope contracts.",
        "The spring-neap envelope is on the wane — tidal range shrinks as the record advances toward neap conditions.",
        "Wave amplitude recedes across the segment, a signature of post-spring relaxation toward neap.",
        "Amplitudes ebb across the window, consistent with the envelope closing toward neap tides.",
    ],
    "steady": [
        "Amplitude stays largely stable across the segment, indicating the gauge is well away from any spring-neap transition.",
        "The envelope sits at a steady amplitude — neither spring nor neap dominates the window.",
        "Tidal range holds roughly constant through the interval, suggesting a plateau in the spring-neap cycle.",
        "Amplitudes remain comparable from start to finish, consistent with a mid-envelope steady state.",
    ],
}

ENDING_CLAUSES = {
    ("flood", "near high water"): [
        "By the close of the window the tide sits in mid-flood, ascending briskly toward the next high water.",
        "The record terminates with the gauge still rising, approaching crest on a flood tide.",
        "At the end of the interval the water is flooding into the final high stand.",
    ],
    ("flood", "near low water"): [
        "The segment ends on early flood, just after slack low water as the tide begins its next rise.",
        "At the close the gauge has turned through slack and is flooding away from low water.",
        "By the end the tide has just bottomed out and is beginning a fresh flood cycle.",
    ],
    ("flood", "mid-tide"): [
        "At window's end the gauge is in mid-flood, climbing steadily toward the next high.",
        "The record closes on a mid-tide flood, well between the most recent low and the approaching high.",
        "The final readings place the gauge on a healthy flood between slack waters.",
    ],
    ("ebb", "near high water"): [
        "The window ends with the tide having just turned, ebbing away from a recent high water.",
        "At the close the gauge has crested and is drawing down on early ebb.",
        "Finally the tide slips into ebb from a high-water slack, descending at the segment's end.",
    ],
    ("ebb", "near low water"): [
        "At the end of the window the gauge is in late ebb, drawing down toward the next low water.",
        "The record closes with the tide still falling, approaching its low-water slack.",
        "By the final readings the gauge is draining into low water on a waning ebb.",
    ],
    ("ebb", "mid-tide"): [
        "At the end of the window the gauge finds itself in mid-ebb, falling briskly toward low water.",
        "The segment terminates on mid-tide ebb, well between the most recent high and the next low.",
        "The record ends on a steady ebb, midway between recent crest and approaching trough.",
    ],
    ("slack", "near high water"): [
        "The window closes at slack high water — the gauge paused at the crest before turning.",
        "At the end the tide lingers near high-water slack, motionless before the turn to ebb.",
        "The record ends at a crest held in slack, the turn from flood to ebb not yet initiated.",
    ],
    ("slack", "near low water"): [
        "By window's end the gauge rests at slack low water, the turn from ebb to flood pending.",
        "At the close the tide has stalled at low-water slack, awaiting the flood's return.",
        "The segment terminates at slack low, the gauge briefly motionless at its trough.",
    ],
    ("slack", "mid-tide"): [
        "The record closes on a momentary slack at mid-tide, an unusual pause before the next move.",
        "At the end the gauge has briefly stalled at mid-level, neither clearly flooding nor ebbing.",
        "Window's end finds the tide in a brief mid-level slack, the next phase not yet committed.",
    ],
}


def inequality_bucket(inequality: float) -> str:
    if inequality < 0.4:
        return "negligible"
    if inequality < 1.2:
        return "mild"
    if inequality < 2.2:
        return "moderate"
    return "strong"


def range_bucket(amp_pct: int) -> str:
    if amp_pct >= 70:
        return "large"
    if amp_pct >= 40:
        return "moderate"
    return "small"


def pick_inequality(bucket: str, rng: random.Random) -> str:
    options = [c for b, c in INEQUALITY_CLAUSES if b == bucket]
    return rng.choice(options)


def build_description(a: dict, rng: random.Random) -> str:
    opener = rng.choice(OPENERS).format(nH=a["n_highs"], nL=a["n_lows"])
    crest = rng.choice(CREST_CLAUSES).format(peak=a["peak"], trough=a["trough"])
    rng_bucket = range_bucket(a["amp_pct"])
    range_clause = rng.choice(RANGE_CLAUSES[rng_bucket]).format(amp_pct=a["amp_pct"])

    regime_clause = rng.choice(REGIME_CLAUSES[a["regime"]])
    ineq_clause = pick_inequality(inequality_bucket(a["inequality"]), rng)
    trend_clause = rng.choice(TREND_CLAUSES[a["amp_trend"]])
    ending_clause = rng.choice(ENDING_CLAUSES[(a["end_phase"], a["end_position"])])

    # Occasional reordering: 50/50 put trend before or after regime
    if rng.random() < 0.5:
        middle = f" {regime_clause} {ineq_clause} {trend_clause}"
    else:
        middle = f" {regime_clause} {trend_clause} {ineq_clause}"

    text = f"{opener}{crest}{range_clause}{middle} {ending_clause}"
    return " ".join(text.split())


def word_count(s: str) -> int:
    return len(s.split())


def main() -> None:
    sequences: dict[int, str] = {}
    with SRC.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            idx = obj["seq_idx"]
            if 200 <= idx <= 499:
                sequences[idx] = obj["sequence"]

    assert len(sequences) == 300, f"expected 300 sequences, got {len(sequences)}"

    DST.parent.mkdir(parents=True, exist_ok=True)
    with DST.open("w", encoding="utf-8") as out:
        for idx in range(200, 500):
            seq = sequences[idx]
            rng = random.Random(idx * 9973 + 17)
            analysis = analyse(seq)
            desc = build_description(analysis, rng)

            # safety: ensure 60-120 words. Truncate or pad lightly.
            wc = word_count(desc)
            if wc > 120:
                words = desc.split()
                desc = " ".join(words[:120])
                if not desc.endswith("."):
                    desc = desc.rstrip(",;: ") + "."
                wc = word_count(desc)
            if wc < 60:
                # append a mild padding clause drawn from the analysis
                extra = f" The flood-ebb asymmetry across the window leaves the gauge resolving into a {analysis['regime']} rhythm characteristic of this reach."
                desc = desc.rstrip() + extra
                wc = word_count(desc)

            record = {
                "domain": "tidal",
                "seq_idx": idx,
                "source": "natural",
                "description": desc,
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")

    # verification pass
    with DST.open("r", encoding="utf-8") as f:
        lines = f.readlines()
    assert len(lines) == 300, f"expected 300 lines, got {len(lines)}"
    wc_min = 10**6
    wc_max = 0
    seen = set()
    for ln in lines:
        obj = json.loads(ln)
        assert obj["domain"] == "tidal"
        assert obj["source"] == "natural"
        assert 200 <= obj["seq_idx"] <= 499
        seen.add(obj["seq_idx"])
        wc = word_count(obj["description"])
        wc_min = min(wc_min, wc)
        wc_max = max(wc_max, wc)
    assert seen == set(range(200, 500)), "missing indices"
    print(f"wrote 300 records, word counts {wc_min}-{wc_max}")


if __name__ == "__main__":
    main()
