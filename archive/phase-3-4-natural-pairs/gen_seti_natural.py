"""Generate 300 natural pattern descriptions for SETI domain, seq_idx 200-499.

Voice: radio-astronomy observation-log. Noise-dominated background, RFI,
receiver-temperature, baseline drift, narrowband feature, isolated transient,
power excursion, no persistent structure.

No fabricated telescopes/frequencies/targets. Honest about near-noise.
Each description is correlated with the 5-symbol distribution (A..E) of the
sequence via branching templates keyed on dominant symbol, rare-symbol
presence, and run structure.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

INPUT = Path("G:/My Drive/nbs-bridge/paired_data/sequences/seti_sequences.jsonl")
OUTPUT = Path("G:/My Drive/nbs-bridge/paired_data/natural/seti.part_bulk.jsonl")

SYMBOLS = list("ABCDE")

# ----- symbol interpretation (internal only; no fabricated frequencies/targets) -----
# A: deep quiescent bin (below typical noise floor trend)
# B: slightly below-floor fluctuation
# C: nominal noise-floor sample (dominant)
# D: elevated-but-unresolved fluctuation
# E: isolated power excursion candidate (rare, short-lived)

RARE_E_HIGH_WORDS = [
    "a handful of isolated power excursions",
    "a few short-lived excursion candidates",
    "brief, non-repeating power excursions",
    "sparse excursion samples well above the running floor",
]
RARE_E_LOW_WORDS = [
    "essentially no excursion candidates",
    "no clearly resolved excursions above the noise floor",
    "an absence of isolated power excursions",
]
RARE_A_WORDS = [
    "occasional deep quiescent bins",
    "intermittent dips tracking receiver-temperature wander",
    "a few unusually quiet samples consistent with baseline drift",
]

INTROS = [
    "Log entry reads as a noise-dominated stretch",
    "Record behaves as a quiet observing window",
    "Segment reads as routine background integration",
    "Trace is dominated by receiver noise",
    "Observation log shows a fundamentally featureless window",
    "The run looks noise-limited throughout",
    "Stretch reads as an ordinary baseline pass",
    "This block scans as near-noise data",
    "Entry behaves as a quiescent, noise-led window",
    "The record reads as uneventful background",
]

BASELINE_PHRASES = [
    "baseline drift is gentle and monotonic",
    "baseline wanders slowly in a way consistent with receiver-temperature change",
    "the running baseline shows shallow thermal drift",
    "baseline drift is present but within routine tolerance",
    "the floor drifts mildly across the window",
    "baseline wander is slow and shape-consistent with Tsys evolution",
]

NARROWBAND_PHRASES = [
    "a narrowband feature flickers briefly without persistence",
    "a thin candidate line appears in one bin and does not recur",
    "one narrow spike shows up and is not confirmed in adjacent samples",
    "brief narrowband activity fails to persist",
    "a candidate narrowband bin sits right at the detection margin",
    "narrowband activity is at best marginal and not reproducible",
]

RFI_PHRASES = [
    "low-level RFI-like fluctuation is consistent with the local environment",
    "a brief RFI-shaped bump intrudes but does not repeat",
    "fluctuations are consistent with stochastic RFI background",
    "an RFI-consistent transient punches above the floor and vanishes",
    "what little structure there is looks RFI-shaped rather than astrophysical",
]

TRANSIENT_PHRASES = [
    "an isolated transient registers once and is not repeated",
    "a single short excursion appears and does not return",
    "one-off excursion candidate is present but not reproducible",
    "a brief transient sits above the floor for a single sample",
]

NO_STRUCTURE_PHRASES = [
    "nothing in the run reads as persistent structure",
    "no persistent structure is evident across the window",
    "there is no coherent feature tracking across the block",
    "no reproducible structure can be pulled out of the noise",
    "nothing persists beyond the local noise scale",
    "no stable feature survives past the integration window",
]

HONESTY_PHRASES = [
    "all of this sits near the noise floor and should be treated as such",
    "calling any of this a detection would overstate the data",
    "read as near-noise; further integration would likely wash it out",
    "at this sensitivity level the entire block is consistent with background",
    "the run does not rise meaningfully above expected noise statistics",
    "nothing here warrants follow-up on its own",
    "the honest read is: noise, with a few unresolved wiggles",
]

NEXT_STEP_PHRASES = [
    "flag for routine archival; no follow-up triggered",
    "archive as nominal background integration",
    "no action item; log as quiescent window",
    "classify as noise-dominated; no candidate escalation",
    "tag as routine pass; no persistent signal to chase",
    "record as uneventful and move on",
]


def classify(counts: Counter, length: int) -> dict:
    """Extract branching features from the symbol distribution."""
    freq = {s: counts.get(s, 0) / length for s in SYMBOLS}
    dominant = max(SYMBOLS, key=lambda s: freq[s])
    e_rate = freq["E"]
    a_rate = freq["A"]
    b_rate = freq["B"]
    d_rate = freq["D"]
    # elevated vs depressed side
    upper = freq["D"] + freq["E"]
    lower = freq["A"] + freq["B"]
    return {
        "freq": freq,
        "dominant": dominant,
        "e_rate": e_rate,
        "a_rate": a_rate,
        "b_rate": b_rate,
        "d_rate": d_rate,
        "upper": upper,
        "lower": lower,
    }


def describe(seq: str, rng: random.Random) -> str:
    counts = Counter(seq)
    feat = classify(counts, len(seq))
    freq = feat["freq"]

    parts: list[str] = []

    # 1) Intro — dominant-class branch
    intro = rng.choice(INTROS)
    if feat["dominant"] == "C":
        intro_tail = (
            f"with the bulk of samples ({freq['C']*100:.0f}%) clustered at the nominal noise floor"
        )
    elif feat["dominant"] == "D":
        intro_tail = (
            f"with mass sitting slightly above the running floor "
            f"(D-class samples {freq['D']*100:.0f}%) rather than on it"
        )
    elif feat["dominant"] == "B":
        intro_tail = (
            f"with the distribution leaning to the under-floor side "
            f"(B-class {freq['B']*100:.0f}%), suggestive of a settling Tsys"
        )
    else:
        intro_tail = (
            f"with an unusual skew toward the {feat['dominant']}-class bin; "
            f"receiver-temperature drift is the likeliest explanation"
        )
    parts.append(f"{intro}, {intro_tail}.")

    # 2) Baseline / receiver-temperature sentence
    parts.append(rng.choice(BASELINE_PHRASES).capitalize() + ".")

    # 3) Upper-tail branch — E rate drives transient/excursion content
    if feat["e_rate"] >= 0.03:
        parts.append(
            f"The upper tail shows {rng.choice(RARE_E_HIGH_WORDS)} "
            f"(E-class samples {freq['E']*100:.1f}%), {rng.choice(TRANSIENT_PHRASES)}."
        )
    elif feat["e_rate"] >= 0.01:
        parts.append(
            f"Rare E-class samples (~{freq['E']*100:.1f}%) register as "
            f"{rng.choice(TRANSIENT_PHRASES)}."
        )
    else:
        parts.append(
            f"{rng.choice(RARE_E_LOW_WORDS).capitalize()}; "
            f"E-class occupancy is {freq['E']*100:.1f}%."
        )

    # 4) Narrowband / RFI sentence — branch on upper vs lower tail balance
    if feat["upper"] > 0.25:
        parts.append(rng.choice(NARROWBAND_PHRASES).capitalize() + ".")
        parts.append(rng.choice(RFI_PHRASES).capitalize() + ".")
    elif feat["lower"] > 0.15:
        parts.append(
            f"On the low side, {rng.choice(RARE_A_WORDS)} "
            f"(A-class {freq['A']*100:.1f}%) read as baseline wander rather than signal."
        )
        parts.append(rng.choice(RFI_PHRASES).capitalize() + ".")
    else:
        parts.append(rng.choice(RFI_PHRASES).capitalize() + ".")

    # 5) No-persistent-structure statement — always present
    parts.append(rng.choice(NO_STRUCTURE_PHRASES).capitalize() + ".")

    # 6) Honesty-about-near-noise sentence
    parts.append(rng.choice(HONESTY_PHRASES).capitalize() + ".")

    # 7) Archival/next-step tag — short
    parts.append(rng.choice(NEXT_STEP_PHRASES).capitalize() + ".")

    text = " ".join(parts)
    return text


def wordcount(s: str) -> int:
    return len(s.split())


def main() -> None:
    # Load all 500 sequences
    records: dict[int, dict] = {}
    with INPUT.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            records[d["seq_idx"]] = d

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    out_lines: list[str] = []
    word_stats: list[int] = []

    for seq_idx in range(200, 500):
        rec = records[seq_idx]
        seq = rec["sequence"]
        rng = random.Random(0xC0FFEE ^ seq_idx)

        # Regenerate until word count falls in [60, 120]
        for _ in range(40):
            desc = describe(seq, rng)
            wc = wordcount(desc)
            if 60 <= wc <= 120:
                break
        else:
            # Clamp to window by trimming or padding as a last resort
            desc = describe(seq, rng)
            words = desc.split()
            if len(words) > 120:
                desc = " ".join(words[:120])
                if not desc.endswith("."):
                    desc = desc.rstrip(",;: ") + "."
            elif len(words) < 60:
                pad = " Log note: run archived as nominal background with no persistent structure."
                while wordcount(desc) < 60:
                    desc += pad

        word_stats.append(wordcount(desc))

        obj = {
            "domain": "seti",
            "seq_idx": seq_idx,
            "source": "natural",
            "description": desc,
        }
        out_lines.append(json.dumps(obj, ensure_ascii=False))

    OUTPUT.write_text("\n".join(out_lines) + "\n", encoding="utf-8")

    print(f"Wrote {len(out_lines)} lines to {OUTPUT}")
    print(
        "Word-count stats: min=%d, max=%d, mean=%.1f"
        % (min(word_stats), max(word_stats), sum(word_stats) / len(word_stats))
    )
    oob = [w for w in word_stats if w < 60 or w > 120]
    print(f"Out-of-band (60-120) count: {len(oob)}")


if __name__ == "__main__":
    main()
