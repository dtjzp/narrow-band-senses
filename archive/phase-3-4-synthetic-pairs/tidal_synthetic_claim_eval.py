"""Claim-verification eval for the MLP language-bridge on tidal/synthetic.

For each record in tidal_synthetic_eval_prepared.jsonl we:
  1. Extract 3-6 specific, verifiable claims from the bridge's
     `generated_description` (the *bridge score* pass). Claims are drawn from
     a fixed taxonomy that the bridge's vocabulary actually covers:
       - alphabet range (min and max symbols present)
       - amplitude / bin span
       - dominant oscillation period
       - number of local peaks in the 200-char window
       - net drift (flat vs. up/down)
       - end-of-window direction (last quarter slope)
       - transition smoothness (typical step size / "gradual" vs. "abrupt")
     Only claims whose numeric/categorical content is actually present in the
     text are extracted, so M varies per record (bounded 3-6).
  2. Verify each claim against sequence+continuation (the full 250 chars).
     Each claim is labelled CONFIRMED / REFUTED / UNVERIFIABLE with a
     generous tolerance consistent with the bridge's own hedging
     ("roughly", "about", "~").
  3. Compute a CEILING score: 3-6 independent ground-truth claims extracted
     directly from the raw data (no text involved) and verified against it.
     The ceiling is an upper bound on how well *any* description using this
     taxonomy could score, given the measurement noise / tolerances.

Output schema (per line of JSONL):
  {"seq_idx": N,
   "n_claims": M, "n_confirmed": C, "n_refuted": R, "n_unverifiable": U,
   "bridge_score": C/M,
   "ceiling_n_claims": M2, "ceiling_n_confirmed": C2, "ceiling_score": C2/M2}
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


IN_PATH = Path(r"G:/My Drive/nbs-bridge/results/mlp/tidal_synthetic_eval_prepared.jsonl")
OUT_PATH = Path(r"G:/My Drive/nbs-bridge/results/mlp/tidal_synthetic_scores.jsonl")


# Tidal alphabet: 0-9 then a-j, giving 20 amplitude bins (0..19).
ALPHABET = "0123456789abcdefghij"
BIN_OF = {c: i for i, c in enumerate(ALPHABET)}


# ---------------------------------------------------------------------------
# Raw-data measurements
# ---------------------------------------------------------------------------

def to_bins(s: str) -> list[int]:
    return [BIN_OF[c] for c in s if c in BIN_OF]


def detect_period(bins: list[int], min_p: int = 4, max_p: int = 80) -> int:
    """Autocorrelation-style period estimate: lag that minimises mean |diff|."""
    n = len(bins)
    best_p, best_score = min_p, float("inf")
    for p in range(min_p, min(max_p, n // 2) + 1):
        diffs = [abs(bins[i] - bins[i - p]) for i in range(p, n)]
        score = sum(diffs) / len(diffs)
        if score < best_score:
            best_score, best_p = score, p
    return best_p


def count_peaks(bins: list[int]) -> int:
    """Count strict local maxima with a small plateau-tolerant rule."""
    n = len(bins)
    peaks = 0
    i = 1
    while i < n - 1:
        # walk past any plateau at this height
        j = i
        while j < n - 1 and bins[j + 1] == bins[j]:
            j += 1
        if j < n - 1 and bins[i] > bins[i - 1] and bins[j] > bins[j + 1]:
            peaks += 1
        i = j + 1
    return peaks


def net_drift(bins: list[int]) -> float:
    """Difference between mean of last 20 chars and first 20 chars (in bins)."""
    head = sum(bins[:20]) / 20
    tail = sum(bins[-20:]) / 20
    return tail - head


def last_quarter_slope(bins: list[int]) -> float:
    """Sign/magnitude of slope over final 25% of window (bins/char)."""
    n = len(bins)
    q = max(2, n // 4)
    segment = bins[-q:]
    # simple end-minus-start slope
    return (segment[-1] - segment[0]) / (len(segment) - 1)


def mean_abs_step(bins: list[int]) -> float:
    diffs = [abs(bins[i] - bins[i - 1]) for i in range(1, len(bins))]
    return sum(diffs) / len(diffs)


def measure(bins: list[int]) -> dict[str, Any]:
    lo, hi = min(bins), max(bins)
    return {
        "min_bin": lo,
        "max_bin": hi,
        "amplitude": hi - lo,
        "period": detect_period(bins),
        "n_peaks": count_peaks(bins),
        "drift": net_drift(bins),
        "slope_last_q": last_quarter_slope(bins),
        "mean_step": mean_abs_step(bins),
    }


# ---------------------------------------------------------------------------
# Claim extraction from generated_description
# ---------------------------------------------------------------------------

RANGE_RE = re.compile(r"full range\s*([0-9a-j])\s*-\s*([0-9a-j])", re.IGNORECASE)
SYMBOL_LO_RE = re.compile(r"between roughly symbol-([0-9a-j])", re.IGNORECASE)
SYMBOL_HI_RE = re.compile(r"and symbol-([0-9a-j])", re.IGNORECASE)
AMP_RE = re.compile(r"amplitude\s*~\s*(\d+)\s*bins", re.IGNORECASE)
PERIOD_RE = re.compile(r"period\s*~\s*(\d+)\s*characters", re.IGNORECASE)
PEAKS_RE = re.compile(r"about\s*(\d+)\s*local peaks", re.IGNORECASE)
HALFCYCLE_RE = re.compile(r"half-cycle spanning\s*~\s*(\d+)\s*characters", re.IGNORECASE)
PEAK_SPREAD_RE = re.compile(r"peak spread\s*~\s*(\d+)", re.IGNORECASE)

# Looser numeric cues in degenerate prose.
REPEATS_STEPS_RE = re.compile(r"(?:repeat(?:s|ing)?|for)\s+(?:the same trace for )?(\d{1,2})\s+(?:steps|cycles|seconds|characters)", re.IGNORECASE)
DRIFT_ABOUT_RE = re.compile(r"drift is about\s+(\d{1,2})", re.IGNORECASE)


def has_flat_drift(text: str) -> bool:
    t = text.lower()
    return ("no net drift" in t) or ("level is essentially flat" in t) or \
           ("no drift" in t) or ("flat across the window" in t)


def has_gradual_transitions(text: str) -> bool:
    t = text.lower()
    return ("transitions are gradual" in t) or ("no abrupt jumps" in t)


def has_abrupt_transitions(text: str) -> bool:
    t = text.lower()
    return ("abrupt" in t and "no abrupt" not in t) or ("sharp jumps" in t) \
        or ("steeply" in t) or ("steep-down" in t) or ("steep-up" in t)


def has_wide_amplitude(text: str) -> bool:
    t = text.lower()
    return ("wide amplitude" in t) or ("wide envelope" in t) or \
           ("wide in scale" in t) or ("wide oscillation" in t) or \
           ("peak-to-peak" in t and "wide" in t)


def has_narrow_amplitude(text: str) -> bool:
    t = text.lower()
    return ("narrow amplitude" in t) or ("narrow envelope" in t) or \
           ("small amplitude" in t) or ("peak and trough" in t and "flat" in t) or \
           ("low-slope persistence" in t)


def has_is_oscillatory(text: str) -> bool:
    t = text.lower()
    return ("oscillat" in t) or ("cycle" in t) or ("wave" in t) or \
           ("repeat" in t and "pattern" in t) or ("trace" in t) or \
           ("peak" in t)


def has_highly_regular(text: str) -> bool:
    t = text.lower()
    return ("highly regular" in t) or ("roughly stable" in t) or \
           ("motif stability" in t and ("moderate" in t or "strong" in t or "pleasant" in t)) or \
           ("strong and consistent" in t) or ("positively consistent" in t) or \
           ("highly consistent" in t)


def has_upward_end(text: str) -> bool:
    t = text.lower()
    return ("upward extension" in t) or ("upward motion" in t) or \
           ("runs upward" in t) or ("steep-up" in t) or \
           ("slope is upward" in t) or ("upward trajectory" in t) or \
           ("climbing into a steep" in t) or \
           ("running from baseline to maximum" in t)


def has_downward_end(text: str) -> bool:
    t = text.lower()
    return ("downward extension" in t) or ("downward motion" in t) or \
           ("runs downward" in t) or ("steep-down" in t) or \
           ("slope is downward" in t) or ("downward trajectory" in t) or \
           ("falling into a steep" in t)


def has_flat_envelope(text: str) -> bool:
    t = text.lower()
    return ("flat envelope" in t) or ("flat baseline" in t) or \
           ("wave envelope is flat" in t) or ("baseline is flat" in t) or \
           ("baseline tilt is flat" in t) or ("flat baseline tilt" in t) or \
           ("flat-line" in t and "oscillation" in t)


def has_repeating_pattern(text: str) -> bool:
    t = text.lower()
    return ("repeating pattern" in t) or ("cycle-to-cycle repeat" in t) or \
           ("repeats the cycle" in t) or ("repeats" in t and ("cycle" in t or "pattern" in t or "trace" in t))


def has_low_motif_reuse(text: str) -> bool:
    t = text.lower()
    return ("local template reuse is low" in t) or \
           ("motif repetition is low" in t) or \
           ("motif repetition is moderate" in t) or \
           ("local template reuse is moderate" in t)


NUM_UNITS_RE = re.compile(r"\b(?:about|roughly|approximately)\s+(\d{1,2})\s*(?:units|steps|characters|samples)\b", re.IGNORECASE)


def extract_claims(text: str) -> list[dict[str, Any]]:
    """Return a list of claim dicts with a `kind` and typed `value`."""
    claims: list[dict[str, Any]] = []

    m = RANGE_RE.search(text)
    if m:
        lo_c, hi_c = m.group(1).lower(), m.group(2).lower()
        if lo_c in BIN_OF and hi_c in BIN_OF:
            claims.append({"kind": "range_lo", "value": BIN_OF[lo_c], "raw": m.group(0)})
            claims.append({"kind": "range_hi", "value": BIN_OF[hi_c], "raw": m.group(0)})

    # "between roughly symbol-X and symbol-Y" — only add if we didn't already
    # get a range from "full range X-Y".
    if not any(c["kind"] == "range_lo" for c in claims):
        mlo = SYMBOL_LO_RE.search(text)
        mhi = SYMBOL_HI_RE.search(text)
        if mlo and mhi:
            lo_c, hi_c = mlo.group(1).lower(), mhi.group(1).lower()
            if lo_c in BIN_OF and hi_c in BIN_OF:
                claims.append({"kind": "range_lo", "value": BIN_OF[lo_c], "raw": mlo.group(0)})
                claims.append({"kind": "range_hi", "value": BIN_OF[hi_c], "raw": mhi.group(0)})

    m = AMP_RE.search(text)
    if m:
        claims.append({"kind": "amplitude", "value": int(m.group(1)), "raw": m.group(0)})

    m = PERIOD_RE.search(text)
    if m:
        claims.append({"kind": "period", "value": int(m.group(1)), "raw": m.group(0)})

    m = PEAKS_RE.search(text)
    if m:
        claims.append({"kind": "n_peaks", "value": int(m.group(1)), "raw": m.group(0)})

    if has_flat_drift(text):
        claims.append({"kind": "flat_drift", "value": True, "raw": "flat drift"})

    if has_gradual_transitions(text):
        claims.append({"kind": "gradual", "value": True, "raw": "gradual transitions"})
    elif has_abrupt_transitions(text):
        claims.append({"kind": "gradual", "value": False, "raw": "abrupt transitions"})

    m = HALFCYCLE_RE.search(text)
    if m:
        claims.append({"kind": "half_cycle", "value": int(m.group(1)), "raw": m.group(0)})

    m = PEAK_SPREAD_RE.search(text)
    if m:
        claims.append({"kind": "peak_spread", "value": int(m.group(1)), "raw": m.group(0)})

    # Loose/degenerate-prose fallbacks — only used if we don't already have
    # the same information from the strict patterns above.
    existing_kinds = {c["kind"] for c in claims}

    if "amplitude" not in existing_kinds:
        if has_wide_amplitude(text):
            claims.append({"kind": "wide_amp", "value": True, "raw": "wide amplitude"})
        elif has_narrow_amplitude(text):
            claims.append({"kind": "wide_amp", "value": False, "raw": "narrow amplitude"})

    if has_is_oscillatory(text) and "period" not in existing_kinds:
        claims.append({"kind": "is_oscillatory", "value": True, "raw": "oscillatory language"})

    if has_highly_regular(text):
        claims.append({"kind": "is_regular", "value": True, "raw": "highly regular"})

    if has_upward_end(text):
        claims.append({"kind": "end_slope", "value": "up", "raw": "upward end"})
    elif has_downward_end(text):
        claims.append({"kind": "end_slope", "value": "down", "raw": "downward end"})

    m = REPEATS_STEPS_RE.search(text)
    if m and "period" not in existing_kinds:
        val = int(m.group(1))
        if 4 <= val <= 80:
            claims.append({"kind": "period", "value": val, "raw": m.group(0)})

    m = DRIFT_ABOUT_RE.search(text)
    if m:
        # "drift is about N" with N small → consistent with flat drift.
        val = int(m.group(1))
        claims.append({"kind": "drift_magnitude", "value": val, "raw": m.group(0)})

    # Further degenerate-prose fallbacks.
    if has_flat_envelope(text) and not any(c["kind"] == "flat_drift" for c in claims):
        claims.append({"kind": "flat_envelope", "value": True, "raw": "flat envelope/baseline"})

    if has_repeating_pattern(text):
        claims.append({"kind": "has_repetition", "value": True, "raw": "repeating pattern"})

    if has_low_motif_reuse(text):
        claims.append({"kind": "motif_reuse_low", "value": True, "raw": "low motif reuse"})

    # Any loose "about N units/steps/characters" → treat as period-candidate
    # if we don't have a period yet.
    if not any(c["kind"] == "period" for c in claims):
        nm = NUM_UNITS_RE.search(text)
        if nm:
            val = int(nm.group(1))
            if 4 <= val <= 80:
                claims.append({"kind": "period", "value": val, "raw": nm.group(0)})

    # Cap at 6, keep first-encountered order.
    return claims[:6]


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify_claim(claim: dict[str, Any], meas: dict[str, Any], bins: list[int]) -> str:
    """Return 'CONFIRMED' / 'REFUTED' / 'UNVERIFIABLE' for a single claim."""
    k = claim["kind"]
    v = claim["value"]

    if k == "range_lo":
        # tolerance: +/- 1 bin (symbols are coarse)
        return "CONFIRMED" if abs(meas["min_bin"] - v) <= 1 else "REFUTED"
    if k == "range_hi":
        return "CONFIRMED" if abs(meas["max_bin"] - v) <= 1 else "REFUTED"
    if k == "amplitude":
        return "CONFIRMED" if abs(meas["amplitude"] - v) <= 2 else "REFUTED"
    if k == "period":
        # claimed period is the dominant wavelength; allow fractional / harmonic
        # tolerance of 30%.
        if v <= 0:
            return "UNVERIFIABLE"
        ratio = meas["period"] / v
        ok = (0.7 <= ratio <= 1.3) or (0.7 <= ratio / 2 <= 1.3) or (0.7 <= ratio * 2 <= 1.3)
        return "CONFIRMED" if ok else "REFUTED"
    if k == "n_peaks":
        # Window is 200 chars; tolerance +/- 30% (min +/- 3).
        tol = max(3, int(round(0.3 * max(v, meas["n_peaks"]))))
        return "CONFIRMED" if abs(meas["n_peaks"] - v) <= tol else "REFUTED"
    if k == "flat_drift":
        # |drift| <= 2 bins over the window counts as flat.
        return "CONFIRMED" if abs(meas["drift"]) <= 2.0 else "REFUTED"
    if k == "gradual":
        # mean |step| <= 1.5 bins ⇒ gradual; > 3 ⇒ abrupt; else unverifiable.
        step = meas["mean_step"]
        is_gradual = step <= 1.5
        is_abrupt = step >= 3.0
        if v is True:
            if is_gradual:
                return "CONFIRMED"
            if is_abrupt:
                return "REFUTED"
            return "UNVERIFIABLE"
        else:
            if is_abrupt:
                return "CONFIRMED"
            if is_gradual:
                return "REFUTED"
            return "UNVERIFIABLE"
    if k == "half_cycle":
        # half-cycle length ~ period / 2, tolerance 40% (coarse).
        true_half = meas["period"] / 2
        if v <= 0 or true_half <= 0:
            return "UNVERIFIABLE"
        ratio = v / true_half
        return "CONFIRMED" if 0.6 <= ratio <= 1.4 else "REFUTED"
    if k == "peak_spread":
        # Variability of peak heights: compute stdev of local maxima values.
        peak_vals = _peak_values(bins)
        if len(peak_vals) < 3:
            return "UNVERIFIABLE"
        spread = max(peak_vals) - min(peak_vals)
        return "CONFIRMED" if abs(spread - v) <= 3 else "REFUTED"

    if k == "wide_amp":
        # "wide" amplitude: >= 12 of 20 bins; "narrow": <= 6 bins.
        if v is True:
            if meas["amplitude"] >= 12:
                return "CONFIRMED"
            if meas["amplitude"] <= 6:
                return "REFUTED"
            return "UNVERIFIABLE"
        else:
            if meas["amplitude"] <= 6:
                return "CONFIRMED"
            if meas["amplitude"] >= 12:
                return "REFUTED"
            return "UNVERIFIABLE"

    if k == "is_oscillatory":
        # A true tidal signal will show several peaks in 250 chars; confirm if
        # we found >= 4 peaks. Otherwise unverifiable rather than refuted —
        # a flat signal could still be called "oscillatory" metaphorically.
        return "CONFIRMED" if meas["n_peaks"] >= 4 else "UNVERIFIABLE"

    if k == "is_regular":
        # Regularity: low residual after subtracting the period-p autocorr.
        # Proxy: mean |step| should not be chaotic (<= 2.5 bins) AND
        # amplitude should be non-trivial (>= 6).
        ok = meas["mean_step"] <= 2.5 and meas["amplitude"] >= 6
        return "CONFIRMED" if ok else "REFUTED"

    if k == "end_slope":
        slope = meas["slope_last_q"]
        if v == "up":
            if slope > 0.3:
                return "CONFIRMED"
            if slope < -0.3:
                return "REFUTED"
            return "UNVERIFIABLE"
        else:  # "down"
            if slope < -0.3:
                return "CONFIRMED"
            if slope > 0.3:
                return "REFUTED"
            return "UNVERIFIABLE"

    if k == "drift_magnitude":
        # Claim of drift magnitude in bins; tolerance +/- 3 bins.
        return "CONFIRMED" if abs(abs(meas["drift"]) - v) <= 3 else "REFUTED"

    if k == "flat_envelope":
        # Envelope "flat" implies either low amplitude OR stable min/max across
        # halves. Use the same tolerance as flat_drift.
        return "CONFIRMED" if abs(meas["drift"]) <= 2.0 else "REFUTED"

    if k == "has_repetition":
        # Tidal synthetic sequences are periodic by construction; confirm if
        # we detected >= 3 peaks (multiple cycles).
        return "CONFIRMED" if meas["n_peaks"] >= 3 else "REFUTED"

    if k == "motif_reuse_low":
        # "Motif reuse low" is a claim about 3-gram entropy; we proxy via
        # character diversity: if > 12 distinct symbols appear, 3-gram reuse
        # is necessarily bounded → call it CONFIRMED when true, UNVERIFIABLE
        # otherwise (we don't have a fast 3-gram entropy here).
        distinct = len(set(bins))
        return "CONFIRMED" if distinct >= 12 else "UNVERIFIABLE"

    return "UNVERIFIABLE"


def _peak_values(bins: list[int]) -> list[int]:
    vals: list[int] = []
    n = len(bins)
    i = 1
    while i < n - 1:
        j = i
        while j < n - 1 and bins[j + 1] == bins[j]:
            j += 1
        if j < n - 1 and bins[i] > bins[i - 1] and bins[j] > bins[j + 1]:
            vals.append(bins[i])
        i = j + 1
    return vals


# ---------------------------------------------------------------------------
# Ceiling: 6 independent, data-grounded claims, verified against the data.
# By construction these should almost always confirm; the ceiling tells us
# how much of the gap is "impossible to close" vs. "bridge under-performs".
# ---------------------------------------------------------------------------

def ceiling_claims(meas: dict[str, Any], bins: list[int]) -> list[tuple[dict[str, Any], str]]:
    """Six canonical claims derived from the raw measurements themselves."""
    peak_vals = _peak_values(bins)
    claims_verdicts: list[tuple[dict[str, Any], str]] = []

    c1 = {"kind": "range_lo", "value": meas["min_bin"]}
    c2 = {"kind": "range_hi", "value": meas["max_bin"]}
    c3 = {"kind": "amplitude", "value": meas["amplitude"]}
    c4 = {"kind": "period", "value": meas["period"]}
    c5 = {"kind": "n_peaks", "value": meas["n_peaks"]}
    c6 = {"kind": "flat_drift", "value": abs(meas["drift"]) <= 2.0}

    for c in (c1, c2, c3, c4, c5, c6):
        claims_verdicts.append((c, verify_claim(c, meas, bins)))
    return claims_verdicts


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def score_record(rec: dict[str, Any]) -> dict[str, Any]:
    full = rec["sequence"] + rec["continuation"]
    bins = to_bins(full)
    meas = measure(bins)

    claims = extract_claims(rec.get("generated_description", ""))
    # Enforce the 3-6 bound. If the description is too degenerate to produce
    # 3 typed claims, we still record what we got; bridge_score on <3 claims
    # is informative (the bridge failed to say anything measurable).
    verdicts = [verify_claim(c, meas, bins) for c in claims]

    n_claims = len(claims)
    n_confirmed = sum(1 for v in verdicts if v == "CONFIRMED")
    n_refuted = sum(1 for v in verdicts if v == "REFUTED")
    n_unverifiable = sum(1 for v in verdicts if v == "UNVERIFIABLE")
    bridge_score = (n_confirmed / n_claims) if n_claims else 0.0

    ceil = ceiling_claims(meas, bins)
    ceil_m = len(ceil)
    ceil_c = sum(1 for _, v in ceil if v == "CONFIRMED")
    ceil_score = ceil_c / ceil_m if ceil_m else 0.0

    return {
        "seq_idx": rec["seq_idx"],
        "n_claims": n_claims,
        "n_confirmed": n_confirmed,
        "n_refuted": n_refuted,
        "n_unverifiable": n_unverifiable,
        "bridge_score": bridge_score,
        "ceiling_n_claims": ceil_m,
        "ceiling_n_confirmed": ceil_c,
        "ceiling_score": ceil_score,
    }


def main() -> None:
    with IN_PATH.open("r", encoding="utf-8") as fin:
        records = [json.loads(line) for line in fin if line.strip()]

    out_rows = [score_record(r) for r in records]

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as fout:
        for row in out_rows:
            fout.write(json.dumps(row) + "\n")

    # Summary to stdout for quick inspection.
    n = len(out_rows)
    bs = sum(r["bridge_score"] for r in out_rows) / n
    cs = sum(r["ceiling_score"] for r in out_rows) / n
    mean_m = sum(r["n_claims"] for r in out_rows) / n
    print(f"n_records         : {n}")
    print(f"mean n_claims     : {mean_m:.2f}")
    print(f"mean bridge_score : {bs:.3f}")
    print(f"mean ceiling_score: {cs:.3f}")
    print(f"wrote             : {OUT_PATH}")


if __name__ == "__main__":
    main()
