"""Claim-verification eval for MLP bridge: tidal / natural.

Reads per-record generated_description, extracts specific verifiable claims,
verifies them against sequence+continuation (character-level tidal quantile
signal: 20 bins encoded as '0'-'9','a'-'j'). Writes per-record JSONL scores.

Usage:
    python score_tidal_natural.py
"""
import json
import re
from pathlib import Path

INPUT = Path("G:/My Drive/nbs-bridge/results/mlp/tidal_natural_eval_prepared.jsonl")
OUTPUT = Path("G:/My Drive/nbs-bridge/results/mlp/tidal_natural_scores.jsonl")


# --------------------------------------------------------------------------
# Signal utilities: decode quantile-bin characters -> integer values 0..19
# --------------------------------------------------------------------------

def decode(s: str):
    out = []
    for c in s:
        if c.isdigit():
            out.append(int(c))
        elif "a" <= c <= "j":
            out.append(10 + ord(c) - ord("a"))
        else:
            # unexpected char; skip
            pass
    return out


def count_extrema(vals, min_prominence=2):
    """Count local maxima (highs) and minima (lows) with a light prominence filter.

    A "high" at i: vals[i] strictly greater than its two neighbors AND within
    a short window the local swing is >= min_prominence.
    """
    highs, lows = [], []
    n = len(vals)
    if n < 3:
        return highs, lows
    for i in range(1, n - 1):
        left = vals[i - 1]
        right = vals[i + 1]
        v = vals[i]
        # Handle plateau peaks: find strict peak by looking further if equal.
        if v > left and v > right:
            # prominence: drop to nearest lower neighbor within window
            w = 8
            lo = min(vals[max(0, i - w): i + w + 1])
            if v - lo >= min_prominence:
                highs.append(i)
        elif v < left and v < right:
            w = 8
            hi = max(vals[max(0, i - w): i + w + 1])
            if hi - v >= min_prominence:
                lows.append(i)
    return highs, lows


def estimate_period(highs):
    if len(highs) < 2:
        return None
    gaps = [highs[i + 1] - highs[i] for i in range(len(highs) - 1)]
    return sum(gaps) / len(gaps)


def amplitude_stats(vals, highs, lows):
    peak_vals = [vals[i] for i in highs] if highs else []
    trough_vals = [vals[i] for i in lows] if lows else []
    return {
        "peak_mean": (sum(peak_vals) / len(peak_vals)) if peak_vals else None,
        "trough_mean": (sum(trough_vals) / len(trough_vals)) if trough_vals else None,
        "peak_max": max(peak_vals) if peak_vals else None,
        "trough_min": min(trough_vals) if trough_vals else None,
        "vmax": max(vals) if vals else None,
        "vmin": min(vals) if vals else None,
    }


def envelope_trend(vals, highs):
    """Return 'rising', 'falling', 'flat' for the high-water envelope across record."""
    if len(highs) < 4:
        return "insufficient"
    peak_vals = [vals[i] for i in highs]
    first_half = peak_vals[: len(peak_vals) // 2]
    second_half = peak_vals[len(peak_vals) // 2:]
    a = sum(first_half) / len(first_half)
    b = sum(second_half) / len(second_half)
    if b - a >= 0.5:
        return "rising"
    if a - b >= 0.5:
        return "falling"
    return "flat"


# --------------------------------------------------------------------------
# Claim extraction from generated_description
# --------------------------------------------------------------------------

NUM = r"(\d+(?:\.\d+)?)"


def extract_claims(text: str):
    """Return list of (claim_kind, params, raw_snippet) tuples.

    Claim kinds:
      - n_highs: expected number of high-water peaks
      - n_lows: expected number of low-water troughs
      - peak_bin: expected high-water bin value (0..19)
      - trough_bin: expected low-water bin value
      - period: approx oscillation period in characters
      - envelope: 'rising' / 'falling' / 'flat' / 'upswing' / 'downswing'
      - semidiurnal: claims semidiurnal M2 dominance (period ~12-13 chars)
      - range_pct: tidal range as pct of amplitude envelope
      - diurnal: claims diurnal dominance (period ~24 chars)
      - mixed: claims mixed-tide pattern
    """
    claims = []
    t = text.lower()

    # number of high waters
    for m in re.finditer(r"(\d+)\s+high\s+waters?", t):
        claims.append(("n_highs", {"n": int(m.group(1))}, m.group(0)))
    for m in re.finditer(r"(\d+)\s+low\s+waters?", t):
        claims.append(("n_lows", {"n": int(m.group(1))}, m.group(0)))

    # peak / high near bin N
    for m in re.finditer(r"high[s]?\s+(?:cresting|crowding|near|at|reach(?:ing)?|peak(?:ing)?)\s+(?:the\s+)?(?:bin\s+)?" + NUM + r"(?:[-\s]bin)?\s*(?:level|bin|mark)?", t):
        try:
            v = float(m.group(1))
            if 0 <= v <= 19:
                claims.append(("peak_bin", {"v": v}, m.group(0)))
        except ValueError:
            pass
    for m in re.finditer(r"cresting\s+near\s+bin\s+" + NUM, t):
        claims.append(("peak_bin", {"v": float(m.group(1))}, m.group(0)))

    # low / trough near bin N
    for m in re.finditer(r"low[s]?\s+(?:settling|near|at|sitting|bottoming)\s+(?:the\s+)?(?:bin\s+)?" + NUM, t):
        try:
            v = float(m.group(1))
            if 0 <= v <= 19:
                claims.append(("trough_bin", {"v": v}, m.group(0)))
        except ValueError:
            pass
    for m in re.finditer(r"lows?\s+near\s+bin\s+" + NUM, t):
        claims.append(("trough_bin", {"v": float(m.group(1))}, m.group(0)))

    # semidiurnal / M2
    if re.search(r"semidiurnal|m2\b|twice[-\s]daily", t):
        claims.append(("semidiurnal", {}, "semidiurnal/M2/twice-daily"))
    # diurnal (if not semidiurnal)
    if re.search(r"\bdiurnal\b", t) and not re.search(r"semidiurnal", t):
        claims.append(("diurnal", {}, "diurnal"))
    # mixed tide
    if re.search(r"mixed[-\s]tide|mixed\s+semidiurnal", t):
        claims.append(("mixed", {}, "mixed-tide"))

    # envelope
    if re.search(r"upswing|rising\s+envelope|amplitude\s+grows|toward\s+syzygy|toward\s+springs?", t):
        claims.append(("envelope", {"dir": "rising"}, "rising envelope"))
    if re.search(r"downswing|falling\s+envelope|amplitude\s+(?:shrinks|wanes|fades)|toward\s+neaps?", t):
        claims.append(("envelope", {"dir": "falling"}, "falling envelope"))

    # range pct (macrotidal / range pct)
    for m in re.finditer(r"(?:range|span|envelope)[^.]{0,30}?" + NUM + r"\s*%", t):
        try:
            v = float(m.group(1))
            if 10 <= v <= 100:
                claims.append(("range_pct", {"v": v}, m.group(0)))
        except ValueError:
            pass

    # period in hours (characters)
    for m in re.finditer(r"period[^.]{0,20}?" + NUM + r"\s*(?:h|hour)", t):
        try:
            v = float(m.group(1))
            claims.append(("period", {"h": v}, m.group(0)))
        except ValueError:
            pass

    # Deduplicate (kind, str(params)) keeping first snippet
    seen = set()
    unique = []
    for kind, params, snip in claims:
        key = (kind, json.dumps(params, sort_keys=True))
        if key not in seen:
            seen.add(key)
            unique.append((kind, params, snip))
    return unique


# --------------------------------------------------------------------------
# Verify a claim against the decoded signal
# --------------------------------------------------------------------------

def verify_claim(kind, params, vals_full, highs, lows, period, stats, env_trend):
    """Return 'confirmed' | 'refuted' | 'unverifiable'."""
    if kind == "n_highs":
        expected = params["n"]
        actual = len(highs)
        if abs(actual - expected) <= max(2, 0.2 * actual):
            return "confirmed"
        return "refuted"
    if kind == "n_lows":
        expected = params["n"]
        actual = len(lows)
        if abs(actual - expected) <= max(2, 0.2 * actual):
            return "confirmed"
        return "refuted"
    if kind == "peak_bin":
        if stats["peak_mean"] is None:
            return "unverifiable"
        # allow +/-2 bins tolerance, and also pass if peak_max within 2
        diff_mean = abs(params["v"] - stats["peak_mean"])
        diff_max = abs(params["v"] - stats["peak_max"])
        if diff_mean <= 2.5 or diff_max <= 2:
            return "confirmed"
        return "refuted"
    if kind == "trough_bin":
        if stats["trough_mean"] is None:
            return "unverifiable"
        diff_mean = abs(params["v"] - stats["trough_mean"])
        diff_min = abs(params["v"] - stats["trough_min"])
        if diff_mean <= 2.5 or diff_min <= 2:
            return "confirmed"
        return "refuted"
    if kind == "semidiurnal":
        # In our 200-char window with 16 highs -> ~12.5 char period.
        # Accept period in [10, 16] as semidiurnal.
        if period is None:
            return "unverifiable"
        if 10 <= period <= 16:
            return "confirmed"
        return "refuted"
    if kind == "diurnal":
        if period is None:
            return "unverifiable"
        if 20 <= period <= 28:
            return "confirmed"
        return "refuted"
    if kind == "mixed":
        # Mixed = clear alternation of high-high vs low-high. Hard to verify
        # strictly; accept if peak_vals show >=2 bin variance.
        if len(highs) < 4:
            return "unverifiable"
        peaks = [vals_full[i] for i in highs]
        mean = sum(peaks) / len(peaks)
        var = sum((p - mean) ** 2 for p in peaks) / len(peaks)
        if var >= 1.0:
            return "confirmed"
        return "refuted"
    if kind == "envelope":
        if env_trend == "insufficient":
            return "unverifiable"
        if params["dir"] == env_trend:
            return "confirmed"
        if env_trend == "flat":
            return "refuted"
        if params["dir"] != env_trend:
            return "refuted"
    if kind == "range_pct":
        if stats["vmax"] is None:
            return "unverifiable"
        actual_pct = (stats["vmax"] - stats["vmin"]) / 19.0 * 100
        if abs(actual_pct - params["v"]) <= 20:
            return "confirmed"
        return "refuted"
    if kind == "period":
        if period is None:
            return "unverifiable"
        # description says period in hours; one char = 1 hour in our encoding
        if abs(period - params["h"]) <= 3:
            return "confirmed"
        return "refuted"
    return "unverifiable"


# --------------------------------------------------------------------------
# Ceiling claims — directly extracted from raw data
# --------------------------------------------------------------------------

def ceiling_claims(vals_full, highs, lows, period, stats, env_trend):
    """Construct 3-6 ground-truth claims from the raw signal and verify each.

    Since we derived these from the data, each is either 'confirmed' (when
    the derivation is meaningful) or 'unverifiable' (when the signal is
    too short / flat). Ceiling_score measures how many of the available
    channels yielded a verifiable, true claim.
    """
    n_claims = 0
    n_confirmed = 0

    # 1. high count
    n_claims += 1
    if len(highs) >= 1:
        n_confirmed += 1
    # 2. low count
    n_claims += 1
    if len(lows) >= 1:
        n_confirmed += 1
    # 3. peak bin mean
    n_claims += 1
    if stats["peak_mean"] is not None:
        n_confirmed += 1
    # 4. trough bin mean
    n_claims += 1
    if stats["trough_mean"] is not None:
        n_confirmed += 1
    # 5. period
    n_claims += 1
    if period is not None:
        n_confirmed += 1
    # 6. envelope trend
    n_claims += 1
    if env_trend != "insufficient":
        n_confirmed += 1
    return n_claims, n_confirmed


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    records = []
    with INPUT.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    out_lines = []
    for r in records:
        seq = r["sequence"]
        cont = r["continuation"]
        full = seq + cont
        vals_full = decode(full)
        highs, lows = count_extrema(vals_full)
        period = estimate_period(highs)
        stats = amplitude_stats(vals_full, highs, lows)
        env_trend = envelope_trend(vals_full, highs)

        claims = extract_claims(r.get("generated_description", ""))
        n_claims = len(claims)
        n_confirmed = 0
        n_refuted = 0
        n_unverifiable = 0
        for kind, params, _snip in claims:
            v = verify_claim(kind, params, vals_full, highs, lows, period, stats, env_trend)
            if v == "confirmed":
                n_confirmed += 1
            elif v == "refuted":
                n_refuted += 1
            else:
                n_unverifiable += 1

        ceil_n, ceil_conf = ceiling_claims(vals_full, highs, lows, period, stats, env_trend)

        row = {
            "seq_idx": r["seq_idx"],
            "n_claims": n_claims,
            "n_confirmed": n_confirmed,
            "n_refuted": n_refuted,
            "n_unverifiable": n_unverifiable,
            "bridge_score": (n_confirmed / n_claims) if n_claims else 0.0,
            "ceiling_n_claims": ceil_n,
            "ceiling_n_confirmed": ceil_conf,
            "ceiling_score": (ceil_conf / ceil_n) if ceil_n else 0.0,
        }
        out_lines.append(json.dumps(row))

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    print(f"Wrote {len(out_lines)} rows to {OUTPUT}")

    # quick aggregate
    scores = [json.loads(l) for l in out_lines]
    if scores:
        avg_bridge = sum(s["bridge_score"] for s in scores) / len(scores)
        avg_ceil = sum(s["ceiling_score"] for s in scores) / len(scores)
        avg_nclaims = sum(s["n_claims"] for s in scores) / len(scores)
        print(f"mean bridge_score = {avg_bridge:.3f}")
        print(f"mean ceiling_score = {avg_ceil:.3f}")
        print(f"mean n_claims/record = {avg_nclaims:.2f}")


if __name__ == "__main__":
    main()
