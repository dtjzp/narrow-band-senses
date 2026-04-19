"""Claim-verification scorer for MLP bridge, domain=gcode, source=natural.

For each of 50 eval records:
  1. Extract 3-6 SPECIFIC verifiable claims from `generated_description`.
     Descriptions use 3D-printing vocab (G0/G1/retraction/feedrate/extrusion).
     Translate to token-level facts checkable against the sequence+continuation.
  2. Verify each claim against (sequence + continuation). Emit
     CONFIRMED / REFUTED / UNVERIFIABLE.
  3. Compute a CEILING: 3-6 independent claims made directly from the data
     (ground-truth facts about sequence+continuation), and their confirmation
     rate (should be ~1.0, gives an upper bound on possible bridge_score).

Output JSONL at G:/My Drive/nbs-bridge/results/mlp/gcode_natural_scores.jsonl.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

IN_PATH = Path("G:/My Drive/nbs-bridge/results/mlp/gcode_natural_eval_prepared.jsonl")
OUT_PATH = Path("G:/My Drive/nbs-bridge/results/mlp/gcode_natural_scores.jsonl")


# ----- helpers: parse gcode token stream -------------------------------------

FLOAT = r"-?\d+(?:\.\d+)?"

def find_axis_values(text: str, axis: str) -> list[float]:
    """Return the list of floats associated with a given axis letter in a gcode stream."""
    pattern = re.compile(rf"(?<![A-Za-z]){axis}({FLOAT})")
    return [float(m.group(1)) for m in pattern.finditer(text)]


def has_command(text: str, cmd: str) -> bool:
    """Check whether a G/M command token (e.g. G0, G1, M204) appears as a standalone token."""
    # cmd followed by a non-digit or end-of-string
    return re.search(rf"(?<![A-Za-z0-9]){re.escape(cmd)}(?![0-9])", text) is not None


def f_values(text: str) -> list[float]:
    return find_axis_values(text, "F")


# ----- claim extractors ------------------------------------------------------

# A "claim" is a dict: {"text": str, "verdict": "CONFIRMED"|"REFUTED"|"UNVERIFIABLE"}
# Each extractor returns a list of claims (or []).

def _x_range(text: str) -> tuple[float, float] | None:
    xs = find_axis_values(text, "X")
    if not xs:
        return None
    return (min(xs), max(xs))


def _y_range(text: str) -> tuple[float, float] | None:
    ys = find_axis_values(text, "Y")
    if not ys:
        return None
    return (min(ys), max(ys))


def _e_range(text: str) -> tuple[float, float] | None:
    es = find_axis_values(text, "E")
    if not es:
        return None
    return (min(es), max(es))


def extract_bridge_claims(desc: str, data: str) -> list[dict]:
    """Extract claims from the bridge-generated description, verify each against `data`.

    Claims are token-level facts: coordinate approximate values, axis progression
    direction, command presence (G0/G1/retraction/M codes), feedrate values.
    """
    claims: list[dict] = []

    x_rng = _x_range(data)
    y_rng = _y_range(data)
    e_rng = _e_range(data)
    fs = f_values(data)

    # 1. Specific X coordinate mentions (e.g. "X134.80", "near X130.12")
    for m in re.finditer(r"X\s*=?\s*(" + FLOAT + r")", desc):
        val = float(m.group(1))
        if val < 0 or val > 10000:
            continue
        if x_rng is None:
            verdict = "REFUTED"  # claim X value but no X in data
        else:
            # Tolerate +/- 2mm around observed X range
            lo, hi = x_rng
            verdict = "CONFIRMED" if (lo - 2.0) <= val <= (hi + 2.0) else "REFUTED"
        claims.append({"text": f"X approximately {val}", "verdict": verdict})
        if len(claims) >= 6:
            break

    # 2. Specific Y coordinate mentions
    if len(claims) < 6:
        for m in re.finditer(r"Y\s*=?\s*(" + FLOAT + r")", desc):
            val = float(m.group(1))
            if val < 0 or val > 10000:
                continue
            if y_rng is None:
                verdict = "REFUTED"
            else:
                lo, hi = y_rng
                verdict = "CONFIRMED" if (lo - 2.0) <= val <= (hi + 2.0) else "REFUTED"
            claims.append({"text": f"Y approximately {val}", "verdict": verdict})
            if len(claims) >= 6:
                break

    # 3. E axis value mentions (e.g. "E=20.049", "from 19.738 toward 19.747")
    if len(claims) < 6:
        for m in re.finditer(r"E\s*=\s*(" + FLOAT + r")", desc):
            val = float(m.group(1))
            if e_rng is None:
                verdict = "REFUTED"
            else:
                lo, hi = e_rng
                # E tolerance ~0.5 (E advances slowly)
                verdict = "CONFIRMED" if (lo - 0.5) <= val <= (hi + 0.5) else "REFUTED"
            claims.append({"text": f"E approximately {val}", "verdict": verdict})
            if len(claims) >= 6:
                break

    # 4. G0 travel command mention
    if len(claims) < 6:
        if re.search(r"\bG0\b|G0\s*travel|travel\s*move", desc, re.IGNORECASE):
            verdict = "CONFIRMED" if has_command(data, "G0") else "REFUTED"
            claims.append({"text": "G0 travel command present", "verdict": verdict})

    # 5. G1 extrusion mention
    if len(claims) < 6:
        if re.search(r"\bG1\b|extrusion|extruding|deposition", desc, re.IGNORECASE):
            verdict = "CONFIRMED" if has_command(data, "G1") else "REFUTED"
            claims.append({"text": "G1 extrusion command present", "verdict": verdict})

    # 6. Retraction (E decreasing) mention
    if len(claims) < 6:
        if re.search(r"retract", desc, re.IGNORECASE):
            if e_rng is None:
                verdict = "REFUTED"
            else:
                # retraction -> some E value should decrease
                es = find_axis_values(data, "E")
                has_decrease = any(es[i + 1] < es[i] for i in range(len(es) - 1))
                verdict = "CONFIRMED" if has_decrease else "REFUTED"
            claims.append({"text": "Retraction (E decreasing) present", "verdict": verdict})

    # 7. Feedrate mention like "F1800" or "high feedrate F1200+"
    if len(claims) < 6:
        m = re.search(r"F\s*(\d{2,5})", desc)
        if m:
            val = float(m.group(1))
            if not fs:
                verdict = "REFUTED"
            else:
                verdict = "CONFIRMED" if any(abs(f - val) <= 300 for f in fs) else "REFUTED"
            claims.append({"text": f"Feedrate ~ F{int(val)}", "verdict": verdict})

    # 8. Layer progression / Z-axis mention
    if len(claims) < 6:
        if re.search(r"\blayer\b|\bZ\b|single layer", desc, re.IGNORECASE):
            has_z = re.search(r"(?<![A-Za-z])Z" + FLOAT, data) is not None
            if re.search(r"Z is absent|single layer|no Z", desc, re.IGNORECASE):
                verdict = "CONFIRMED" if not has_z else "REFUTED"
            else:
                verdict = "UNVERIFIABLE"
            claims.append({"text": "Z/layer state mentioned", "verdict": verdict})

    # 9. Small XY increments mention ("0.01 mm", "tiny", "densely-sampled")
    if len(claims) < 6:
        if re.search(r"tiny|densely|0\.0\d|hundredths|few thousandths|small", desc, re.IGNORECASE):
            # Verify: consecutive X deltas actually small (<0.5mm)
            xs = find_axis_values(data, "X")
            if len(xs) >= 2:
                deltas = [abs(xs[i + 1] - xs[i]) for i in range(len(xs) - 1)]
                small = sum(1 for d in deltas if d < 0.5)
                verdict = "CONFIRMED" if small >= len(deltas) * 0.6 else "REFUTED"
            else:
                verdict = "UNVERIFIABLE"
            claims.append({"text": "Small XY increments (dense sampling)", "verdict": verdict})

    # 10. M-code / accel reconfiguration mention
    if len(claims) < 6:
        if re.search(r"\bM20\d\b|accel|jerk|M204|M205", desc, re.IGNORECASE):
            has_m = re.search(r"M20\d", data) is not None
            verdict = "CONFIRMED" if has_m else "REFUTED"
            claims.append({"text": "M20x (accel/jerk) command present", "verdict": verdict})

    # Ensure at least 3 claims. If extractor yielded <3, pad with default
    # structural claims that are trivially evaluable from the text itself.
    if len(claims) < 3:
        # Default claim: "description mentions extrusion" -> confirmed if G1 in data.
        if not any("extrusion" in c["text"].lower() for c in claims):
            verdict = "CONFIRMED" if has_command(data, "G1") else "REFUTED"
            claims.append({"text": "Extrusion context (fallback)", "verdict": verdict})
        if len(claims) < 3:
            verdict = "UNVERIFIABLE"
            claims.append({"text": "Description produced (non-empty fallback)", "verdict": verdict})
        if len(claims) < 3:
            claims.append({"text": "Padding claim", "verdict": "UNVERIFIABLE"})

    return claims[:6]


def extract_ceiling_claims(data: str) -> list[dict]:
    """Build 3-6 independent claims directly from the data (sequence+continuation).

    These are ground-truth claims — verifying them against the same data should
    give a ~1.0 confirmation rate (minus unverifiable edge cases), establishing
    the upper bound of achievable bridge_score.
    """
    claims: list[dict] = []

    # 1. G1 present
    if has_command(data, "G1"):
        claims.append({"text": "G1 extrusion command present", "verdict": "CONFIRMED"})
    # 2. G0 present
    if has_command(data, "G0"):
        claims.append({"text": "G0 travel command present", "verdict": "CONFIRMED"})
    # 3. X axis range
    xs = find_axis_values(data, "X")
    if xs:
        claims.append({
            "text": f"X values in [{min(xs):.2f}, {max(xs):.2f}]",
            "verdict": "CONFIRMED",
        })
    # 4. Y axis range
    ys = find_axis_values(data, "Y")
    if ys:
        claims.append({
            "text": f"Y values in [{min(ys):.2f}, {max(ys):.2f}]",
            "verdict": "CONFIRMED",
        })
    # 5. E axis range
    es = find_axis_values(data, "E")
    if es:
        claims.append({
            "text": f"E values in [{min(es):.4f}, {max(es):.4f}]",
            "verdict": "CONFIRMED",
        })
    # 6. Feedrate if any
    fs = f_values(data)
    if fs:
        claims.append({
            "text": f"Feedrate F{int(fs[0])} appears",
            "verdict": "CONFIRMED",
        })
    # 7. Whether retraction happens (ground-truth fact)
    if es and any(es[i + 1] < es[i] for i in range(len(es) - 1)):
        claims.append({"text": "Retraction (E decreasing) present", "verdict": "CONFIRMED"})

    # If somehow under 3, pad with the trivial "sequence is non-empty"
    while len(claims) < 3:
        claims.append({"text": "Data stream non-empty", "verdict": "CONFIRMED"})

    return claims[:6]


# ----- main ------------------------------------------------------------------

def score_record(rec: dict) -> dict:
    data = rec["sequence"] + rec["continuation"]
    desc = rec["generated_description"] or ""

    b_claims = extract_bridge_claims(desc, data)
    c_claims = extract_ceiling_claims(data)

    def tally(claims):
        n = len(claims)
        c = sum(1 for x in claims if x["verdict"] == "CONFIRMED")
        r = sum(1 for x in claims if x["verdict"] == "REFUTED")
        u = sum(1 for x in claims if x["verdict"] == "UNVERIFIABLE")
        return n, c, r, u

    n, c, r, u = tally(b_claims)
    n2, c2, _, _ = tally(c_claims)

    return {
        "seq_idx": rec["seq_idx"],
        "n_claims": n,
        "n_confirmed": c,
        "n_refuted": r,
        "n_unverifiable": u,
        "bridge_score": c / n if n else 0.0,
        "ceiling_n_claims": n2,
        "ceiling_n_confirmed": c2,
        "ceiling_score": c2 / n2 if n2 else 0.0,
    }


def main() -> None:
    records = [json.loads(l) for l in IN_PATH.open(encoding="utf-8") if l.strip()]
    assert len(records) == 50, f"expected 50 records, got {len(records)}"

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as out:
        for rec in records:
            score = score_record(rec)
            out.write(json.dumps(score, ensure_ascii=False) + "\n")

    # Quick aggregate for the report
    scores = [json.loads(l) for l in OUT_PATH.open(encoding="utf-8") if l.strip()]
    mean_bridge = sum(s["bridge_score"] for s in scores) / len(scores)
    mean_ceiling = sum(s["ceiling_score"] for s in scores) / len(scores)
    total_claims = sum(s["n_claims"] for s in scores)
    total_conf = sum(s["n_confirmed"] for s in scores)
    print(f"wrote {OUT_PATH} ({len(scores)} records)")
    print(f"  mean bridge_score  = {mean_bridge:.3f}")
    print(f"  mean ceiling_score = {mean_ceiling:.3f}")
    print(f"  efficiency         = {mean_bridge / mean_ceiling:.3f}" if mean_ceiling else "")
    print(f"  total claims       = {total_claims}  confirmed={total_conf}")


if __name__ == "__main__":
    main()
