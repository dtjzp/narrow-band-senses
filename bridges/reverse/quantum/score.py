#!/usr/bin/env python
"""Re-derive bridges/reverse/quantum/scorecard_heldout_temp08.json.

Deterministic, CPU-only, regex-based — no GPU, no pyqasm/qiskit parser, no
randomness. Runs in < 5 s.

What is re-derived vs snapshotted:

    Re-derived from poc_generated_temp08.jsonl
      - total                    (parse_ok, category_match, feature_match)
      - per_category              (n, parse_ok_rate, category_match_rate,
                                   feature_match_rate, unique_rate per category)
      - total_rates

    Snapshotted from sibling JSON files (upstream pipeline artefacts that
    cannot be re-derived from the 148 eval generations alone)
      - roundtrip_rate           ← roundtrip_rate.json   (paired corpus, n=1468)
      - decision_checkpoint      ← decision_checkpoint.json
      - training_meta            ← training_meta.json (subset fields)

The labeller (classify_quantum_window) is a verbatim port of the author's
private overnight-scripts/labellers.py used to build the paired corpus and
to re-label generated sequences. Porting it verbatim is intentional: the
scorecard is a round-trip of the exact same classifier, so the reproduction
match is definitional. Any future scorer change should also update the
corpus labeller.

Usage
-----
    python bridges/reverse/quantum/score.py
    python bridges/reverse/quantum/score.py --out scorecard_replay.json
    python bridges/reverse/quantum/score.py --self-test

Options:
    --generations PATH    default: bridges/reverse/quantum/poc_generated_temp08.jsonl
    --out PATH            default: bridges/reverse/quantum/scorecard_heldout_temp08.replay.json
    --compare             after writing, diff against scorecard_heldout_temp08.json
    --self-test           run internal labeller sanity checks and exit
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent

# ─── Quantum labeller (ported verbatim from author's labellers.py) ───────────

Q_GATES = ["h", "x", "y", "z", "cx", "cz", "ccx", "rx", "ry", "rz", "u3", "measure"]


def classify_quantum_window(text: str) -> dict:
    """Deterministic regex-based QASM-window labeller.

    Counts 12 gate keywords with word-boundary regex, derives gate-class
    totals (n_cx, n_rot, n_meas, n_single), and assigns one of 5 categories
    by priority:

        measurement-heavy   if n_meas  >= 0.2 * total
        highly-entangled    elif n_cx  >= 0.3 * total
        entangling          elif n_cx  > 0
        parameterised       elif n_rot > 0
        single-qubit        else

    Fallback when total == 0: 'single-qubit' if ';' >= 5 else 'parameterised'.
    """
    counts = {}
    for g in Q_GATES:
        counts[g] = len(re.findall(r"(?<![A-Za-z])" + re.escape(g) + r"(?![A-Za-z])", text))
    n_cx = counts.get("cx", 0) + counts.get("cz", 0) + counts.get("ccx", 0)
    n_rot = counts.get("rx", 0) + counts.get("ry", 0) + counts.get("rz", 0) + counts.get("u3", 0)
    n_meas = counts.get("measure", 0)
    n_single = counts.get("h", 0) + counts.get("x", 0) + counts.get("y", 0) + counts.get("z", 0)
    total = n_cx + n_rot + n_meas + n_single

    qubit_idx = re.findall(r"q\[(\d+)\]", text)
    n_qubits = max([int(q) for q in qubit_idx], default=0) + 1 if qubit_idx else 1

    if total == 0:
        n_semicolons = text.count(";")
        category = "single-qubit" if n_semicolons >= 5 else "parameterised"
        return {
            "category": category,
            "n_qubits": n_qubits,
            "n_gates": 0,
            "gate_counts": counts,
            "n_cx": 0, "n_rot": 0, "n_meas": 0, "n_single": 0,
        }

    if n_meas >= 0.2 * total:
        category = "measurement-heavy"
    elif n_cx >= 0.3 * total:
        category = "highly-entangled"
    elif n_cx > 0:
        category = "entangling"
    elif n_rot > 0:
        category = "parameterised"
    else:
        category = "single-qubit"

    return {
        "category": category,
        "n_qubits": n_qubits,
        "n_gates": total,
        "gate_counts": counts,
        "n_cx": n_cx, "n_rot": n_rot, "n_meas": n_meas, "n_single": n_single,
    }


# ─── Scorer ──────────────────────────────────────────────────────────────────

def score_generations(records: list) -> dict:
    """Compute total / per_category / total_rates from generated-sequence records.

    Each record must have keys 'category' (expected label) and
    'generated_sequence' (the LM output being scored).
    """
    per_cat = defaultdict(
        lambda: {"n": 0, "parse_ok": 0, "category_match": 0, "feature_match": 0, "unique": set()}
    )
    total = {"n": 0, "parse_ok": 0, "category_match": 0, "feature_match": 0}

    for r in records:
        cat = r["category"]
        per_cat[cat]["n"] += 1
        total["n"] += 1

        lbl = classify_quantum_window(r["generated_sequence"])

        if lbl.get("category") not in (None, "invalid"):
            per_cat[cat]["parse_ok"] += 1
            total["parse_ok"] += 1

        if lbl.get("category") == cat:
            per_cat[cat]["category_match"] += 1
            total["category_match"] += 1

        # Feature match (quantum-specific): CX-gate presence must agree
        # with expected entangling category membership.
        feat_ok = (lbl.get("n_cx", 0) > 0) == (cat in ("entangling", "highly-entangled"))
        if feat_ok:
            per_cat[cat]["feature_match"] += 1
            total["feature_match"] += 1

        per_cat[cat]["unique"].add(r["generated_sequence"])

    per_cat_out = {}
    for cat, d in per_cat.items():
        n = d["n"]
        per_cat_out[cat] = {
            "n": n,
            "parse_ok_rate": d["parse_ok"] / max(n, 1),
            "category_match_rate": d["category_match"] / max(n, 1),
            "feature_match_rate": d["feature_match"] / max(n, 1),
            "unique_rate": len(d["unique"]) / max(n, 1),
        }

    return {
        "total": total,
        "per_category": per_cat_out,
        "total_rates": {
            "parse_ok": total["parse_ok"] / max(total["n"], 1),
            "category_match": total["category_match"] / max(total["n"], 1),
            "feature_match": total["feature_match"] / max(total["n"], 1),
        },
    }


def build_scorecard(generations_path: Path, domain_dir: Path) -> dict:
    records = [json.loads(line) for line in generations_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    scorecard = score_generations(records)

    roundtrip_path = domain_dir / "roundtrip_rate.json"
    if roundtrip_path.exists():
        rt = json.loads(roundtrip_path.read_text(encoding="utf-8"))
        scorecard["roundtrip_rate"] = rt["rate"]

    dc_path = domain_dir / "decision_checkpoint.json"
    if dc_path.exists():
        scorecard["decision_checkpoint"] = json.loads(dc_path.read_text(encoding="utf-8"))

    tm_path = domain_dir / "training_meta.json"
    if tm_path.exists():
        tm_full = json.loads(tm_path.read_text(encoding="utf-8"))
        scorecard["training_meta"] = {
            "best_val_loss": tm_full.get("best_val_loss"),
            "best_epoch": tm_full.get("best_epoch"),
            "n_train": tm_full.get("n_train"),
            "n_val": tm_full.get("n_val"),
            "n_eval": tm_full.get("n_eval"),
        }

    return scorecard


# ─── Comparison helper ───────────────────────────────────────────────────────

def _deep_close(a, b, tol=1e-10):
    """Recursive near-equality: numeric fields within tol, others exact."""
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a.keys()) != set(b.keys()):
            return False, f"keys differ: {sorted(a)} vs {sorted(b)}"
        for k in a:
            ok, msg = _deep_close(a[k], b[k], tol)
            if not ok:
                return False, f".{k}: {msg}"
        return True, ""
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return False, f"length {len(a)} vs {len(b)}"
        for i, (x, y) in enumerate(zip(a, b)):
            ok, msg = _deep_close(x, y, tol)
            if not ok:
                return False, f"[{i}]: {msg}"
        return True, ""
    if isinstance(a, float) or isinstance(b, float):
        if a is None or b is None:
            return a == b, f"{a!r} vs {b!r}"
        return abs(float(a) - float(b)) <= tol, f"{a} vs {b} (|Δ|={abs(a-b)})"
    return a == b, f"{a!r} vs {b!r}"


def compare_scorecards(ours: dict, shipped: dict, tol: float = 1e-10) -> tuple[bool, list[str]]:
    """Return (all_match, list_of_diffs). Only diffs if non-trivial."""
    diffs = []
    for key in sorted(set(ours) | set(shipped)):
        if key not in ours:
            diffs.append(f"missing in replay: {key}")
            continue
        if key not in shipped:
            diffs.append(f"extra in replay: {key}")
            continue
        ok, msg = _deep_close(ours[key], shipped[key], tol)
        if not ok:
            diffs.append(f"{key}: {msg}")
    return (len(diffs) == 0, diffs)


# ─── Self-test ───────────────────────────────────────────────────────────────

def _self_test() -> None:
    """Minimal hand-crafted labeller sanity checks."""
    # Single-qubit: three H gates, no rotations/CX/measure
    r = classify_quantum_window("h q[0]; x q[0]; y q[0];")
    assert r["category"] == "single-qubit", r["category"]
    assert r["n_cx"] == 0 and r["n_rot"] == 0 and r["n_meas"] == 0
    assert r["n_single"] == 3

    # Entangling: one CX only (1/2 = 0.5, which >= 0.3 → highly-entangled)
    r = classify_quantum_window("cx q[0],q[1]; h q[0];")
    assert r["category"] == "highly-entangled", r["category"]

    # Entangling (below 30%): one CX with lots of single-qubit gates
    r = classify_quantum_window("cx q[0],q[1]; h q[0]; h q[1]; x q[0]; y q[0];")
    # total = 1 + 0 + 0 + 4 = 5; n_cx/total = 0.2 < 0.3 → entangling
    assert r["category"] == "entangling", r["category"]

    # Parameterised: rotations only
    r = classify_quantum_window("rx(0.5) q[0]; ry(0.3) q[0]; rz(0.1) q[0];")
    assert r["category"] == "parameterised", r["category"]
    assert r["n_rot"] == 3

    # Measurement-heavy: ≥20% measurements
    r = classify_quantum_window("measure q[0] -> c[0]; measure q[1] -> c[1]; h q[0]; x q[0];")
    # total = 0 + 0 + 2 + 2 = 4; n_meas/total = 0.5 ≥ 0.2 → measurement-heavy
    assert r["category"] == "measurement-heavy", r["category"]

    # Fallback (total == 0): many semicolons → single-qubit
    r = classify_quantum_window(";;;;;;;")
    assert r["category"] == "single-qubit", r["category"]

    # Scoring sanity: 2 records, one correct, one wrong
    recs = [
        {"category": "single-qubit", "generated_sequence": "h q[0]; x q[0]; y q[0];"},
        {"category": "entangling", "generated_sequence": "h q[0]; x q[0]; y q[0];"},  # mislabelled gen
    ]
    sc = score_generations(recs)
    assert sc["total"] == {"n": 2, "parse_ok": 2, "category_match": 1, "feature_match": 1}, sc["total"]
    assert abs(sc["total_rates"]["parse_ok"] - 1.0) < 1e-12
    assert abs(sc["total_rates"]["category_match"] - 0.5) < 1e-12

    print("self-test OK")


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--generations", type=Path, default=HERE / "poc_generated_temp08.jsonl")
    ap.add_argument("--out", type=Path, default=HERE / "scorecard_heldout_temp08.replay.json")
    ap.add_argument("--compare", action="store_true",
                    help="Diff the replay against the shipped scorecard_heldout_temp08.json")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--tolerance", type=float, default=1e-10,
                    help="Numeric tolerance for --compare")
    args = ap.parse_args()

    if args.self_test:
        _self_test()
        return 0

    if not args.generations.exists():
        print(f"error: generations file not found: {args.generations}", file=sys.stderr)
        return 2

    scorecard = build_scorecard(args.generations, HERE)
    args.out.write_text(json.dumps(scorecard, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")
    print(f"  total_rates: {scorecard['total_rates']}")
    print(f"  categories : {list(scorecard['per_category'].keys())}")

    if args.compare:
        shipped_path = HERE / "scorecard_heldout_temp08.json"
        if not shipped_path.exists():
            print(f"warning: shipped scorecard not found for --compare: {shipped_path}", file=sys.stderr)
            return 0
        shipped = json.loads(shipped_path.read_text(encoding="utf-8"))
        ok, diffs = compare_scorecards(scorecard, shipped, tol=args.tolerance)
        if ok:
            print(f"  compare   : MATCH (tolerance {args.tolerance})")
            return 0
        print(f"  compare   : MISMATCH ({len(diffs)} diff{'s' if len(diffs) != 1 else ''})")
        for d in diffs[:20]:
            print(f"    - {d}")
        if len(diffs) > 20:
            print(f"    ... ({len(diffs) - 20} more)")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
