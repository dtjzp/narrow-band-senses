"""Within-window character entropy computation across PoC domains.

For each domain, compute mean per-window H_1 (first-order character entropy)
on non-overlapping 200-char windows, normalised by domain H_0.

Output: per-domain H_win_normalised + correlation with PASS/FAIL classification
across the 11 PoC domains.

Usage: python compute_window_entropy.py
"""
from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from statistics import mean, stdev

REPO = Path(__file__).resolve().parents[2]
SURVEY_DATA = REPO / "1-research" / "nbs-survey" / "data" / "processed"
EXP2_DATA = REPO / "1-research" / "nbs-experiment-2" / "data"

WINDOW = 200
STRIDE = 200

# PoC domains with bridge outcomes.
# PASS classifications: Stable PASS (Spec 2-verified) or single-seed PASS.
# FAIL classifications tagged by mechanism.
DOMAINS = {
    # PASS — high within-window vocabulary diversity expected
    "quantum":     {"file": SURVEY_DATA / "quantum_1M.txt",       "ss": 0.698, "verdict": "PASS", "mech": None},
    "python_code": {"file": EXP2_DATA   / "python_code_1M.txt",   "ss": 0.523, "verdict": "PASS", "mech": None},
    "gcode":       {"file": EXP2_DATA   / "gcode_1M.txt",         "ss": 0.323, "verdict": "PASS", "mech": None},
    "midi":        {"file": SURVEY_DATA / "midi_1M.txt",          "ss": 0.340, "verdict": "PASS", "mech": None},
    "smiles":      {"file": EXP2_DATA   / "smiles_1M.txt",        "ss": 0.553, "verdict": "PASS-legacy", "mech": None},
    # FAIL — within-window vocabulary repetition hypothesis (attractor collapse)
    "rna":         {"file": SURVEY_DATA / "rna_1M.txt",           "ss": 0.675, "verdict": "FAIL", "mech": "attractor"},
    "bioreactor":  {"file": SURVEY_DATA / "bioreactor_1M.txt",    "ss": 0.931, "verdict": "FAIL", "mech": "attractor"},
    "atc":         {"file": SURVEY_DATA / "atc_1M.txt",           "ss": 0.563, "verdict": "FAIL", "mech": "attractor"},
    # FAIL — low-SS framework prediction
    "network":     {"file": SURVEY_DATA / "network_1M.txt",       "ss": 0.126, "verdict": "FAIL", "mech": "low-SS"},
    "dna_coding":  {"file": EXP2_DATA   / "dna_coding_1M.txt",    "ss": 0.033, "verdict": "FAIL", "mech": "low-SS"},
    # FAIL — compositional hierarchy
    "reactions":   {"file": SURVEY_DATA / "reactions_1M.txt",     "ss": 0.449, "verdict": "FAIL", "mech": "compositional"},
}


def shannon_entropy(counts: dict) -> float:
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return -sum(c / total * math.log2(c / total) for c in counts.values() if c > 0)


def window_entropies(stream: str, window: int, stride: int) -> list[float]:
    """Compute H_1 for each non-overlapping window."""
    hs = []
    n = len(stream)
    for i in range(0, n - window + 1, stride):
        w = stream[i : i + window]
        counts = Counter(w)
        hs.append(shannon_entropy(counts))
    return hs


def spearman_rho(x: list[float], y: list[float]) -> tuple[float, int]:
    """Compute Spearman rank correlation coefficient."""
    n = len(x)
    def rank(values):
        paired = sorted(enumerate(values), key=lambda p: p[1])
        ranks = [0.0] * len(values)
        i = 0
        while i < len(paired):
            j = i
            while j + 1 < len(paired) and paired[j + 1][1] == paired[i][1]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                ranks[paired[k][0]] = avg
            i = j + 1
        return ranks
    rx, ry = rank(x), rank(y)
    mx, my = mean(rx), mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    dy = math.sqrt(sum((b - my) ** 2 for b in ry))
    if dx == 0 or dy == 0:
        return 0.0, n
    return num / (dx * dy), n


def main() -> None:
    rows = []
    for name, cfg in DOMAINS.items():
        path = cfg["file"]
        if not path.exists():
            print(f"[skip] {name}: missing {path}")
            continue
        stream = path.read_text(encoding="utf-8", errors="replace")
        # Compute H0 (first-order over full stream)
        h0_counts = Counter(stream)
        h0 = shannon_entropy(h0_counts)
        alphabet = len(h0_counts)
        # Compute per-window H1
        hs = window_entropies(stream, WINDOW, STRIDE)
        h_win_mean = mean(hs)
        h_win_sd = stdev(hs) if len(hs) > 1 else 0.0
        # Normalise by H0 — gives [0, 1] scale comparing within-window diversity to global
        h_win_norm = h_win_mean / h0 if h0 > 0 else 0.0
        rows.append({
            "domain": name,
            "ss": cfg["ss"],
            "verdict": cfg["verdict"],
            "mech": cfg["mech"],
            "n_chars": len(stream),
            "alphabet": alphabet,
            "h0": h0,
            "h_win_mean": h_win_mean,
            "h_win_sd": h_win_sd,
            "h_win_norm": h_win_norm,
            "n_windows": len(hs),
        })

    # Print table sorted by verdict then H_win
    print(f"{'Domain':<14} {'SS':>6} {'H_0':>6} {'H_win':>7} {'H_win/H_0':>10} {'alpha':>5} {'verdict':<14} mechanism")
    print("-" * 90)
    for r in sorted(rows, key=lambda r: (r["verdict"], -r["h_win_norm"])):
        print(f"{r['domain']:<14} {r['ss']:>6.3f} {r['h0']:>6.3f} {r['h_win_mean']:>7.3f} "
              f"{r['h_win_norm']:>10.3f} {r['alphabet']:>4d} {r['verdict']:<14} {r['mech'] or '—'}")

    # Correlation: h_win_norm vs pass_vs_fail (binary)
    pass_scores = [1 if r["verdict"].startswith("PASS") else 0 for r in rows]
    h_win_scores = [r["h_win_norm"] for r in rows]
    rho, n = spearman_rho(h_win_scores, pass_scores)
    print(f"\nSpearman ρ(H_win/H_0, PASS) = {rho:+.3f}  (n={n})")

    # Separate correlations for within-window vocab FAIL mechanism
    attractor_rows = [r for r in rows if r["mech"] == "attractor" or r["verdict"].startswith("PASS")]
    if len(attractor_rows) >= 3:
        pass2 = [1 if r["verdict"].startswith("PASS") else 0 for r in attractor_rows]
        h2 = [r["h_win_norm"] for r in attractor_rows]
        rho2, n2 = spearman_rho(h2, pass2)
        print(f"Spearman ρ(H_win/H_0, PASS | excluding low-SS + compositional) = {rho2:+.3f}  (n={n2})")

    # Save JSON
    out = REPO / "1-research" / "nbs-bridge" / "window_entropy_results.json"
    out.write_text(json.dumps(rows, indent=2))
    print(f"\nWrote: {out}")


if __name__ == "__main__":
    main()
