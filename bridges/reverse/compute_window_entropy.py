"""Within-window character entropy computation across PoC domains.

For each domain, compute mean per-window H_1 (first-order character entropy)
on non-overlapping 200-char windows, normalised by domain H_0.

Output: per-domain H_win_normalised + correlation with PASS/FAIL classification
across the 11 PoC domains.

Usage:
    python compute_window_entropy.py                  # uses ./data by default
    python compute_window_entropy.py --data-dir PATH  # override
    python compute_window_entropy.py --check-only     # verify against committed results

The raw per-domain *_1M.txt streams are not shipped in the public repo
(they live in the author's Drive workspace pending Zenodo deposit). The
pre-computed result (window_entropy_results.json, alongside this script)
is the canonical artefact for reviewers; --check-only prints it without
needing the raw data.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from statistics import mean, stdev

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_JSON = SCRIPT_DIR / "window_entropy_results.json"

WINDOW = 200
STRIDE = 200

# PoC domains with bridge outcomes.
# PASS classifications: Stable PASS (Spec 2-verified) or single-seed PASS.
# FAIL classifications tagged by mechanism.
#
# File paths are relative to --data-dir (default: ./data/). When the raw
# streams aren't available on the local machine, --check-only prints the
# pre-computed results.
DOMAIN_CONFIG = {
    "quantum":     {"file": "quantum_1M.txt",       "ss": 0.698, "verdict": "PASS",        "mech": None},
    "python_code": {"file": "python_code_1M.txt",   "ss": 0.523, "verdict": "PASS",        "mech": None},
    "gcode":       {"file": "gcode_1M.txt",         "ss": 0.323, "verdict": "PASS",        "mech": None},
    "midi":        {"file": "midi_1M.txt",          "ss": 0.340, "verdict": "PASS",        "mech": None},
    "smiles":      {"file": "smiles_1M.txt",        "ss": 0.553, "verdict": "PASS-legacy", "mech": None},
    "rna":         {"file": "rna_1M.txt",           "ss": 0.675, "verdict": "FAIL",        "mech": "attractor"},
    "bioreactor":  {"file": "bioreactor_1M.txt",    "ss": 0.931, "verdict": "FAIL",        "mech": "attractor"},
    "atc":         {"file": "atc_1M.txt",           "ss": 0.563, "verdict": "FAIL",        "mech": "attractor"},
    "network":     {"file": "network_1M.txt",       "ss": 0.126, "verdict": "FAIL",        "mech": "low-SS"},
    "dna_coding":  {"file": "dna_coding_1M.txt",    "ss": 0.033, "verdict": "FAIL",        "mech": "low-SS"},
    "reactions":   {"file": "reactions_1M.txt",     "ss": 0.449, "verdict": "FAIL",        "mech": "compositional"},
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


def print_rows(rows: list[dict]) -> None:
    """Print the H_win/H_0 table and correlation."""
    print(f"{'Domain':<14} {'SS':>6} {'H_0':>6} {'H_win':>7} {'H_win/H_0':>10} {'alpha':>5} {'verdict':<14} mechanism")
    print("-" * 90)
    for r in sorted(rows, key=lambda r: (r["verdict"], -r["h_win_norm"])):
        print(f"{r['domain']:<14} {r['ss']:>6.3f} {r['h0']:>6.3f} {r['h_win_mean']:>7.3f} "
              f"{r['h_win_norm']:>10.3f} {r['alphabet']:>4d} {r['verdict']:<14} {r['mech'] or '—'}")

    pass_scores = [1 if r["verdict"].startswith("PASS") else 0 for r in rows]
    h_win_scores = [r["h_win_norm"] for r in rows]
    rho, n = spearman_rho(h_win_scores, pass_scores)
    print(f"\nSpearman ρ(H_win/H_0, PASS) = {rho:+.3f}  (n={n})")

    attractor_rows = [r for r in rows if r["mech"] == "attractor" or r["verdict"].startswith("PASS")]
    if len(attractor_rows) >= 3:
        pass2 = [1 if r["verdict"].startswith("PASS") else 0 for r in attractor_rows]
        h2 = [r["h_win_norm"] for r in attractor_rows]
        rho2, n2 = spearman_rho(h2, pass2)
        print(f"Spearman ρ(H_win/H_0, PASS | excluding low-SS + compositional) = {rho2:+.3f}  (n={n2})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=SCRIPT_DIR / "data",
                        help="Directory containing per-domain {name}_1M.txt streams. "
                             "Default: ./data/ next to this script.")
    parser.add_argument("--check-only", action="store_true",
                        help="Load and print the committed window_entropy_results.json "
                             "without recomputing. Useful when the raw streams aren't available.")
    parser.add_argument("--out", type=Path, default=RESULTS_JSON,
                        help=f"Output JSON path. Default: {RESULTS_JSON}")
    args = parser.parse_args()

    if args.check_only:
        if not RESULTS_JSON.exists():
            print(f"[error] {RESULTS_JSON} not found — cannot --check-only.")
            return
        rows = json.loads(RESULTS_JSON.read_text(encoding="utf-8"))
        print(f"Loaded pre-computed results from {RESULTS_JSON.name}\n")
        print_rows(rows)
        return

    rows = []
    missing = []
    for name, cfg in DOMAIN_CONFIG.items():
        path = args.data_dir / cfg["file"]
        if not path.exists():
            missing.append((name, path))
            continue
        stream = path.read_text(encoding="utf-8", errors="replace")
        h0_counts = Counter(stream)
        h0 = shannon_entropy(h0_counts)
        alphabet = len(h0_counts)
        hs = window_entropies(stream, WINDOW, STRIDE)
        h_win_mean = mean(hs)
        h_win_sd = stdev(hs) if len(hs) > 1 else 0.0
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

    if missing:
        print(f"[warn] {len(missing)}/{len(DOMAIN_CONFIG)} domain streams not found under {args.data_dir}:")
        for name, path in missing:
            print(f"       {name}: {path}")
        print("Run with --check-only to view the committed pre-computed results, "
              "or point --data-dir at the directory containing {name}_1M.txt files.\n")

    if not rows:
        return

    print_rows(rows)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, indent=2))
    print(f"\nWrote: {args.out}")


if __name__ == "__main__":
    main()
