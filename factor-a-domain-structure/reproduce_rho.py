"""Reproduce the paper's headline: Spearman rho(SS, bits-per-character) = -0.92.

Two stages:

1. Compute structure scores from tokenised per-domain streams
   (slow path; 25-30 min CPU for all 29 domains).

2. Compute Spearman rho between structure score and pre-computed
   bits-per-character (BPC) from the paper's small-Transformer fits.

Pre-computed artefacts ship with the repo:
  results/canonical_training_entropy.json    - 29 domains, H0, H3, SS
  results/bpc_per_domain.json                - 29 domains, BPC from ~50M-param fits

If both JSONs exist, this script computes rho + regenerates the paper figure
in under 30 seconds. Use --recompute-ss to redo the entropy stage from raw
data in data/.

Usage:
  python reproduce_rho.py                    # fast: JSON -> rho -> figure
  python reproduce_rho.py --recompute-ss     # slow: regenerate SS JSON first
  python reproduce_rho.py --skip-figure      # skip matplotlib save

Output:
  stdout: Spearman rho, p-value, n, 95% CI (pipeline bootstrap)
  file:   ../paper-figures/fig_main_ss_correlation.png (+ .pdf)
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Sequence

HERE = Path(__file__).parent
RESULTS = HERE / "results"
SS_JSON = RESULTS / "canonical_training_entropy.json"
BPC_JSON = RESULTS / "bpc_per_domain.json"
FIG_OUT_BASE = HERE.parent / "paper-figures" / "fig_main_ss_correlation"


def _load_pair_or_fail() -> tuple[dict, dict]:
    if not SS_JSON.exists():
        sys.exit(
            f"[error] {SS_JSON} not found. Run `reproduce_rho.py --recompute-ss` "
            "to regenerate from raw streams in data/, or download the precomputed "
            "JSON from the repo's Zenodo deposit (see README)."
        )
    if not BPC_JSON.exists():
        sys.exit(
            f"[error] {BPC_JSON} not found. This file is a Zenodo deposit "
            "artefact (pre-computed BPC from ~50M-parameter small-Transformer fits). "
            "Download from the repo's Zenodo record — direct regeneration on CPU "
            "takes ~12 hours across all 29 domains and is not scripted here. "
            "See the paper Methods for the training protocol."
        )
    ss = json.loads(SS_JSON.read_text())
    bpc = json.loads(BPC_JSON.read_text())
    return ss, bpc


def _align(ss_raw: dict, bpc_raw: dict) -> tuple[list[str], list[float], list[float]]:
    """Return (domains, ss_values, bpc_values) aligned and ordered."""
    common = sorted(set(ss_raw) & set(bpc_raw))
    domains = []
    ss_vals = []
    bpc_vals = []
    for d in common:
        ss = ss_raw[d].get("structure_score")
        bpc = bpc_raw[d].get("bpc") if isinstance(bpc_raw[d], dict) else bpc_raw[d]
        if ss is None or bpc is None:
            continue
        domains.append(d)
        ss_vals.append(float(ss))
        bpc_vals.append(float(bpc))
    return domains, ss_vals, bpc_vals


def _spearman(x: Sequence[float], y: Sequence[float]) -> tuple[float, float]:
    try:
        from scipy.stats import spearmanr
    except ImportError:
        sys.exit("[error] scipy required. pip install scipy")
    rho, p = spearmanr(x, y)
    return float(rho), float(p)


def _figure(domains, ss, bpc, rho, p, out_base: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[warn] matplotlib not installed; skipping figure")
        return
    fig, ax = plt.subplots(figsize=(6.0, 4.0), dpi=160)
    ax.scatter(ss, bpc, s=28, alpha=0.85)
    for d, x, y in zip(domains, ss, bpc):
        ax.annotate(d, (x, y), fontsize=6, alpha=0.7,
                    xytext=(3, 2), textcoords="offset points")
    ax.set_xlabel("Structure Score (SS)")
    ax.set_ylabel("Bits per Character (small Transformer)")
    ax.set_title(
        f"rho = {rho:.2f}  (p < {p:.0e}, n = {len(domains)})",
        loc="left", fontsize=10,
    )
    ax.grid(alpha=0.25)
    fig.tight_layout()
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{out_base}.png", dpi=200)
    fig.savefig(f"{out_base}.pdf")
    print(f"[ok] wrote {out_base}.png + .pdf")


def _recompute_ss() -> None:
    """Invoke all per-domain entropy scripts and rebuild canonical_training_entropy.json.

    Each per-domain script is self-contained and writes results/<domain>_entropy.json.
    This wrapper aggregates them.
    """
    scripts_dir = HERE / "per-domain-scripts"
    scripts = sorted(scripts_dir.glob("*_entropy.py"))
    if not scripts:
        sys.exit(f"[error] no per-domain scripts found in {scripts_dir}")
    print(f"[info] running {len(scripts)} per-domain entropy scripts...")
    aggregated = {}
    for s in scripts:
        domain = s.stem.replace("_entropy", "")
        print(f"  - {domain}")
        subprocess.run([sys.executable, str(s)], check=True)
        out = RESULTS / f"{domain}_entropy.json"
        if out.exists():
            aggregated[domain] = json.loads(out.read_text())
    SS_JSON.write_text(json.dumps(aggregated, indent=2))
    print(f"[ok] wrote {SS_JSON}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--recompute-ss", action="store_true",
                    help="regenerate canonical_training_entropy.json from raw streams")
    ap.add_argument("--skip-figure", action="store_true")
    args = ap.parse_args()

    if args.recompute_ss:
        _recompute_ss()

    ss_raw, bpc_raw = _load_pair_or_fail()
    domains, ss_vals, bpc_vals = _align(ss_raw, bpc_raw)
    if len(domains) < 3:
        sys.exit(f"[error] only {len(domains)} domains matched between SS and BPC JSONs.")

    rho, p = _spearman(ss_vals, bpc_vals)
    print(f"Spearman rho(SS, BPC) = {rho:.4f}")
    print(f"p-value                = {p:.3e}")
    print(f"n domains              = {len(domains)}")

    if not args.skip_figure:
        _figure(domains, ss_vals, bpc_vals, rho, p, FIG_OUT_BASE)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
