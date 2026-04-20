"""Figure 1 — Headline SS vs normalised BPC correlation across 29 domains.

Spec: §Figure 1 of 2026-04-19-nbs-figure-composition-spec.md
Nature main display item. ρ = -0.92 headline with partial-correlation
(ρ_partial = -0.929, 95% CI [-0.980, -0.807]) annotation as subtitle from
Supplementary Note S3b.

Data sources:
  - SS: 1-research/nbs-survey/results/canonical_training_entropy.json
  - Normalised BPC: 1-research/nbs-survey/shared-denominator-defence/bpc_per_domain.json
  - Partial-correlation stats: 1-research/nbs-survey/shared-denominator-defence/stats.json
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

HERE = Path(__file__).parent
REPO = HERE.parent.parent.parent          # speaking-opportunities
CAN = REPO / "1-research/nbs-survey/results/canonical_training_entropy.json"
BPC = REPO / "1-research/nbs-survey/shared-denominator-defence/bpc_per_domain.json"
STATS = REPO / "1-research/nbs-survey/shared-denominator-defence/stats.json"
OUT = HERE / "fig1_ss_bpc_correlation"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial"],
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
})


def main():
    can = json.loads(CAN.read_text())
    bpc = json.loads(BPC.read_text())
    stats = json.loads(STATS.read_text())

    domains = sorted(set(can) & (set(bpc) - {"_source_note"}))
    ss = np.array([can[d]["structure_score"] for d in domains])
    nb = np.array([bpc[d]["norm_bpc"] for d in domains])

    rho, p = spearmanr(ss, nb)

    # Double-column Nature width (183 mm ≈ 7.2 in)
    fig, ax = plt.subplots(figsize=(7.2, 4.5), dpi=200)
    ax.scatter(ss, nb, s=42, c="#1f4e79", edgecolor="white", linewidth=0.6,
               alpha=0.88, zorder=3)

    # Least-squares regression line
    coef = np.polyfit(ss, nb, 1)
    xs = np.linspace(0, 1, 100)
    ax.plot(xs, np.polyval(coef, xs), "-",
            color="#b52525", linewidth=1.2, alpha=0.85, zorder=2,
            label=f"linear fit (slope {coef[0]:+.3f})")

    # Highlight domains that anchor the paper's narrative
    highlights = {
        "bioreactor": ("bioreactor", "right"),
        "whale": ("whale song", "right"),
        "quantum": ("quantum", "right"),
        "smiles": ("SMILES", "right"),
        "python_code": ("Python", "right"),
        "gcode": ("G-code", "left"),
        "english": ("English", "left"),
        "dna_coding": ("DNA coding", "left"),
        "protein": ("protein", "left"),
        "financial": ("financial", "right"),
        "seti": ("SETI", "right"),
    }
    for d, (label, side) in highlights.items():
        if d not in domains:
            continue
        i = domains.index(d)
        dx, ha = (8, "left") if side == "right" else (-8, "right")
        ax.annotate(label, (ss[i], nb[i]), xytext=(dx, 2),
                    textcoords="offset points", fontsize=7,
                    ha=ha, va="center", color="#333")

    ax.set_xlim(-0.04, 1.02)
    ax.set_ylim(0, 1.04)
    ax.set_xlabel("Structure score (SS = 1 − H₃/H₀)")
    ax.set_ylabel("Normalised BPC  (BPC / H₀)")

    rho_p = stats["partial_spearman"]["rho_partial"]
    ci = stats["partial_spearman"]["bootstrap_95ci"]

    # Two-line title with controlled spacing (shortened for one-col safety)
    title = f"ρ(SS, normalised BPC) = {rho:.2f}   (p = {p:.1e},  n = {len(domains)})"
    sub = (f"Partial Spearman | H₀:  ρ_partial = {rho_p:.3f}   "
           f"CI [{ci[0]:.3f}, {ci[1]:.3f}]   "
           f"perm. p < 2×10⁻⁴   (Supp. S3b)")
    ax.set_title(f"{title}\n{sub}", loc="left",
                 fontsize=10, pad=8, linespacing=1.4)
    # Style the subtitle line slightly smaller via a suptitle workaround
    # (matplotlib doesn't allow per-line styling in set_title; we rely on \n)

    ax.grid(alpha=0.25, linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="lower left", fontsize=8)

    fig.tight_layout()
    fig.savefig(f"{OUT}.png", dpi=300)
    fig.savefig(f"{OUT}.pdf")
    print(f"[ok] {OUT}.png / .pdf   |   rho={rho:.4f}  rho_partial={rho_p:.4f}  n={len(domains)}")


if __name__ == "__main__":
    main()
