"""Figure 5 — SS × TRL playbook visualisation.

Spec: §Figure 5 of 2026-04-19-nbs-figure-composition-spec.md.
Table 4.5 rendered as figure. Three SS tiers on y-axis, baseline and
full-stack TRL as horizontal bars, primary-lever annotation, and scale-
rescue indicators per tier.

Data: Table 4.5 content (hard-coded from the paper's §4.5).
"""
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

HERE = Path(__file__).parent
OUT = HERE / "fig5_ss_trl_playbook"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial"],
    "font.size": 9,
    "axes.labelsize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 9,
    "axes.titlesize": 11,
})

# Table 4.5 content
TIERS = [
    {
        "name": "High (SS > 0.5)",
        "ss_range": (0.5, 1.0),
        "baseline_trl": (3, 4),
        "fullstack_trl": (4, 5),
        "fullstack_qualifier": "conditional on H_win/H₀ ≥ 0.80",
        "examples": "whale, tidal, bioreactor, SMILES, Python, Quantum, RNA, ATC, reactions",
        "evidence": "PASS: Quantum, Python, SMILES-legacy   FAIL: RNA, bioreactor, ATC, reactions",
        "primary_lever": "Factor D + within-window vocabulary check (H_win/H₀)",
        "scale_rescue": "partial",  # compositional-hierarchy branch rescuable via Factor-B
        "scale_note": "Factor-B scaling plausibly rescues compositional-hierarchy failures (§5.4 Prediction 11, untested)",
        "color_baseline": "#a8c3db",
        "color_fullstack": "#1f4e79",
    },
    {
        "name": "Mid (0.1 < SS < 0.5)",
        "ss_range": (0.1, 0.5),
        "baseline_trl": (3, 3),
        "fullstack_trl": (4, 5),
        "fullstack_qualifier": "modality-specific variation",
        "examples": "English, Greek, G-code, MIDI, weather, network, DAS seismic",
        "evidence": "PASS: G-code (Tests A+B)   S/XL split: MIDI   FAIL: network (lower boundary)",
        "primary_lever": "Factors B + D jointly; H_win/H₀ check still applies",
        "scale_rescue": "partial",
        "scale_note": "Modality-dependent; G-code numeric precision gap remains (§S19)",
        "color_baseline": "#c3d9b6",
        "color_fullstack": "#4a7a2b",
    },
    {
        "name": "Low (SS < 0.15)",
        "ss_range": (0.0, 0.15),
        "baseline_trl": (2, 2),
        "fullstack_trl": (2, 3),
        "fullstack_qualifier": "detection-utility only",
        "examples": "DNA coding, financial, SETI, DNA non-coding, protein, network",
        "evidence": "FAIL (generative): Network, DNA coding   Reframe as: anomaly detection via likelihood ratio",
        "primary_lever": "Do not attempt generative reverse-bridging; reframe as detection",
        "scale_rescue": "none",
        "scale_note": "Factor-B scaling predicted NOT to rescue (signal intrinsics, not capacity)",
        "color_baseline": "#e0c0c0",
        "color_fullstack": "#a03030",
    },
]


def main():
    fig, ax = plt.subplots(figsize=(9.5, 5.6), dpi=200)

    # Y positions: top=High, bottom=Low
    y_positions = [2, 1, 0]  # reversed so High at top
    tier_height = 0.42

    for y, tier in zip(y_positions, TIERS):
        b_lo, b_hi = tier["baseline_trl"]
        f_lo, f_hi = tier["fullstack_trl"]
        # Baseline bar (bottom half of the tier band)
        ax.barh(y - tier_height / 4, b_hi - b_lo, left=b_lo,
                height=tier_height / 2, color=tier["color_baseline"],
                edgecolor="white", linewidth=0.6, zorder=3,
                label="Baseline TRL" if y == 2 else None)
        ax.text(b_lo + (b_hi - b_lo) / 2 if b_hi > b_lo else b_lo, y - tier_height / 4,
                f"TRL {b_lo}–{b_hi}" if b_hi > b_lo else f"TRL {b_lo}",
                ha="center", va="center", fontsize=8, color="#333", zorder=4)

        # Full-stack bar (top half)
        ax.barh(y + tier_height / 4, f_hi - f_lo if f_hi > f_lo else 0.4, left=f_lo,
                height=tier_height / 2, color=tier["color_fullstack"],
                edgecolor="white", linewidth=0.6, zorder=3,
                label="Full-stack TRL" if y == 2 else None)
        ax.text(f_lo + (max(f_hi - f_lo, 0.4)) / 2, y + tier_height / 4,
                f"TRL {f_lo}–{f_hi}" if f_hi > f_lo else f"TRL {f_lo}",
                ha="center", va="center", fontsize=8, color="white",
                fontweight="bold", zorder=4)

    # Tier name labels (y-axis) — shortened to avoid truncation
    tier_names = [
        "High\nSS > 0.5",
        "Mid\n0.1 < SS < 0.5",
        "Low\nSS < 0.15",
    ]
    ax.set_yticks(y_positions)
    ax.set_yticklabels(tier_names, fontsize=9, fontweight="bold")

    # X-axis (TRL 1-9 scale, but we only go to 6)
    ax.set_xlim(0.5, 6.5)
    ax.set_xticks(range(1, 7))
    ax.set_xlabel("Technology Readiness Level  (1 = proof of concept, 9 = deployed)")

    # Tier row backgrounds for readability
    for y, tier in zip(y_positions, TIERS):
        ax.axhspan(y - tier_height / 2 - 0.05, y + tier_height / 2 + 0.05,
                   color="#f5f5f5", zorder=0)

    # Primary-lever annotation below each row's bars
    for y, tier in zip(y_positions, TIERS):
        ax.text(0.6, y - tier_height / 2 - 0.04,
                f"Lever: {tier['primary_lever']}",
                fontsize=7.3, color="#444", style="italic", va="top", zorder=5)

    # Scale-rescue chip on the right side of each row
    rescue_colors = {"partial": "#ff9800", "full": "#2e7d32", "none": "#666666"}
    rescue_labels = {"partial": "Partial scale rescue",
                     "full": "Scale rescue likely",
                     "none": "Scale NOT rescuable"}
    for y, tier in zip(y_positions, TIERS):
        col = rescue_colors[tier["scale_rescue"]]
        label = rescue_labels[tier["scale_rescue"]]
        ax.text(6.4, y + tier_height / 4, label,
                fontsize=7, color=col, ha="right", va="center",
                fontweight="bold", zorder=6,
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                          edgecolor=col, linewidth=1.0))
        # Examples row below bars
        ex = tier['examples']
        if len(ex) > 68:
            ex = ex[:66] + "…"
        ax.text(6.4, y - tier_height / 4, f"Ex: {ex}",
                fontsize=6.8, color="#333", ha="right", va="center", zorder=5)

    # Two-line title
    ax.set_title(
        "SS × TRL playbook:  achievable capability tier vs domain structure\n"
        "Two-metric screening (SS + H_win/H₀) determines tier + failure class before training.",
        loc="left", pad=10, fontsize=10, linespacing=1.4,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", alpha=0.25, linewidth=0.5, zorder=0)
    ax.set_ylim(-0.6, 2.6)

    # Legend top-right (inside axes)
    legend_elems = [
        mpatches.Patch(facecolor="#888", alpha=0.4, label="Baseline TRL  (S-size + GPT-2 small)"),
        mpatches.Patch(facecolor="#333", alpha=0.85, label="Full-stack TRL  (tested recipe)"),
    ]
    ax.legend(handles=legend_elems, loc="lower right", frameon=True,
              fontsize=8, framealpha=0.95, edgecolor="#ccc",
              bbox_to_anchor=(1.0, 0.015))

    fig.tight_layout()
    fig.savefig(f"{OUT}.png", dpi=300)
    fig.savefig(f"{OUT}.pdf")
    print(f"[ok] {OUT}.png / .pdf")


if __name__ == "__main__":
    main()
