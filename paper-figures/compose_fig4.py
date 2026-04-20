"""Figure 4 — Three-branch failure typology in (SS, H_win/H₀) space.

Spec: §Figure 4 of 2026-04-19-nbs-figure-composition-spec.md.
2D scatter of all PoCs on SS (x) × H_win/H₀ (y), VERDICT colour-coded and
failure-mechanism shape-coded. Regions for the three-branch typology shaded.

Data source: 1-research/nbs-bridge/window_entropy_results.json (all 11
domains — 8 original + bioreactor v1 + ATC + reactions v2, plus gcode via
its distinct Tests A+B capability path).
"""
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
from matplotlib.lines import Line2D

HERE = Path(__file__).parent
REPO = HERE.parent.parent.parent
WINDOW = REPO / "1-research/nbs-bridge/window_entropy_results.json"
OUT = HERE / "fig4_three_branch_typology"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial"],
    "font.size": 9,
    "axes.labelsize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.titlesize": 10.5,
})

MARKER = {
    "PASS":        ("o", "#2e7d32", "PASS"),
    "PASS-legacy": ("s", "#7b7b7b", "Legacy (pre-gate)"),
    "FAIL/attractor":     ("X", "#c62828", "FAIL — within-window attractor"),
    "FAIL/compositional": ("P", "#ef6c00", "FAIL — compositional hierarchy"),
    "FAIL/low-SS":        ("v", "#6d4c41", "FAIL — low-SS (info-theoretic)"),
}


def verdict_key(r):
    v = r["verdict"]; m = r.get("mech")
    if v == "PASS":
        return "PASS"
    if v == "PASS-legacy":
        return "PASS-legacy"
    if v == "FAIL" and m in ("attractor", "compositional", "low-SS"):
        return f"FAIL/{m}"
    return "PASS"  # fallback


def main():
    data = json.loads(WINDOW.read_text())

    fig, ax = plt.subplots(figsize=(7.2, 5.2), dpi=200)

    # Three-branch region shading
    # Branch 1 (low-SS): SS < 0.15
    ax.add_patch(Rectangle((0, 0), 0.15, 1.0, facecolor="#6d4c41", alpha=0.07, zorder=0))
    # Branch 2 (within-window attractor failure): H_win/H0 < 0.80
    ax.add_patch(Rectangle((0.15, 0), 0.85, 0.80, facecolor="#c62828", alpha=0.06, zorder=0))
    # PASS region: SS > 0.15 AND H_win/H0 > 0.80
    ax.add_patch(Rectangle((0.15, 0.80), 0.85, 0.20, facecolor="#2e7d32", alpha=0.08, zorder=0))

    # Region labels (positioned to avoid data points)
    ax.text(0.075, 0.55, "Branch 1\nLow-SS\ninfo-theoretic",
            ha="center", va="center", fontsize=8, color="#6d4c41", style="italic", alpha=0.85)
    ax.text(0.50, 0.45, "Branch 2\nWithin-window attractor collapse\n(H_win/H₀ < 0.80)",
            ha="center", va="center", fontsize=8.5, color="#c62828", style="italic", alpha=0.85)
    ax.text(0.58, 0.96, "PASS region   (SS ≥ 0.15  AND  H_win/H₀ ≥ 0.80)",
            ha="center", va="center", fontsize=8.5, color="#2e7d32",
            style="italic", alpha=0.95, fontweight="bold")

    # Plot points
    for r in data:
        key = verdict_key(r)
        marker, color, _ = MARKER[key]
        ax.scatter(r["ss"], r["h_win_norm"], marker=marker, s=110, c=color,
                   edgecolor="white", linewidth=1.0, zorder=4)

        # Label offset — avoid overlap with markers
        x, y = r["ss"], r["h_win_norm"]
        dx, dy, ha = 0.012, 0.008, "left"
        # Manual adjustments for crowded regions
        adj = {
            "bioreactor":  (-0.015, 0.035, "right"),
            "python_code": (0.015, -0.003, "left"),
            "dna_coding":  (0.015, 0.000, "left"),
            "network":     (0.015, 0.008, "left"),
            "atc":         (0.015, -0.020, "left"),
            "reactions":   (0.015, 0.010, "left"),
            "rna":         (0.025, 0.010, "left"),
            "quantum":     (-0.015, -0.005, "right"),
            "gcode":       (0.015, -0.012, "left"),
            "midi":        (-0.015, 0.020, "right"),
            "smiles":      (-0.015, -0.020, "right"),
        }
        dx, dy, ha = adj.get(r["domain"], (dx, dy, ha))
        label = r["domain"].replace("_", " ")
        ax.annotate(label, (x, y), xytext=(x + dx, y + dy),
                    fontsize=7.2, ha=ha, va="center", color="#333",
                    zorder=5)

    # Framework-predictor line for Branch 2: H_win/H0 = 0.80
    ax.axhline(0.80, linestyle="--", color="#c62828", linewidth=0.8, alpha=0.5, zorder=1)
    # Framework-predictor line for Branch 1: SS = 0.15
    ax.axvline(0.15, linestyle="--", color="#6d4c41", linewidth=0.8, alpha=0.5, zorder=1)

    ax.set_xlim(0, 1.0)
    ax.set_ylim(0.18, 1.02)
    ax.set_xlabel("Structure score  (SS = 1 − H₃/H₀)")
    ax.set_ylabel("Within-window vocabulary diversity  (H_win / H₀)")

    # Two-line title (main + subtitle shorter to avoid truncation)
    ax.set_title(
        "Three-branch reverse-bridge failure typology  (n = 11 PoCs)\n"
        "Two-metric screening: Branch 3 (compositional) is distinguishable mechanistically (§4.3).",
        loc="left", pad=10, fontsize=10, linespacing=1.4,
    )

    # Custom legend
    legend_elems = []
    for key in ["PASS", "PASS-legacy", "FAIL/attractor", "FAIL/compositional", "FAIL/low-SS"]:
        m, c, label = MARKER[key]
        legend_elems.append(Line2D([0], [0], marker=m, color="w", markerfacecolor=c,
                                    markeredgecolor="white", markersize=10,
                                    label=label, linestyle=""))
    ax.legend(handles=legend_elems, loc="lower left", frameon=True, fontsize=8,
              framealpha=0.95, edgecolor="#ccc", bbox_to_anchor=(0.22, 0.0))

    ax.grid(alpha=0.15, linewidth=0.5, zorder=1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(f"{OUT}.png", dpi=300)
    fig.savefig(f"{OUT}.pdf")
    print(f"[ok] {OUT}.png / .pdf  |  n={len(data)} domains")


if __name__ == "__main__":
    main()
