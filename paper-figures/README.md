# Paper Figures — Indexed by Section

Cross-cutting figures used in the paper's main text. Per-domain reverse-bridge PoC figures (scorecards, diagnostics, feature heatmaps) live in the corresponding `../bridges/reverse/{domain}/figures/` directory.

All figures: CC-BY 4.0. Raster at 200 dpi; vector PDF alongside.

## Paper-caption → repo-filename mapping

The paper's current captions use slightly different filenames than the repo uses internally. Use this mapping if you're reading the paper PDF and looking for a specific figure file here:

| Paper caption name | Repo filename(s) | Compose script |
|---|---|---|
| `figure1_entropy_compressibility.{pdf,png}` | shipped under this name; `fig_main_ss_correlation.{pdf,png}` is the regenerated version from `reproduce_rho.py` | `compose_fig1.py` (+ `reproduce_rho.py`) |
| `fig2_forward_bridge_universality.{pdf,png}` | shipped; `fig_bridge_contribution.{pdf,png}` is the alternate-naming aggregate | `compose_fig2.py` |
| `fig3_reverse_bridge_10poc.{pdf,png}` | shipped | `compose_fig3.py` |
| `fig4_three_branch_typology.{pdf,png}` | shipped | `compose_fig4.py` |
| `fig5_ss_trl_playbook.{pdf,png}` | shipped | `compose_fig5.py` |

All five paper-caption figures are now present as both PDF and PNG, alongside their `compose_figN.py` generation scripts. The alternate-named versions (`fig_main_ss_correlation`, `fig_bridge_contribution`, `fig_lm_scaling`, etc.) are retained for the repo-native numbering cross-referenced in this file's §3 / §4 tables below.

## §3 Entropy criterion (Factor A)

| Figure | File | Shows |
|---|---|---|
| Fig 1 | `fig_main_ss_correlation.{png,pdf}` | ρ = −0.92 scatter: Structure Score vs Bits-per-Character, n = 29 |

## §4 Bridge construction

### §4.2 Forward-bridge universality (Factor A × D)

| Figure | File | Shows |
|---|---|---|
| Fig 2 | `fig_bridge_contribution.{png,pdf}` | Bridge-vs-zero-baseline bits across all LM scales × 16 domains |
| Fig S6 | `fig_bridge_contribution_xl_n16.{png,pdf}` | Extended n = 16 at XL encoder |

### §4.3 LM capacity compresses SS dependence (Factor C)

| Figure | File | Shows |
|---|---|---|
| Fig 3 | `fig_lm_scaling.{png,pdf}` | Bridge val-loss vs LM scale, coloured by SS, n = 8 |
| Fig S7 | `fig_lm_scaling_xl_n16.{png,pdf}` | Same at n = 16, XL encoder |

### §4.4 Domain-model capacity (Factor B)

| Figure | File | Shows |
|---|---|---|
| Fig 4 | `fig_s_model_scaling.{png,pdf}` | Val loss vs domain-model scale (S/M/XL) |
| Fig S11 | `fig_arch_comparison.{png,pdf}` | MLP vs Q-Former vs Linear projection (Factor B — architecture axis) |

### §4.5 Reverse-bridge capability (10 domains)

Per-domain scorecard + diagnostics figures live in `../bridges/reverse/{domain}/figures/`:

- `gcode/figures/` — Tests A & B (source-fidelity, paraphrase-robustness), hybrid ablation
- `midi/figures/` — scorecard, ablation, waveform
- `smiles/figures/` — scorecard, ablation, molecules render
- `python/figures/` — scorecard, ablation, feature heatmap (v2)
- `quantum/figures/` — scorecard, ablation
- `rna/figures/` — scorecard, diagnostics
- `network/figures/` — scorecard, ablation, diagnostics
- `dna_coding/figures/` — scorecard, diagnostics

### §4.6 Paired-description quality (Factor D)

| Figure | File | Shows |
|---|---|---|
| Fig 5 | `fig_source_comparison.{png,pdf}` | Natural vs synthetic vs semantic, per domain |
| Fig 6 | `fig_degeneracy.{png,pdf}` | Zero + random baselines vs trained bridge, all domains |
| Fig S15 | `fig_degeneracy_xl_n16.{png,pdf}` | Same at n = 16, XL encoder |

## Regenerating

Most figures regenerate from aggregate JSONs via `scripts/make_figures.py` (in the source repo; not yet ported to this public layout). Per-domain reverse-bridge figures are produced by each PoC's `score_and_visualise.py` or equivalent.

Order of regeneration:

1. Run per-domain entropy (`../factor-a-domain-structure/reproduce_rho.py`) → Fig 1
2. Aggregate forward-bridge runs (`../factor-c-language-model-capacity/aggregate_all.py`) → Figs 2–6
3. Run per-PoC reverse-bridge scoring → per-domain figures

For the paper submission, figures are final; regeneration is for reviewers who want to replicate.

## Colour legend

Across all cross-cutting figures, domains are coloured by **SS-sorted viridis**: low-SS = purple/blue, high-SS = yellow. Within a figure the category order is fixed (high → low SS left-to-right) for visual comparison across figures.
