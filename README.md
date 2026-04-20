# Narrow-Band Senses

Companion repository for:

> **"Narrow-Band Senses: An Information-Theoretic Framework for Multimodal AI Perception"**
> Daniel Ziekenoppasser-Powell · 2026

## TL;DR

An information-theoretic framework that predicts *before training* which data modalities admit transformer-based bridges to language, then constructs those bridges and demonstrates them empirically. Five headline results:

1. **ρ(SS, normalised BPC) = −0.92 across 29 domains** (partial ρ = −0.929 controlling for H₀; 5000-permutation p < 2 × 10⁻⁴). The observed value is the finite-sample approximation of a theoretical limit ρ → −1 under standard information-theoretic assumptions.
2. **Forward bridges are universal** at SS > 0: 3–5 bits of validation-loss reduction uniformly across 16 domains at Pythia-1B × XL, seed-invariant (bridge-contribution SD ≤ 0.003 bits across 3 seeds × 3 domains).
3. **Reverse bridges across 10 symbolic modalities** split cleanly under a mandatory mode-collapse diagnostic gate into 5 PASS and 6 FAIL, with FAILs sorting into three mechanism-distinct classes.
4. **H_win/H₀ is a secondary predictor** (Spearman ρ = +0.73 on an n = 8 subset) distinguishing the attractor-collapse failure class that SS alone cannot resolve.
5. **Four-factor decomposition** (A: domain structure · B: domain-model capacity · C: LM capacity · D: paired-description quality) identifies **Factor D as the dominant engineering lever** where SS supports it.

**Three-branch failure typology** (applies to reverse-bridge FAILs):

- **low-SS** — SS < 0.15 (Network, DNA-coding); framework prescribes detection-utility only
- **within-window-attractor** — H_win/H₀ low (bioreactor, ATC, MIDI-XL); bridge collapses to per-category prototype sequences
- **compositional-hierarchy** — char-level training cannot assemble higher-order structure (reactions: requires `reactants>>products` pairing)

Plus **variance-bound** (RNA): multi-run non-reproducibility within a single configuration; flagged as a distinct failure mode motivating the multi-seed reporting rule.

## Reproduce — headline result

```bash
git clone https://github.com/dtjzp/narrow-band-senses.git
cd narrow-band-senses
pip install -r requirements.txt          # PyTorch, pandas, scipy, matplotlib
python factor-a-domain-structure/reproduce_rho.py
# → Spearman rho(SS, BPC) = -0.9236  (p = 9.29e-13, n = 29)
# → regenerates paper-figures/fig_main_ss_correlation.{png,pdf}
```

CPU-only, runs in under 30 seconds. Both pre-computed inputs ship in the repo:

- **`factor-a-domain-structure/results/canonical_training_entropy.json`** — 29 domains, H₀/H₃/SS.
- **`factor-a-domain-structure/results/bpc_per_domain.json`** — 29 domains, normalised BPC from small-Transformer fits (values transcribed from paper Tables S2a–S2b + §S6; `_source_note` field in the JSON documents the derivation).

The `--recompute-ss` mode recomputes SS from raw text streams and takes ~25 min CPU, but requires the raw `{domain}_1M.txt` streams (on Drive / pending Zenodo). Regenerating BPC from scratch requires training 29 small-Transformer fits on A100 (~30 A100-hours) and is not scripted here; see paper Methods §4.6.

H_win/H₀ metric (secondary predictor underpinning the three-branch failure typology) is also fully reproducible locally:

```bash
python bridges/reverse/compute_window_entropy.py --check-only
# → prints the 11-domain H_win/H₀ table from bridges/reverse/window_entropy_results.json
```

## Navigation

| Directory | What's here |
|---|---|
| [`factor-a-domain-structure/`](factor-a-domain-structure/) | Entropy criterion + 29-domain structure scores. The ρ = −0.92 headline. |
| [`factor-b-domain-model-capacity/`](factor-b-domain-model-capacity/) | S vs M vs XL domain-model scaling experiments. |
| [`factor-c-language-model-capacity/`](factor-c-language-model-capacity/) | GPT-2 small/medium/Pythia-1B comparisons + aggregation. |
| [`factor-d-paired-description-quality/`](factor-d-paired-description-quality/) | Natural vs synthetic vs semantic pairs. The dominant factor. |
| [`bridges/forward/`](bridges/forward/) | Forward-bridge architecture + training recipe + degeneracy controls. |
| [`bridges/reverse/`](bridges/reverse/) | 10 reverse-bridge domain PoCs + mode-collapse methodology + H_win/H₀ metric. |
| [`experiment/code/`](experiment/code/) | Character-transformer stack used for BPC evaluation (XS/S/M/L). |
| [`paper-figures/`](paper-figures/) | All paper figures, indexed by section. |
| [`protocols/`](protocols/) | protocols.io drafts for the three core methods. |
| [`archive/`](archive/) | Approaches that didn't work, with post-hoc analysis. |

## What the paper demonstrates

1. **Structure score predicts compressibility at scale.** A character-level H₃/H₀ ratio (the structure score, SS), computed offline in seconds, predicts Pythia-1B bits-per-character across 29 data modalities at ρ = −0.92 (p < 10⁻¹²). Partial correlation controlling for H₀ is ρ = −0.929; 5000-permutation test gives p < 2 × 10⁻⁴. Shuffle-baseline and out-of-training tidal verification corroborate.

2. **Forward bridges are broadly constructable.** A BLIP-2-style MLP projection from a frozen domain encoder to a frozen LM achieves 3–5 bits of validation-loss reduction over a zero-conditioning baseline at every LM scale tested, across 16 domains spanning SS = 0.001 to 0.773. Bridges contribute real information transfer everywhere — not confabulation from the LM prior.

3. **Reverse bridges are capability-class-dependent.** Text → domain generation under the canonical Protocol-03 recipe (S-size fine-tuned + 64 soft prefix tokens + temperature sampling) PASSes cleanly on **Quantum** (Stable PASS, 5 seeds, cat-match 70.4% ± 4.0), **Python** (Stable PASS, 5 seeds, 41.6% ± 4.3), **G-code** (PASS via non-saturable Tests A + B), **MIDI-S** (PASS at S; XL-scale mode-collapses, documented), and **SMILES-legacy** (PASS-legacy, pre-mandatory-diagnostic-gate). Fails on **Network + DNA-coding** (low-SS framework validation), **bioreactor + ATC** (within-window-attractor, H_win/H₀ = 0.29 and 0.67), **reactions** (compositional-hierarchy), and **RNA** (variance-bound across 9 iterations).

4. **The four-factor framework explains why.** Factor D dominates where SS supports it. Factor A × D interaction: SS > 0.3 + informative vocabulary → PASS; SS > 0.3 + narrow vocabulary → partial capability. Factor B × D interaction: reverse bridges use S-size only — XL collapses under balanced descriptions (documented in [`archive/xl-reverse-bridge-collapse/`](archive/xl-reverse-bridge-collapse/)).

## Methodology contributions

Seven standalone methodology recommendations for pair-task bridge studies emerged from the programme:

- Mandatory mode-collapse diagnostic gate (unique-gen rate + intra-category Hamming + cross-category leakage)
- Multi-seed reporting rule (motivated by RNA variance-bound finding)
- Temperature sampling, not greedy (MIDI greedy 36% vs sampled 60% on identical checkpoint)
- Category-aware feature validator (Python v2 fix: 9.4% → 43.4%)
- Data-driven category selection (Spec 3 v2 labeller gate)
- Round-trip labeller accuracy ≥ 95% (self-consistency gate)
- Pre-stated predictions committed to git before outcome observation

## Data availability

- **Structure-score inputs** (tokenised windows, 29 domains): included in [`factor-a-domain-structure/data/`](factor-a-domain-structure/data/) as JSON. ~20 MB.
- **Paired-description corpora** (semantic, 16 domains, BAL-1600 each): Zenodo — DOI to follow.
- **Model checkpoints** (S / M / XL per domain, forward + reverse bridges): Zenodo — DOI to follow.
- **Reverse-bridge generations** (held-out eval sets, all 10 domains): Zenodo — DOI to follow.

Until the Zenodo deposit is live, pointers in each `factor-*/results/` and `bridges/reverse/{domain}/` name the Drive locations where the authors maintain the artefacts; request access via the paper's corresponding author.

## Citation

```bibtex
@article{ziekenoppasserpowell2026nbs,
  title   = {Narrow-Band Senses: An Information-Theoretic Framework for
             Multimodal AI Perception},
  author  = {Ziekenoppasser-Powell, Daniel},
  year    = {2026},
  journal = {Nature Machine Intelligence (submitted)}
}
```

See also [`CITATION.cff`](CITATION.cff). Protocols are separately citable with DOIs listed in [`protocols/README.md`](protocols/README.md).

## License

- **Code** (`*.py`): MIT.
- **Documentation** (`*.md`) and **figures** (`paper-figures/`): CC-BY 4.0.

See [`LICENSE`](LICENSE) for full text of both.

## Contact

Daniel Ziekenoppasser-Powell — independent researcher, UK — [danielzp.com](https://danielzp.com).
Issues and pull requests welcomed via GitHub.

## AI tool use disclosure

This repository and the associated paper's experimental programme were prepared with the assistance of Anthropic's Claude Opus 4.7 operating as a coding agent under the author's direction. All experimental design, interpretation, and final claims are the author's. Compute: Colab Pro+ (~230 A100 credits consumed). Detailed disclosure in the paper's "AI tool use" section.
