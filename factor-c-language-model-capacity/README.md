# Factor C — Language-Model Capacity

**Definition**: the capacity of the frozen target language model (LM) that the forward bridge projects into. Varied in this work across three public checkpoints:

- **GPT-2 small** (124 M params, Radford et al. 2019)
- **GPT-2 medium** (355 M params, same family)
- **Pythia-1B** (1.01 B params, Biderman et al. 2023)

## Key finding

**Forward bridges are constructable at every tested LM scale, uniformly across the SS spectrum.**

- Bridge contribution (bits of validation-loss reduction vs zero-conditioning baseline) is **3.0–4.5 bits at Pythia-1B, uniformly across SS = 0.001 to 0.773**, with 16 domains tested (n = 16).
- The Spearman correlation between SS and the zero-minus-best metric is **non-monotonic in LM scale**: approximately +0.30 at GPT-2 small, +0.12 at GPT-2 medium, +0.32 at Pythia-1B (paper §4.3 table, n = 16).
- This is a revised result from the v1 manuscript, which reported a monotone-decay pattern on an older `improvement` metric at n = 8 (+0.76 → +0.19 → −0.14). The n = 16 extension with the cleaner zero-minus-best metric flattens the picture: feasibility is universal, with a weak and non-monotonic SS dependence that reflects decoding ceilings rather than encoding failure.

## Figures

| Figure | Shows |
|---|---|
| [`../paper-figures/fig_lm_scaling.png`](../paper-figures/fig_lm_scaling.png) | Bridge validation loss vs LM scale, per domain (original n = 8) |
| [`../paper-figures/fig_lm_scaling_xl_n16.png`](../paper-figures/fig_lm_scaling_xl_n16.png) | Extended n = 16, XL-size encoder, three LM scales |
| [`../paper-figures/fig_bridge_contribution.png`](../paper-figures/fig_bridge_contribution.png) | Zero-minus-best bits across LM scales, coloured by SS |

## What's in this directory

| File | Purpose |
|---|---|
| `aggregate_all.py` | Aggregates per-run training JSONs from Drive into the flat `bridge_results_v2.json` + prints Spearman tables per (arch, source, lm, s_size, metric). |
| `architecture.md` | Details of the forward-bridge projection into each LM's embedding space; tokeniser handling; prefix length |
| `results/bridge_results_v2.json.md` | Schema documentation for the aggregated results file (the JSON itself lives on Drive / Zenodo) |
| `results/lm_scale_correlation_tables.md` | Per-metric correlation tables across (n = 8, n = 16) × (GPT-2 small, GPT-2 medium, Pythia-1B) |

## Reproduce

Aggregate the correlation tables from per-run training JSONs:

```bash
cd factor-c-language-model-capacity
python aggregate_all.py   # reads G:/My Drive/nbs-bridge/results/, writes .../bridge_results_v2.json
```

Stdout: Spearman ρ tables per (arch, source, lm, s_size, metric). See `results/lm_scale_correlation_tables.md` for the committed reference copy.

## How Factor C interacts with the other factors

- **A × C**: Not interacting strongly. Bridges work across the SS range at every LM scale. Low-SS domains produce lower absolute val-loss reduction but still produce meaningful bridge-vs-zero deltas at Pythia-1B.
- **B × C**: At constrained C (GPT-2 small), B matters more (see Factor B README). At Pythia-1B, the marginal value of XL-size encoders compresses. The LM's depth absorbs what the encoder could have contributed.
- **C × D**: Less studied directly; all Factor C runs use semantic-balanced pairs. Natural/synthetic pairs were retired before the Pythia-1B extension (see `../archive/phase-3-4-natural-pairs/` and `.../synthetic-pairs/`).

## What this factor does NOT do

- C does not saturate within the 124 M → 1 B range. Bridge contribution still grows somewhat from GPT-2 medium → Pythia-1B in absolute terms, although SS-dependence compresses.
- C is not a substitute for Factor D. Scaling the LM does not rescue a bad paired-description corpus. The Factor D dominance claim (Factor D README) is the corollary.

## Open questions

- Does bridge contribution keep growing with LM capacity beyond Pythia-1B? Untested here — budgeted for follow-up (Pythia-2.8B or equivalent, ~60 A100-hr).
- Does the non-monotonicity in ρ(SS, zero−best) persist with encoder-side in-context conditioning (as opposed to soft-prefix)? Untested.

## References in the paper

- §4.3 Language-model capacity compresses SS dependence (rewritten to reflect n = 16 non-monotonicity)
- §5.4 Testable predictions (the v2 prediction of ρ ∈ [+0.35, +0.65] at n = 16 did not land; actual +0.30. Failed prediction reported honestly in v3.)
- §S18 n = 16 Domain Expansion
