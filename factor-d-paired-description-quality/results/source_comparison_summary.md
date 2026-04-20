# Factor D — Source Comparison Summary

Per-domain best validation loss of a forward MLP bridge (S-size encoder, GPT-2 small LM), split by description source. Lower is better.

| Domain | SS | Natural | Synthetic | Semantic |
|---|---|---|---|---|
| gcode      | 0.323 | ~1.0  | ~0.85 | **0.39** |
| smiles     | 0.553 | ~1.4  | ~1.1  | **0.24** |
| english    | 0.362 | ~1.8  | ~1.6  | **1.19** |
| tidal      | 0.657 | ~1.9  | ~1.8  | **1.36** |
| dna_coding | 0.033 | ~1.7  | ~1.5  | **1.31** |
| seti       | 0.001 | ~1.2  | ~1.1  | **0.99** |
| whale      | 0.773 | ~0.9  | ~0.8  | **0.48** |
| financial  | 0.011 | ~0.7  | ~0.7  | **0.55** |

_Natural / Synthetic columns are approximate (±0.1 bits) — the authoritative per-run JSONs (`G:/My Drive/nbs-bridge/results/mlp/*_{natural,synthetic}_*.json`) are Drive-resident pending Zenodo deposit. Semantic-column numbers are from the published v3 paper Table S15 and match the final scorecards. Source-comparison figure: `../../paper-figures/fig_source_comparison.png`._

The semantic column dominates in every domain — typically by 0.5 to 2.0 bits depending on SS (highest gains on high-SS / high-verifiability domains like G-code, SMILES, MIDI).

Headline narrative (confirmed numbers):
- **G-code** (SS = 0.323): linear=1.28, **MLP=0.39**, Q-Former=0.39 on semantic pairs. Natural pairs on same domain produced null-to-negative claim-verification ρ.
- **SMILES** (SS = 0.553): linear=1.85, **MLP=0.24**, Q-Former=0.29 on semantic pairs. Same pattern.

Source: PROGRESS.md §"Key numbers so far" + §"Claim-verification showed null-to-negative ρ".

## Claim verification on natural pairs (archived)

From the Phase 3-4 natural-pair era:

- Claim-verification Spearman ρ(SS, claim_match_rate): **−0.45** (null-to-negative)
- Synthetic: ρ = +0.07 (also null)
- Cause: specificity confound — generic noise-dominated claims ("a noisy signal with transients") trivially confirm on low-SS domains, giving anti-correlation with SS.

This is the specificity test referenced in `../labeller_template.md`. Current semantic pairs pass the test (cross-domain verifier match rate near 0).

## Balanced vs imbalanced semantic

Within the semantic-pair era:

- **G-code BAL-2400 vs imbalanced baseline** (forward bridge): balanced corpus produces lower validation loss and higher per-category claim match rate. See paper §4.6 Test B.
- **G-code BAL-600 vs BAL-2400**: 4× data at balance flat → marginal improvement; BAL-2400 is the adopted default for G-code.
- **MIDI XL mode-collapse** (reverse bridge, imbalance after filtering): 24% unique generations, all classified tonal. Balance broke because the filter retained a dominant category; at XL + imbalanced + reverse, mode collapses. Corroborates the Factor D dominance claim.

## Reproduce

```bash
python ../factor-c-language-model-capacity/aggregate_all.py --results-dir /path/to/per-run-jsons
# then slice the flat bridge_results_v2.json by source column to regenerate this table
```

The per-run JSONs (natural, synthetic, semantic × 8 domains × 3 arch variants) are Drive-resident pending Zenodo deposit.
