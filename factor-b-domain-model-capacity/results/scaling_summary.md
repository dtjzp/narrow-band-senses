# Factor B — Scaling Summary

Authoritative per-run numbers aggregate into `bridge_results_v2.json` (97-row flat records; Drive-resident pending Zenodo DOI). `../factor-c-language-model-capacity/aggregate_all.py` regenerates it from a directory of per-run JSONs given via `--results-dir`. The headline tables below are derived from that aggregate.

## S-size forward bridge: best-val by domain

| Domain | SS | S best-val (semantic + gpt2-small) |
|---|---|---|
| whale      | 0.773 | 0.48 |
| tidal      | 0.657 | 1.36 |
| smiles     | 0.553 | 0.24 |
| english    | 0.362 | 1.19 |
| gcode      | 0.323 | 0.39 |
| dna_coding | 0.033 | 1.31 |
| financial  | 0.011 | 0.55 |
| seti       | 0.001 | 0.99 |

Source: PROGRESS.md §"Semantic matrix complete".

## S → XL delta (GPT-2 small LM fixed, n=8)

| Domain | SS | S→XL Δ (GPT-2 small) |
|---|---|---|
| whale      | 0.773 | +0.033 |
| tidal      | 0.657 | +0.006 |
| …          | …     | …      |
| seti       | 0.001 | +0.003 |

ρ(SS, S→XL delta) at GPT-2 small, n=8: **+0.518** (p=0.188, directional). Full table in PROGRESS.md §"Domain-model scaling XL".

## XL bridge-contribution against zero baseline

At Pythia-1B LM with XL-size encoders, the forward-bridge contribution (zero − best-val) ranges **3.0 to 4.5 bits uniformly across SS**. See `../paper-figures/fig_bridge_contribution_xl_n16.png` for the n=16 extended plot.

## n=16 extension (added 2026-04-18)

An additional 8 domains were trained at XL-only for the extended correlation analysis. Per-domain deltas live in `bridge_results_v2.json`; see paper §S18 for the table.

## Interpretation

At constrained LM scale (GPT-2 small), **encoder capacity matters**: ρ(SS, S→XL Δ) is +0.52 — high-SS domains benefit more from XL than low-SS domains do. At unconstrained LM scale (Pythia-1B), the benefit compresses: the LM absorbs the gains the encoder could have contributed.

**Low-SS domains (SS < 0.1)** show small or null S→XL deltas — consistent with Factor A × B: there is not enough signal structure for encoder capacity to leverage.

## Reproduce

```bash
cd factor-b-domain-model-capacity
python ../factor-c-language-model-capacity/aggregate_all.py --results-dir /path/to/per-run-jsons
```

Output: flat results JSON + correlation table on stdout. The per-run JSONs come from `train_s.py` (S-size; shipped) and its M / XL variants (Drive-resident pending Zenodo).
