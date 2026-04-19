# Factor C — Correlation Tables

Authoritative: regenerate from `bridge_results_v2.json` by running `../aggregate_all.py`.

## ρ(SS, bridge_val) — how well does SS predict absolute val loss?

|  | GPT-2 small | GPT-2 medium | Pythia-1B |
|---|---|---|---|
| **n = 8, S-size encoder, semantic** | (see aggregate_all output) | (same) | (same) |
| **n = 8, XL-size encoder, semantic** | (same) | (same) | (same) |
| **n = 16, XL-size encoder, semantic** | (same) | (same) | (same) |

## ρ(SS, zero − best_val) — how well does SS predict bridge contribution?

|  | GPT-2 small | GPT-2 medium | Pythia-1B |
|---|---|---|---|
| **n = 16, XL-size encoder, semantic** | **+0.30** | **+0.12** | **+0.32** |

Source: handover spec §4.2 (2026-04-19) after n=16 extension. Non-monotonic — this supersedes the v1 manuscript's monotonic decay claim on the older `improvement` metric.

## ρ(SS, improvement) — legacy metric, for comparison only

|  | GPT-2 small | GPT-2 medium | Pythia-1B |
|---|---|---|---|
| **n = 8, XL-size encoder, semantic** | +0.762 | +0.19 | −0.14 |

`improvement` was defined as `(zero_prompt_val − bridge_val) / zero_prompt_val`. The denominator sensitivity to LM-level absolute val loss made this metric noisy across LM scales; the paper now uses `zero − best_val` (bits, unnormalised).

## Bridge contribution (zero − best_val) — absolute bits by domain, n = 8

See `../paper-figures/fig_bridge_contribution.png` and `fig_bridge_contribution_xl_n16.png`. Contribution is **3.0–4.5 bits at Pythia-1B** uniformly across SS = 0.001 to 0.773. Bridge helps everywhere — the degeneracy hypothesis (that bridges only confabulate at low SS) is falsified.

## Reproduce

```bash
python ../aggregate_all.py
```

Stdout will print fuller tables including per-metric slice correlations. Save the output alongside this file when regenerating.
