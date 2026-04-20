# Factor D — Paired-Description Quality

**Definition**: the properties of the text-side corpus used to train a bridge. In this work we compare three kinds of paired description, and study how **balance** within a kind changes bridge behaviour.

**Three kinds of pair**:

1. **Natural** — prose written in the voice of the domain (e.g. "a radio-astronomy observation log" for SETI, "a MIDI performance annotation" for music). Phase 1.4 of the original experiment; retired after null results.
2. **Synthetic** — abstract pattern descriptions ("short run of 1s, then a long run of 0s"). Phase 1.5; retired after null results.
3. **Semantic** — machine-auto-generated **verifiable** descriptions, derived by running a deterministic labeller over each sequence window that extracts structured claims (e.g. for G-code: `{category: perimeter, g2g3_arc_count: 7, x_range_mm: [42.1, 78.3], …}`). This is what works. Canonical from 2026-04-17 onward.

**Balance**: within the semantic kind, we vary *how balanced* the corpus is across categories. `BAL-1600` is 320 pairs per category × 5 categories. `BAL-2400` extends to ~480 per category. Imbalanced corpora (natural category distributions, dominated by 60-80% majority categories) exist for comparison.

## Key finding (the dominance claim)

**Factor D dominates**. Concretely:

- **Semantic pairs beat natural + synthetic pairs at every comparable point** in the Factor A × B × C grid. See paper §S15 and `../paper-figures/fig_source_comparison.png`.
- **Balanced semantic > imbalanced semantic at small model scale**, validated on G-code BAL-2400 forward bridge (beats prior imbalanced recipe) and corroborated by the MIDI XL mode-collapse failure when balance breaks at scale. See paper §4.6.
- **The factor-dominance rule of thumb**: *balanced-pairs-at-S-size beats imbalanced-pairs-at-XL-size* on reverse-bridge conditional fidelity. This is the validated recipe for reverse bridges.

This finding is why the reverse-bridge chapter exists: without Factor D's insight, the naive scaling approach (XL + natural pairs) gives mode-collapsed outputs that look like capability at the aggregate level but fail per-category. See `../bridges/reverse/mode-collapse-diagnostics.md`.

## Figures

| Figure | Shows |
|---|---|
| [`../paper-figures/fig_source_comparison.png`](../paper-figures/fig_source_comparison.png) | Natural vs synthetic vs semantic pair bridge-val side-by-side |
| [`../paper-figures/fig_degeneracy.png`](../paper-figures/fig_degeneracy.png) | Semantic bridge vs zero + random baselines |
| [`../paper-figures/fig_degeneracy_xl_n16.png`](../paper-figures/fig_degeneracy_xl_n16.png) | Same at XL encoder, n = 16 |

## What's in this directory

| File | Purpose |
|---|---|
| `semantic-labellers/` | Per-domain verifiable auto-labellers. The labeller for a domain is a deterministic Python function that reads a tokenised window and emits a structured claim dict. This dict is then rendered as natural-language description via a fixed template. |
| `claim-verifiers/` | The reverse operation: read a generated description (or sequence) and extract which claims are verifiably true. Used for both held-out verification and mode-collapse diagnostics. |
| `labeller_template.md` | Specification every semantic labeller must obey: input contract, output schema, determinism, round-trip accuracy target. |
| `results/source_comparison_summary.md` | Per-domain best-val loss broken out by source kind (natural/synthetic/semantic). |

Canonical per-domain semantic labellers are Drive-resident pending Zenodo deposit on paper acceptance. See [`semantic-labellers/README.md`](semantic-labellers/README.md) and [`claim-verifiers/README.md`](claim-verifiers/README.md) for the current placeholder, interface contract ([`labeller_template.md`](labeller_template.md)), and corresponding-author contact for early access.

## Why "verifiable"?

The natural-pair and synthetic-pair approaches failed because descriptions were not *verifiable*: a text saying "a noisy signal with occasional transients" could be trivially confirmed by any domain with noise and transients. Cross-domain, the descriptions underdetermined the sequence — so the bridge had nothing specific to encode.

Semantic pairs fix this by construction. Every claim in a semantic description is computed from the sequence, so verifying the bridge's reconstruction against the original claim set is meaningful. Chance-level reconstruction is low (≈chance for categorical, 0 for continuous).

The critical construction lesson: **pair quality ≠ description length ≠ description fluency**. A 3-line semantic description outperforms a paragraph-long natural description. Fluency is not the bridge-relevant signal; verifiability is.

## Labeller + verifier round-trip

For each domain, the labeller and verifier are a pair:

```python
window -> labeller -> claim_dict -> template -> description
description -> verifier -> extracted_claims
```

Round-trip accuracy is computed by asking whether `extracted_claims == claim_dict` on the held-out set. All Tier-1 labellers achieve 100% round-trip on their own outputs (trivially, by construction). Cross-domain round-trip (verifier run on a different domain's description) should be near-zero — this is the specificity test.

See `labeller_template.md` for the interface.

## How Factor D interacts with the other factors

- **A × D**: Orthogonal. Semantic pairs work across the SS range. Low-SS domains with near-noise signal still admit a small number of verifiable categorical claims (e.g. for SETI: "sparse transients? yes/no"; for financial: "trend up / down / flat").
- **B × D**: Factor D dominates Factor B. Balanced semantic at S-size beats imbalanced natural at XL-size. This is the G-code BAL-2400 result.
- **C × D**: Semantic pairs work at every LM scale. Scaling Factor C does not rescue a bad pair corpus — the failure mode of natural/synthetic pairs persists at Pythia-1B.

## Open questions

- Does the Factor D dominance claim extend beyond 10 domains? Tested on G-code (confirmed) and MIDI (corroborating negative via XL failure). Untested elsewhere.
- Is there an even-better pair design above "balanced semantic"? Candidates: adversarially-hard semantic (where descriptions are intentionally close across categories), compositional semantic (describe sub-windows separately), multi-granularity semantic. All future work.

## References in the paper

- §4.6 Paired-description quality dominates reverse conditional fidelity (main result)
- §S12 Semantic Auto-Labeller Specification
- §S15 Reverse Bridge Fidelity Analysis: Imbalanced and Balanced Corpora

## Archive

Natural and synthetic-pair experiments are not in this directory — they live at [`../archive/phase-3-4-natural-pairs/`](../archive/phase-3-4-natural-pairs/) and [`../archive/phase-3-4-synthetic-pairs/`](../archive/phase-3-4-synthetic-pairs/), with READMEs explaining the specificity confound and why they produced null results.
