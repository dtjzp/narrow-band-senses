# Archive — Approaches We Tried That Didn't Work

This directory preserves approaches that were explored during the development of the NBS language-bridge paper and discarded after empirical tests. Each subdirectory contains:

- A README explaining what was tried, what result came out, and what was learned
- The scripts used (as-is, without modification from their original form)
- Pointers to the data, if the raw data is still accessible

**Why this exists**: reviewers rightly ask "did you try X?" where X is often the naive first-choice. Preserving the attempts with honest reporting pre-empts the question and documents the empirical path from "we thought this would work" to "we found out it didn't and here's why."

## Contents

| Directory | What was tried | Why it was abandoned |
|---|---|---|
| [`phase-2-cka/`](phase-2-cka/) | CKA similarity between encoder hidden states and LM hidden states | Null result; CKA is not sensitive enough to the kind of alignment bridges require |
| [`phase-3-4-natural-pairs/`](phase-3-4-natural-pairs/) | Natural prose descriptions as bridge-training targets (style-matched voice per domain) | Null-to-negative claim-verification ρ due to specificity confound |
| [`phase-3-4-synthetic-pairs/`](phase-3-4-synthetic-pairs/) | Synthetic pattern-abstract descriptions ("short run of 1s, then long run of 0s") | Null claim-verification ρ; descriptions underdetermined the sequence |
| [`xl-reverse-bridge-collapse/`](xl-reverse-bridge-collapse/) | XL-size domain model for reverse bridge | 2–24% unique generations across G-code and MIDI — mode collapse at the XL + soft-prefix recipe |

## Timeline

| Phase | Period | Output |
|---|---|---|
| Phase 0-1 | Pre-2026-04-17 | Data setup, encoding choices |
| Phase 2 | ~2026-04-12 | CKA null result (`phase-2-cka/`) |
| Phase 3-4 | 2026-04-13 to 2026-04-17 | Natural + synthetic pair attempts, both null |
| Tier-1 semantic | 2026-04-17 onward | **Breakthrough** — verifiable semantic descriptions. This is the canonical recipe in the paper. |
| XL reverse bridge | 2026-04-18 | Collapsed; S-size adopted as the reverse-bridge default |

## Reading order

Start with Phase 2 (the CKA null) for historical interest, then Phase 3-4 (the null pair-era) to understand why semantic pairs matter, then XL reverse-bridge collapse for the Factor B × D interaction that justifies the S-only reverse-bridge recipe.

## The rescue pattern

In each of these failures, the fix was not "more scale" or "more compute" or "more data." The fix was:

1. **Phase 2** → abandoned. CKA was the wrong measurement.
2. **Phase 3-4 (natural)** → replaced by semantic pairs. Verifiability, not fluency, was the missing ingredient.
3. **Phase 3-4 (synthetic)** → replaced by semantic pairs. Specificity, not abstraction, was the missing ingredient.
4. **XL reverse bridge** → replaced by S-size reverse bridge + balanced pairs. Factor D dominance validated.

The paper's four-factor framework is in part a retrospective account of *which* factor each of these approaches got wrong. See `../factor-d-paired-description-quality/README.md` for the successor recipe that worked.
