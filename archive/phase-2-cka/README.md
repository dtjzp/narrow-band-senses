# Phase 2 — CKA Similarity (Null Result)

## What was tried

Measured **Centred Kernel Alignment (CKA)** between the top-layer hidden state of the frozen domain model (S-size, trained on the domain's canonical stream) and the top-layer hidden state of GPT-2 small on matched paired prose.

Rationale: if narrow-band senses are "just another signal with high internal structure," a frozen encoder trained on the signal should produce hidden states that CKA-align with the LM's response to descriptions of the same content. If CKA ρ(SS, CKA) is positive, bridges should exist along the predicted SS gradient.

## What happened

Across the 8-domain (n=8) test set with both natural and synthetic paired prose, **ρ(SS, CKA) ≈ 0.1 on natural pairs, ≈ 0 on synthetic pairs** (not significant). CKA scores were all in the 0.01–0.05 range — uniformly low, with no SS-dependence.

See `1-research/nbs-experiment/results/correlation_results.json` for the historical record (ρ = −0.045, n = 10 on the earlier 10-domain pilot).

## Why it didn't work

CKA measures *representational similarity*, which is a much weaker condition than *bridge buildability*. Two representations can have low CKA and still admit a learned linear (or MLP) projection between them — CKA is invariant to learnable rescaling and rotation, but it requires the invariances to be approximately matched *without* any learned transformation.

Bridges add a learned transformation. So the absence of CKA similarity does not imply the absence of a trainable bridge. Phase 3-4 later showed that actual bridges (not CKA) did produce meaningful signal at the semantic-pair level.

## What was learned

1. **CKA is not sensitive to the kind of alignment a trainable bridge creates.** For this research question, measure the bridge's performance directly; don't proxy through representational-similarity metrics.
2. **The null is informative, not noise.** It ruled out a "domains are already aligned; you just need to measure it" hypothesis.

## Files

- `run_phase2_local.py` — **archived, not runnable**. Imports a `phase2_cka` module that lived at `G:/My Drive/nbs-bridge/scripts/` and was never ported to the public repo. Kept for historical reference; the file now `SystemExit`s with a pointer to this README if invoked.
- `prep_eval_local.py` — **archived, not runnable**. Same pattern: depends on Drive-only `phase34_prepare_eval` module + corpora.

Raw CKA scores lived in `G:/My Drive/nbs-bridge/results/cka/` during the study; the numeric summary is preserved in `1-research/nbs-experiment/results/correlation_results.json` (ρ = −0.045, n = 10 on the earlier 10-domain pilot).

**Why are these scripts shipped at all?** To preserve the full provenance of what was tried on the path to the working recipe. Phase 2 was a legitimate null result, not a failed experiment to be hidden. Deleting the scripts would destroy the audit trail; leaving them runnable would mislead reviewers who try them. The middle ground is what's done here: scripts kept, guarded with an explicit `SystemExit`, README flagged honestly.

## What replaced this

Phase 3-4 (bridge training with claim-verification) in place of CKA; see `../phase-3-4-natural-pairs/` and `../phase-3-4-synthetic-pairs/` for the next attempts (both null) and then `../../factor-d-paired-description-quality/` for the successful semantic-pair recipe.
