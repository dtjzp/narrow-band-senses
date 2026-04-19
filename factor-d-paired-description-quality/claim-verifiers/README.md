# Claim verifiers — pending Zenodo deposit

The per-domain claim verifiers used to score reverse-bridge outputs
against the category claims in paired prompts are **not shipped in
the public repository at this stage**. They live in the author's
Drive workspace at `G:/My Drive/nbs-bridge/scripts/{domain}_claim_verifier.py`
and will be mirrored to Zenodo on paper acceptance.

Each verifier takes (generated_sequence, claimed_category) → bool
and is used in the category-aware feature validator (Python v2 fix:
9.4% → 43.4%; see paper §4.6 Methodology).

Reviewers needing a verifier for a specific domain prior to the
Zenodo release can contact the corresponding author
([danielzp.com](https://danielzp.com)).

## Methodology reference

The category-aware validator rule: feature validators must be
category-aware. Each category claims a subset of features; matching
is averaged only over claimed features. Pattern codified in
[`../labeller_template.md`](../labeller_template.md).
