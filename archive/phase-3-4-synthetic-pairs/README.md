# Phase 3-4 — Synthetic Pattern-Abstract Pairs (Null)

## What was tried

Trained bridges on **synthetic pattern descriptions** — abstract pattern vocabularies meant to be domain-agnostic: "short run of 1s, then long run of 0s," "repeated triplet followed by pause," "monotonic increase to plateau."

Rationale: natural prose overfit to domain-specific clichés (see `../phase-3-4-natural-pairs/README.md`); perhaps a shared abstract vocabulary — treating every domain's sequence at the pattern-primitive level — would factor out the specificity confound.

## What happened

Claim-verification Spearman **ρ(SS, claim_match_rate) = +0.07**, n=8 (null, not significant).

Synthetic descriptions didn't hurt (unlike natural), but they didn't help either. Bridges trained on synthetic pairs produced val-loss curves that converged but no cross-domain signal emerged.

## Why it didn't work — under-determination

Synthetic descriptions *under-determined* their target sequences. The pattern vocabulary was broad enough that many sequences satisfied the claim set:

- "short run of 1s, then long run of 0s" fits arbitrary run-length-encoded sparse signal
- "monotonic increase to plateau" fits many shapes

The bridge learned to emit text consistent with any matching signal, not the specific one from its training pair. Loss decreased (pattern description is a valid summary) but conditional fidelity (text → specific sequence reconstruction) stayed at chance.

This is the dual of the natural-pair failure: natural pairs were too specific to be verifiable; synthetic pairs were too generic to be discriminating.

## What was learned

1. **Pair specificity must be a two-way street.** A description must be specific enough that different sequences produce different descriptions (discriminating) AND short enough / structured enough to be verifiable (parseable by a verifier).
2. **The sweet spot is "structured verifiable claims"**: category + count + range + boolean features tied to the sequence by deterministic extraction. See `../../factor-d-paired-description-quality/labeller_template.md`.

## Files

- `verify_gcode_synthetic.py` — G-code synthetic-pair verification
- `score_financial_synthetic.py`, `score_seti_synthetic.py`, `score_whale_synthetic_bridge.py` — synthetic-pair scorers
- `claim_verify_english_synthetic.py`, `tidal_synthetic_claim_eval.py` — per-domain claim verifiers

Raw per-domain synthetic pairs are in `G:/My Drive/nbs-bridge/paired_data/synthetic/{domain}.jsonl`.

## What replaced this

Tier-1 **semantic pairs** (`../../factor-d-paired-description-quality/`), which achieve both specificity and verifiability by construction via deterministic per-domain labellers.
