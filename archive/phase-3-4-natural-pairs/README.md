# Phase 3-4 — Natural Prose Pairs (Null-to-Negative)

## What was tried

Trained linear and MLP bridges on **natural prose descriptions** — text written in the voice of a specific domain, style-matched to a human working in that field:

- **SETI**: radio-astronomy observation-log voice ("Observed a brief narrow-band signal at 1420 MHz; amplitude consistent with instrumental noise but with an unusual 40-second burst")
- **Tidal**: oceanographic measurement-report voice
- **Whale**: cetacean-bioacoustics field-note voice
- **Financial**: market-report voice
- **English**: literary-commentary voice
- **G-code**: CNC-operator voice
- **SMILES**: medicinal-chemistry voice
- **DNA coding**: molecular-biology lab-note voice

Corpus size: 300–500 pairs per domain. Generated with care (human-edited where possible, LLM-generated elsewhere).

## What happened

Claim-verification Spearman **ρ(SS, claim_match_rate) = −0.45** across n=8 domains (null-to-negative; directional but wrong sign).

- Low-SS domains (SETI, financial) produced **higher** claim-match rates than high-SS domains.
- High-SS domains (whale, SMILES) produced lower claim-match rates.

This is the opposite of the framework's prediction.

## Why it didn't work — the specificity confound

Natural descriptions of low-SS domains tend to use **generic noise-dominated claims**: "a noisy signal with occasional transients," "background activity with brief excursions." These claims trivially confirm on any low-SS stream — because any low-SS stream has noise and occasional transients by definition.

High-SS domains, paradoxically, generate more *specific* natural descriptions because the signal supports specificity. And specific claims are harder to verify — the bridge has to actually produce the right structure, not just any structure.

The null-to-negative ρ came from the **verifier's specificity mismatch**, not from the bridge failing. The descriptions were the wrong target, not the wrong method.

## What was learned

1. **Verifiability ≠ fluency.** Natural prose descriptions are fluent but not verifiable by the bridge because verification requires structured claims.
2. **Pair generation has a quality axis beyond length, voice, or style.** The axis that matters is "how many discriminating structural claims can the description support?"
3. **Cross-domain specificity test** (applying a verifier for domain X to descriptions from domain Y) exposes the specificity confound. Natural pairs fail this test; semantic pairs pass it.

## Files

- `gen_seti_natural.py`, `gen_tidal_natural.py`, `gen_gcode_part1.py` — natural-pair generation scripts
- `score_english_natural_mlp.py`, `score_gcode_natural_mlp.py`, `score_seti_natural_mlp.py`, `score_smiles_natural.py`, `score_tidal_natural.py`, `score_whale_synthetic_bridge.py` (last one is misnamed — does natural whale), `verify_dna_coding_natural.py`, `verify_whale_natural.py` — natural-pair claim-verification scorers
- `eval_financial_natural_mlp.py` — eval prep

Raw per-domain natural pairs are in `G:/My Drive/nbs-bridge/paired_data/natural/{domain}.jsonl`.

## What replaced this

Tier-1 **semantic pairs** (`../../factor-d-paired-description-quality/`). Verifiable, structured, balanced, auto-generated. This is what works.
