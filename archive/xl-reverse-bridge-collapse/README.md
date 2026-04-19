# XL Reverse-Bridge Collapse (Capacity-Destructive Interaction)

## What was tried

For the text→domain reverse bridge, scaled the domain model from S-size (~50M params) to XL-size (~380M params), holding other factors fixed. Expectation, naïvely: more encoder capacity → better reverse-bridge generation.

## What happened

**At XL, the reverse bridge mode-collapses.** Observed across two independent domains:

- **MIDI BAL-1600 XL**: 24% unique generations, 100% classified as the single `tonal` category regardless of prompt. Temperature sampling at 0.8, top-k 40.
- **G-code BAL-600 XL**: 2% unique generations. Total collapse.
- **G-code BAL-2400 XL**: OOM on A100 at standard batch size; smaller batch didn't converge to non-collapsed state in comparable wall-clock budget.

At S-size, the same recipe produced non-collapsed generations (e.g. MIDI S: 60.3% overall fidelity, 78–100% unique per category on Python and quantum PoCs).

## Why it didn't work

Two interacting failure modes:

### 1. Capacity-destructive interaction with soft-prefix

XL-size domain models have a richer internal attractor landscape than S-size. When conditioned on 64 soft prefix tokens (small relative to the model's intrinsic dimensionality), the conditioning signal cannot overcome the model's existing strong-attractor priors. The model reverts to its most frequent training-data configuration regardless of prompt.

This is the structural Factor B × D interaction predicted by the paper's §4.6: at XL, the domain model's prior dominates; at S, the bridge's conditioning has sufficient leverage.

### 2. Loss landscape geometry

XL fine-tuning at 0.1× bridge LR produces smaller per-step updates to the domain model. Combined with XL's sharper loss landscape, training converges to a narrow minimum that happens to encode a single canonical output. S-size fine-tuning at the same relative LR is more exploratory.

Increasing fine-tune LR at XL helps marginally but at the cost of destroying domain-model priors (val loss diverges).

## What was learned

1. **Reverse bridges use S-size only.** This is now a hard rule in the paper's methodology (§4.6) and repo (`../../bridges/reverse/architecture.md`).
2. **More capacity is not always better.** For reverse bridges specifically, there is an inverted-U: S works, M might work, XL collapses. The sweet spot is at the "just enough domain knowledge + high conditionability" boundary.
3. **The Factor D dominance claim is strengthened.** Balanced S + semantic pairs beats imbalanced XL + anything. This was the decisive validation of the four-factor framework's predictive claim.
4. **Forward bridges are immune to this failure mode.** The forward-bridge direction (domain → text) works at XL because the soft prefix is not trying to override a frozen autoregressive decoder's strong priors — it is providing additional input to an LM that is already open-to-conditioning via its native language modelling.

## Files

- (pointers only) `G:/My Drive/nbs-bridge/results/reverse/gcode_semantic_v2ft_sXL_reverse_*.{pt,jsonl,json}` — the G-code XL reverse checkpoint + generated samples + training log
- (pointers only) `G:/My Drive/nbs-bridge/results/reverse/midi_semantic_v2ft_sXL_reverse_*` — MIDI XL equivalents
- `../../bridges/reverse/midi/REPORT.md` §XL Mode-Collapse Finding — per-PoC detail

## What replaced this

S-size reverse bridges (S-only is the default from 2026-04-18 onward). The G-code BAL-2400 S bridge achieves 60%+ cat-match with balanced per-category diversity; MIDI S-sampled achieves 60.3% overall fidelity. See `../../bridges/reverse/README.md` cross-PoC table.

## Caveat and follow-up

The XL-reverse-bridge follow-up spec (`docs/superpowers/specs/2026-04-18-nbs-xl-reverse-and-source-fidelity-design.md`) proposes a different recipe for XL — fine-tune the domain model at a higher relative LR, pair with a source-fidelity test (Tests A + B from G-code) to separate bridge contribution from domain-model priors. This is queued pending remaining A100 budget.

The current conclusion — "XL reverse-bridge collapses under the standard recipe" — is robust. The conjecture that "XL could work with a different recipe" is open.
