# Reverse Bridges: Text → Domain

A **reverse bridge** takes a natural-language prompt and generates a tokenised sequence in the target domain (G-code, MIDI, SMILES, Python, quantum, RNA, network-flow, DNA-coding, ATC, reactions). The bridge architecture is a fine-tuned domain model conditioned on `n_soft = 64` learnable soft-prefix tokens that encode the prompt; the prompt is tokenised via the LM's tokeniser and passed through a trainable MLP to produce the soft prefix. See [`architecture.md`](architecture.md).

## Scope

**10 domain PoCs, spanning SS = 0.033 to 0.698.** Each PoC is a self-contained evaluation producing:

- A balanced, stratified paired corpus (BAL-960–2400 semantic, 4–5 categories)
- A fine-tuned S-size domain model
- A trained reverse bridge
- Held-out eval (128–329 generations, temp = 0.8 sampling)
- Mandatory mode-collapse diagnostics (v2 protocol, all PoCs from 2026-04-19)
- A per-PoC `REPORT.md` with honest headline + per-category breakdown

Directory per domain: [`gcode/`](gcode/), [`midi/`](midi/), [`smiles/`](smiles/), [`python/`](python/), [`quantum/`](quantum/), [`rna/`](rna/), [`network/`](network/), [`dna_coding/`](dna_coding/), [`atc/`](atc/), [`reactions/`](reactions/).

## 10-domain cross-PoC summary

| # | Domain | SS | Verdict | Failure mode | Honest headline |
|---|---|---|---|---|---|
| 1 | quantum    | 0.698 | **Stable PASS** (5 seeds) | — | cat-match 70.4% ± 4.0, feat-match 78.6% ± 5.0 — strongest demo |
| 2 | python     | 0.523 | **Stable PASS** (5 seeds) | — | cat-match 41.6% ± 4.3 (2× chance over 5 cats), feat-match 51.9% ± 2.3 |
| 3 | gcode      | 0.323 | **PASS** (Tests A + B) | — | M1 source fidelity 67.8% win-rate; paraphrase robustness 40% bridge vs 22% regex |
| 4 | midi       | 0.340 | **PASS at S / FAIL at XL** (documented) | within-window-attractor (XL only) | S-scale: cat-match 30%, tonal 86%, directional 70–87%; XL-scale: 24% unique, all tonal |
| 5 | smiles     | 0.553 | **PASS-legacy** (pre-gate) | — | parse 79%, cat-match 32%, MW-match 11%; drug-like outputs in renders |
| 6 | rna        | 0.675 | **FAIL** | variance-bound | parse 100% (bracket balance learned); non-reproducible across 9 iterations |
| 7 | network    | 0.126 | **FAIL** | low-SS | 60% cross-category leakage to `nine-heavy` attractor; aggregate 50% is artefact |
| 8 | dna_coding | 0.033 | **FAIL** | low-SS | intra-cat Hamming FAIL on 2/4 cats; second low-SS framework validation |
| 9 | atc        | 0.563 | **FAIL** (3 seeds) | within-window-attractor | cat-match 52.1% ± 9.9 *achieved through* per-category attractor collapse (0/5 unique-gen ≥ 30%) |
| 10 | reactions | 0.449 | **FAIL** (3 seeds) | compositional-hierarchy | cat-match 3.4% ± 5.9; bridge generates local SMILES but ~90% lack `>>` separator |

**Net: 5 PASS, 5 FAIL** (6 if RNA's variance-bound is counted as FAIL). The FAILs are not framework failures — they split into three mechanism-distinct classes that reshape §4.3 of the paper.

## Three-branch failure typology

Six of the ten domains FAIL the mandatory diagnostic gate. Those failures sort mechanistically into three branches:

- **Low-SS** (Network 0.126, DNA-coding 0.033). Below the SS ≥ 0.15 threshold. Aggregate cat-match looks OK but per-category signal is absent; the bridge produces alphabet-valid but prompt-insensitive outputs. *Framework prediction: detection-utility only, not generation-utility.* Validated across 2 independent encoding types (digit distributions + nucleotide alphabets).
- **Within-window-attractor** (bioreactor [v1, SS = 0.931], ATC [SS = 0.563]). High or moderate SS but low within-window vocabulary diversity (H_win/H₀ = 0.29 and 0.67). Bridge collapses to per-category prototype sequences; aggregate cat-match *achieved through* mode collapse rather than despite it. MIDI-XL is a second instance at scale: XL-size on the same data collapses where S-size passes.
- **Compositional-hierarchy** (reactions, SS = 0.449). Char-level training can't assemble higher-order structure. Bridge generates plausible molecule-level SMILES tokens but ~90% fail to produce the `reactants>>products` pairing that defines a reaction.

Plus **variance-bound** as a distinct failure mode:

- **RNA** (SS = 0.675). Nine distinct recipe iterations; all FAIL diagnostics, but identical settings produced substantially different unique-gen rates on rerun (34/28/22% → 9/9/22%). The failure is not mode-collapse-consistent — it's non-reproducibility within a single configuration. This motivated the multi-seed reporting rule (see §4.6 of the paper).

See the [`compute_window_entropy.py`](compute_window_entropy.py) script and [`window_entropy_results.json`](window_entropy_results.json) for the H_win/H₀ metric values across all 11 domains (Spearman ρ(H_win/H₀, PASS) = +0.73 on n = 8 subset after excluding variance-bound RNA and legacy SMILES).

## Key methodology: mode-collapse diagnostics

Aggregate cat-match numbers hide mode collapse. **Five of ten PoCs had aggregate cat-match ≥ 30% that turned out to be mode-collapse artefacts on per-category inspection** (Network v1 46%, Python v1 feat-match 9%, ATC 52%, RNA per-seed bests, and reactions elimination-category 13–40%). The methodology lesson is spelled out in [`mode-collapse-diagnostics.md`](mode-collapse-diagnostics.md).

Every new PoC from 2026-04-19 onward runs the mandatory diagnostic suite. Earlier PoCs (SMILES, MIDI) were done pre-gate — their legacy numbers are reported honestly but the formal verdict field is absent. The ten current REPORTs use verdict enum strings from a fixed set: `Stable PASS`, `PASS`, `PASS (Tests A + B)`, `PASS at S / FAIL at XL`, `PASS-legacy`, `FAIL`. FAILs additionally carry a failure-mode string from the three-branch typology above (plus `variance-bound`).

## The recipe that works

From Factor D × Factor B interaction:

- **S-size domain model only.** XL mode-collapses on balanced descriptions — see [`../../archive/xl-reverse-bridge-collapse/`](../../archive/xl-reverse-bridge-collapse/).
- **Fine-tune domain model at 0.1 × bridge LR** (jointly trained).
- **64 soft prefix tokens**, learned from scratch.
- **Semantic pairs, balanced** (BAL-960–2400 per 4–5 categories).
- **Temperature-0.8 sampling, top-k 40** (greedy decoding collapses — see MIDI S-scale: greedy 36% vs sampled 60%).
- **Train 30 epochs max, patience 10** on val loss.
- **Multi-seed reporting** for capability claims (Stable PASS requires ≥ 4/5 seeds PASS AND cat-match SD ≤ 5 pp).
- **Mandatory mode-collapse diagnostics** post-training on held-out eval.

## Multi-seed Stable PASS

Two domains have multi-seed Stable PASS evidence at 5 seeds:

| Domain | cat-match (mean ± SD) | feat-match (mean ± SD) | Verdicts |
|---|---|---|---|
| quantum | 0.704 ± 0.040 | 0.786 ± 0.050 | 5 PASS / 0 FAIL |
| python | 0.416 ± 0.043 | 0.519 ± 0.023 | 5 PASS / 0 FAIL |

Both meet the pre-agreed Stable PASS criterion (≥ 4/5 PASS AND cat-match SD ≤ 5 pp). ATC and reactions ran 3 seeds each under Spec 3 v2; both FAIL on every seed with SDs consistent with their failure mode (ATC: 9.9 pp cross-seed swing, attractor-driven; reactions: 5.9 pp on 3.4% mean, near-floor). G-code multi-seed is deferred (labeller-integration task, ~1 hour).

## Per-PoC structure

Each `{domain}/` directory contains:

- `REPORT.md` — authoritative per-PoC report with honest framing (40–80 lines, Quantum-exemplar shape)
- `decision_checkpoint.json` — corpus build + prevalence stats
- `training_meta.json` — training trace (v2 PoCs only)
- `poc_generated_temp08.jsonl` — held-out generations
- `scorecard_heldout_temp08.json` — per-category + per-feature rates
- `mode_collapse_diagnostics.json` — mandatory diagnostic output (v2 PoCs only)
- `train.log` — training trace
- `roundtrip_rate.json` — labeller round-trip accuracy

## Reproduce one domain

```bash
cd bridges/reverse/quantum
# Training + generation requires A100 (2–5 min training + 1 min eval)
# See protocols/protocol_03_reverse_bridge_training.md for end-to-end
```

Wall-clock on A100: 2–5 min training + 1 min eval. The full 10-domain set completes in ~30 minutes of A100 time plus corpus-building time (labellers are per-domain; see [`../../factor-d-paired-description-quality/labeller_template.md`](../../factor-d-paired-description-quality/labeller_template.md)).

### Re-derive a scorecard from shipped generations (no GPU)

Quantum ships a CPU-only replay scorer at [`quantum/score.py`](quantum/score.py):

```bash
python bridges/reverse/quantum/score.py --compare
```

It re-derives `scorecard_heldout_temp08.json`'s `total`, `per_category`, and
`total_rates` fields from `poc_generated_temp08.jsonl` in under 5 seconds.
The other 9 reverse-bridge PoCs use the same scoring protocol but have not
yet had their private labellers ported — that is tracked as follow-up work.

## How to read a PoC REPORT

1. **Check the Verdict field first** in the Held-out evaluation section. Enum: `Stable PASS`, `PASS`, `PASS (Tests A + B)`, `PASS at S / FAIL at XL`, `PASS-legacy`, `FAIL`.
2. **If FAIL, check the Failure mode field**: `low-SS`, `within-window-attractor`, `compositional-hierarchy`, `variance-bound`.
3. **Read the per-category breakdown table** before the aggregate — attractor collapse hides in the aggregates.
4. **Check unique-generation rate per category**. Any category < 30% is mode-collapse-suspect.
5. **Check whether one category dominates cat-match**. If category A scores 100% and others score 0%, the bridge collapsed to A.

Full protocol: [`mode-collapse-diagnostics.md`](mode-collapse-diagnostics.md).

## References in the paper

- §4.2 Forward bridges — universal construction across 16 domains
- §4.3 Reverse bridges — 10-domain capability demonstration + three-branch failure typology
- §4.4 Four-factor framework — Factor D dominance + Factor A × D and Factor B × D interactions
- §4.5 SS × TRL playbook — two-metric screening (SS + H_win/H₀)
- §4.6 Methodology — mandatory diagnostic gate + multi-seed rule + temperature sampling + category-aware validator
- §5.4 Testable predictions — pre-stated predictions + Prediction 12 (H_win/H₀ validation at n ≥ 6 held-out)
- §6.2 Limitations — single-seed scope for MIDI-S + SMILES-legacy; labeller-design heterogeneity; compositional-hierarchy rescue (Prediction 11) untested
