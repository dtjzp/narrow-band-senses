# gcode reverse-bridge PoC REPORT

Generated: 2026-04-18 (Tests A + B: 2026-04-19)

## Decision checkpoint

- Window size chosen: **(labeller-template; balanced corpus bal2400)**
- Category distribution: `{extrusion, travel, mixed, retraction, accel}` — 5 categories, balanced 2400 pairs source
- Skipped categories (<5%): `[]`
- Fallback applied: `None`

## Corpus

- Pairs: **2400** (source) → n=329 held-out eval from val+eval splits
- Dist after build: `{extrusion: 78, travel: 5, mixed: 78, retraction: 81, accel: 87}` (held-out n=329)
- Skipped categories at build: `[]`
- Round-trip labeller accuracy: **100.00%** (template-regex labeller on its own template output)

## Training (reverse bridge v2, S-size fine-tuned)

- Checkpoint: **BAL-2400 S reverse v2, n_soft=64, d_out=256** (fine-tune, S-model at 0.1× bridge LR)
- Best val loss / epoch: not persisted (G-code run pre-dates v2 protocol's mandatory `training_meta.json`)
- Splits: train=1920 val=240 eval=240 (from source 2400)

## Held-out evaluation (temperature=0.8, top-k=40)

- Parse success: **100.00%** (bridge-only, n=329)
- Category match: **55.0%** (bridge-only, n=329; vs 20% chance)
- Feature match (domain-specific): **65.1%** (command-count ±2, bridge-only, n=329)
- **Verdict**: **PASS (Tests A + B)**

### Per-category breakdown

| Category | n | parse | cat-match | feature-match | unique |
|---|---|---|---|---|---|
| extrusion | 78 | 100% | 14% | 100% (cmd ±2) | — |
| travel | 5 | 100% | 100% | 100% | — |
| mixed | 78 | 100% | 22% | 56% (cmd ±2) | — |
| retraction | 81 | 100% | 93% | 78% (cmd ±2) | — |
| accel | 87 | 100% | 84% | 28% (cmd ±2) | — |

## Notes

- S-size domain model only. Top-k 40, temp 0.8. Single-seed (seed 42); G-code multi-seed deferred to next session (labeller-integration pending per `multi-seed-report.md` §5.1).
- **Test A (source-fidelity on n=329)**: bridge-only vs claim-only M1 XY-shape Fréchet — bridge wins on **67.8%** of prompts (mean 15.16 vs 15.54). Non-saturable metric the claim-fill baseline cannot reach. See `scorecard_heldout.json` `"source_fidelity"` and `1-research/nbs-bridge/poc/scorecard_paraphrase.json`.
- **Test B (paraphrase robustness, n=45 paraphrased prompts)**: bridge-only trajectory-match **40%** vs regex-extractor **22%**; hybrid-LLM reaches 100%. Bridge's GPT-2 encoder gives paraphrase invariance the template-regex lacks.
- Hybrid "100%" numbers (claim-only + synthesiser) are largely tautological by construction and are NOT the bridge-capability claim; cite Tests A + B instead. See `scorecard.json`, `scorecard_paraphrase.json`.
- See paper §4.3 and `mode-collapse-diagnostics.md` for cross-PoC context.

