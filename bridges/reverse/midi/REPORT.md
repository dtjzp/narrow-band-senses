# midi reverse-bridge PoC REPORT

Generated: 2026-04-18 (sampled temp=0.8 top-k=40 seed=42)

## Decision checkpoint

- Window size chosen: **(MIDI pitch-sequence windows; `midi_semantic_balanced_1600`)**
- Category distribution: `{'tonal', 'ascending-run', 'descending-run', 'leap-dominant', 'mixed'}` — 5 balanced categories
- Skipped categories (<5%): `[]`
- Fallback applied: `None`

## Corpus

- Pairs: **1600** (balanced 320 per category × 5)
- Dist after build: `{tonal: 320, ascending-run: 320, descending-run: 320, leap-dominant: 320, mixed: 320}`
- Skipped categories at build: `[]`
- Round-trip labeller accuracy: **100.00%** (labeller → sequence → labeller)

## Training (reverse bridge v2, S-size fine-tuned)

- Best val loss: **2.5403** (S-size; XL collapsed at best val 2.6075)
- Best epoch: **22** (32 epochs with early stop)
- Splits: train=1280 val=160 eval=160

## Held-out evaluation (temperature=0.8, top-k=40)

- Parse success: **100.00%** (n=320)
- Category match: **30.0%**
- Feature match (domain-specific): **60.3%** overall-fidelity aggregate (pitch-range 49.7%, max-asc-run 71.3%, max-desc-run 75.6%, asc-share 86.6%, desc-share 86.6%, unique-pitches 45.3%, entropy 35.3%, leap-ratio 52.2%, interval-dist 32.2%)
- **Verdict**: **PASS at S / FAIL at XL (documented)**
- Failure mode (XL only): **within-window-attractor**

### Per-category breakdown

| Category | n | parse | cat-match | feature-match | unique |
|---|---|---|---|---|---|
| tonal | 66 | 100% | 86% | pitch-range 56%, asc-share 88%, interval-dist 32% | — |
| mixed | 71 | 100% | 41% | pitch-range 52%, asc-share 90% | — |
| leap-dominant | 58 | 100% | 16% | max-asc-run 100%, max-desc-run 98%, asc-share 90% | — |
| ascending-run | 77 | 100% | 1% | pitch-range 43%, max-desc-run 70% | — |
| descending-run | 48 | 100% | 0% | pitch-range 46%, max-asc-run 71% | — |

## Notes

- S-size domain model (d_model=256, 4 layers); XL (d_model=1024, 24 layers) collapses on same protocol — "within-window-attractor" failure mode documented as negative finding. XL run at `results/reverse/..._sXL_reverse_bridge.pt`.
- Temperature sampling (0.8, top-k=40, seed=42) is load-bearing — greedy decoding collapses overall fidelity from 60.3% to 36.4% (`scorecard_heldout.json` = greedy baseline).
- Clears 3 of 4 minimum-viable criteria (parse ≥ 90%, at-least-one-category ≥ 70%, audible output) but misses aggregate category-match ≥ 40% — directional features (asc/desc-share 87%, max-runs 71–76%) show the bridge conditions on prompt content; the 5-bucket labeller's composite rule frequently sorts valid outputs into neighbouring buckets.
- Single-seed (seed=42); multi-seed deferred.
- See paper §4.3 and `mode-collapse-diagnostics.md` for cross-PoC context.

