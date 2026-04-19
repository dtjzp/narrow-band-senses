# rna reverse-bridge PoC REPORT

Generated: 2026-04-19 06:27 (sampled temp=0.8 top-k=40 seed=42)

## Decision checkpoint

- Window size chosen: **200**
- Category distribution: `{nested-helical: 0.596, bulge-rich: 0.214, simple-hairpin: 0.108, multiloop: 0.082}` (after `<>`→`()` normalisation + pseudoknot filter)
- Skipped categories (<5%): `[]` (spec default had `complex-stem-loop` at 1.2% — dropped pre-DC)
- Fallback applied: `None`

## Corpus

- Pairs: **1280** (balanced 320 per category × 4)
- Dist after build: `{nested-helical: 320, bulge-rich: 320, simple-hairpin: 320, multiloop: 320}`
- Skipped categories at build: `[]`
- Round-trip labeller accuracy: **100.00%**

## Training (reverse bridge v2, S-size fine-tuned)

- Best val loss: **0.497**
- Best epoch: **22** (30-epoch max, patience 10; fresh 3-epoch S-model retrain — no prior `rna_S_s42.pt`)
- Splits: train=1024 val=128 eval=128

## Held-out evaluation (temperature=0.8, top-k=40)

- Parse success (bracket balance): **100.00%** (n=128)
- Category match: **65.6%** (aggregate, mode-collapse artefact — see Notes)
- Feature match (domain-specific): nesting-depth 55%, n-stems 45%, max-loop-size 41%, max-stem-length 23%, n-paired 23%
- **Verdict**: **FAIL**
- Failure mode: **variance-bound**

### Per-category breakdown

| Category | n | parse | cat-match | feature-match | unique |
|---|---|---|---|---|---|
| simple-hairpin | 32 | 100% | 100% | n-paired 69%, n-stems 100%, max-stem-length 66%, nesting-depth 97% | **3.1%** |
| nested-helical | 32 | 100% | 50% | max-loop-size 69% | 18.8% |
| bulge-rich | 32 | 100% | 100% | max-loop-size 3%, n-stems 38% | 25.0% |
| multiloop | 32 | 100% | 13% | nesting-depth 100%, max-loop-size 66% | 25.0% |

Cross-category leakage to dominant (`simple-hairpin`): **59.4%**. Mode-collapse diagnostic verdict: **FAIL** (unique-rate FAIL 3/4, intra-cat-Hamming FAIL — simple-hairpin dist=0.0).

## Notes

- Diagnostic-gate FAIL: the aggregate 65.6% cat-match is a mode-collapse artefact — simple-hairpin's 32 generations are byte-identical (dist 0.0, unique 3.1%). Parse-level capability (100% WUSS bracket balance across 200-char windows with nested stems) is real — long-range syntactic dependency is recovered.
- Failure mode **variance-bound**: RNA is the motivating case for the Spec 2 multi-seed reproducibility rule (`multi-seed-report.md` TL;DR; paper §4.6). The simple-hairpin attractor is a local minimum the fine-tune does not reliably escape across runs.
- Single-seed gate evaluation; multi-seed re-runs planned but not in this PoC's data. SS = 0.675 (third-highest in survey) — first high-SS domain to FAIL the diagnostic gate.
- See paper §4.3 and `mode-collapse-diagnostics.md` for cross-PoC context.

_(Original long REPORT archived at `1-research/nbs-bridge/paper-reports/rna/REPORT.md`.)_
