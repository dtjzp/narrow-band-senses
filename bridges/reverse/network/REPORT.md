# network reverse-bridge PoC REPORT

Generated: 2026-04-19 05:58 (v2 rerun; sampled temp=0.8 top-k=40 seed=42)

## Decision checkpoint

- Window size chosen: **200**
- Category distribution: `{nine-heavy: 0.328, zero-dense: 0.245, five-heavy: 0.216, one-heavy: 0.157, other: 0.053}` (digits 0-9, data-driven redefinition)
- Skipped categories (<5%): `[]` (v1 letter-encoding categories invalid — dropped pre-DC)
- Fallback applied: `None`

## Corpus

- Pairs: **1280** (balanced 320 per category × 4)
- Dist after build: `{zero-dense: 320, nine-heavy: 320, five-heavy: 320, one-heavy: 320}`
- Skipped categories at build: `[]`
- Round-trip labeller accuracy: **100.00%**

## Training (reverse bridge v2, S-size fine-tuned)

- Best val loss: **0.630**
- Best epoch: **16** (30-epoch max, patience 10, stopped at 26)
- Splits: train=1024 val=128 eval=128

## Held-out evaluation (temperature=0.8, top-k=40)

- Parse success: **100.00%** (n=128)
- Category match: **50.0%** (aggregate; mode-collapse artefact — see Notes)
- Feature match (domain-specific): feature-zero 85.2%, feature-topnz 37.5%
- **Verdict**: **FAIL**
- Failure mode: **low-SS**

### Per-category breakdown

| Category | n | parse | cat-match | feature-match | unique |
|---|---|---|---|---|---|
| zero-dense | 32 | 100% | 100% | feature-zero 100%, feature-topnz 50% | **6.3%** |
| nine-heavy | 32 | 100% | 100% | feature-zero 100%, feature-topnz 100% | 21.9% |
| five-heavy | 32 | 100% | 0% | feature-zero 100%, feature-topnz 0% | 34.4% |
| one-heavy | 32 | 100% | 0% | feature-zero 41%, feature-topnz 0% | 21.9% |

Cross-category leakage to dominant (`nine-heavy`): **60.2%**. Mode-collapse diagnostic verdict: **FAIL** (unique-rate FAIL 4/4, intra-cat-Hamming FAIL — zero-dense dist 0.94, nine-heavy 6.7).

## Notes

- SS = 0.126 (below the SS ≥ 0.15 detection threshold). Framework predicts detection-utility only — no capability claim expected at this SS.
- Diagnostic-gate FAIL: 50% aggregate cat-match is mode-collapse — bridge has 1–2 attractors per category that happen to re-classify correctly for zero-dense/nine-heavy. No prompt-conditional diversity on five-heavy/one-heavy. Same pattern as v1 (archived in `v1-collapsed/`), even with v2 corrections (window 50→200, data-driven categories, balanced corpus).
- Single-seed (seed=42). Network is a framework-prediction-validation PoC — the failure confirms SS-threshold prediction, not a capability miss.
- See paper §4.3 and `mode-collapse-diagnostics.md` for cross-PoC context.

_(Original long REPORT archived at `1-research/nbs-bridge/paper-reports/network/REPORT.md`.)_
