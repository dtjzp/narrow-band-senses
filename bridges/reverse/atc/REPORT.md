# atc reverse-bridge PoC REPORT

Generated: 2026-04-19 (Spec 3 v2 multi-seed)

## Decision checkpoint

- Window size chosen: **200**
- Category distribution: `{climbing: 0.283, descending: 0.283, mid-cruise: 0.259, low-altitude: 0.093, high-cruise: 0.081}`
- Skipped categories (<5%): `[]`
- Fallback applied: `None`

## Corpus

- Pairs: **1353** (balanced where possible; high-cruise 184, low-altitude 209, others 320)
- Dist after build: `{climbing: 320, descending: 320, mid-cruise: 320, low-altitude: 209, high-cruise: 184}`
- Skipped categories at build: `[]`
- Round-trip labeller accuracy: **100.00%**

## Training (reverse bridge v2, S-size fine-tuned)

- Best val loss: **0.398** (seed 42)
- Best epoch: **24**
- Splits: train=1082 val=134 eval=137
- Seeds: **3 (42, 7, 1337)**

## Held-out evaluation (temperature=0.8, top-k=40)

- Parse success: **100.00%** (all seeds)
- Category match: **52.1% ± 9.9** (3-seed mean ± SD; per-seed 53.3 / 41.6 / 61.3)
- Feature match (domain-specific): not applicable — ATC labeller category-only
- **Verdict**: **FAIL**
- Failure mode: **within-window-attractor**

### Per-category breakdown

| Category | n | parse | cat-match | feature-match | unique |
|---|---|---|---|---|---|
| climbing | 32 | 100% | 0.31 / 0.00 / 1.00 | — | 10.4% (mean) |
| mid-cruise | 32 | 100% | 1.00 / 0.97 / 0.00 | — | 5.2% (mean) |
| high-cruise | 19 | 100% | 1.00 / 1.00 / 1.00 | — | 5.3% (mean) |
| low-altitude | 22 | 100% | 0.41 / 0.00 / 0.05 | — | 12.1% (mean) |
| descending | 32 | 100% | 0.09 / 0.22 / 1.00 | — | 8.3% (mean) |

Per-seed diagnostic verdicts: **FAIL / FAIL / FAIL**. Prediction check: cat-match ≥ 0.50 MET; ≥ 3/5 categories unique-gen ≥ 0.30 NOT MET (0/5 meet).

## Notes

- SS = 0.563 (mid-high). Categories swing between 0 and 1 cat-match across seeds — the bridge finds a single intra-window attractor per category and whether it re-classifies correctly is a near-binary function of the seed.
- All 3 seeds FAIL the unique-gen diagnostic (max 18%, mean ~8%). Cat-match cross-seed SD is 9.9pp — above the 5pp Stable-PASS cutoff.
- Failure mode **within-window-attractor**: the bridge collapses each category to a small set of prototype windows; prompt-conditional diversity is absent despite the nominal 52% aggregate cat-match.
- See paper §4.3 and `spec3-v2-report.md` §1–6 for full context.
