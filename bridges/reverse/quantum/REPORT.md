# quantum reverse-bridge PoC REPORT

Generated: 2026-04-18 22:09:54 (5-seed update: 2026-04-19)

## Decision checkpoint

- Window size chosen: **200**
- Category distribution: `{'single-qubit': 0.122, 'measurement-heavy': 0.234, 'entangling': 0.028, 'highly-entangled': 0.482, 'parameterised': 0.134}`
- Skipped categories (<5%): `['entangling']`
- Fallback applied: `None`

## Corpus

- Pairs: **1468**
- Dist after build: `{'measurement-heavy': 320, 'highly-entangled': 320, 'single-qubit': 320, 'parameterised': 320, 'entangling': 188}`
- Skipped categories at build: `[]`
- Round-trip labeller accuracy: **100.00%**

## Training (reverse bridge v2, S-size fine-tuned)

- Best val loss: **0.927** (seed 42)
- Best epoch: **19**
- Splits: train=1174 val=146 eval=148
- Seeds: **5 (42, 7, 1337, 2026, 43)**

## Held-out evaluation (temperature=0.8, top-k=40)

- Parse success: **100.00%**
- Category match: **70.4% ± 4.0** (5-seed mean ± SD)
- Feature match (domain-specific): **78.6% ± 5.0** (5-seed mean ± SD)
- **Verdict**: **Stable PASS**

### Per-category breakdown

| Category | n | parse | cat-match | feature-match | unique |
|---|---|---|---|---|---|
| parameterised | 32 | 100.00% | 68.1% ± 12.0 | 84.4% (s42) | 94.4% ± 2.6 |
| highly-entangled | 32 | 100.00% | 88.1% ± 11.1 | 90.6% (s42) | 90.6% ± 5.4 |
| measurement-heavy | 32 | 100.00% | 85.0% ± 14.6 | 68.8% (s42) | 76.3% ± 4.7 |
| entangling | 20 | 100.00% | 28.0% ± 12.0 | 70.0% (s42) | 89.0% ± 2.2 |
| single-qubit | 32 | 100.00% | 66.9% ± 30.2 | 93.8% (s42) | 41.3% ± 2.6 |

## Notes

- S-size domain model only (no XL). Top-k 40, temp 0.8.
- 5-seed means/SDs from `1-research/nbs-bridge/multi-seed-report.md` §3.1; feature-match column reports seed-42 value (per-seed feature breakdown not exported in multi-seed driver).
- Stable PASS per Spec 2 rubric (cat-match SD ≤ 5pp, 5/5 seeds PASS). Single-qubit unique-gen 41% soft-passes per Spec 2 §5.1 (short gate-heavy programs repeat naturally at n=20).
- See paper §4.3 and `mode-collapse-diagnostics.md` for cross-PoC context.