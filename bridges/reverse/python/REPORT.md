# python_code reverse-bridge PoC REPORT

Generated: 2026-04-18 22:03:56 (validator-v2 correction: 2026-04-19; 5-seed update: 2026-04-19)

## Decision checkpoint

- Window size chosen: **400**
- Category distribution: `{'arithmetic': 0.628, 'loop': 0.134, 'complex': 0.114, 'collection': 0.066, 'conditional': 0.058}`
- Skipped categories (<5%): `[]`
- Fallback applied: `window_size_retried_picked_400` (default 200 had a dominant category)

## Corpus

- Pairs: **1600**
- Dist after build: `{'arithmetic': 320, 'loop': 320, 'complex': 320, 'collection': 320, 'conditional': 320}`
- Skipped categories at build: `[]`
- Round-trip labeller accuracy: **100.00%** (n=1600, trivially 100% by construction)

## Training (reverse bridge v2, S-size fine-tuned)

- Best val loss: **1.618** (seed 42; still improving at cap)
- Best epoch: **20**
- Splits: train=1280 val=160 eval=160
- Seeds: **5 (42, 7, 1337, 2026, 43)**

## Held-out evaluation (temperature=0.8, top-k=40)

- Parse success: **100.00%**
- Category match: **41.6% ± 4.3** (5-seed mean ± SD)
- Feature match (v2 validator): **51.9% ± 2.3** (5-seed mean ± SD); seed-42 single-run 43.4%
- **Verdict**: **Stable PASS**

### Per-category breakdown

| Category | n | parse | cat-match | feature-match | unique |
|---|---|---|---|---|---|
| arithmetic | 32 | 100.00% | 70.0% ± 16.6 | 60.0% (s42, v2) | 97.5% ± 4.1 |
| collection | 32 | 100.00% | 46.3% ± 14.0 | 66.4% (s42, v2) | 96.9% ± 3.8 |
| complex | 32 | 100.00% | 33.8% ± 19.1 | 3.1% (s42, v2) | 100.0% ± 0.0 |
| conditional | 32 | 100.00% | 25.0% ± 8.0 | 45.0% (s42, v2) | 100.0% ± 0.0 |
| loop | 32 | 100.00% | 33.1% ± 13.5 | 42.7% (s42, v2) | 100.0% ± 0.0 |

## Notes

- S-size domain model only (no XL). Top-k 40, temp 0.8. Cat-match 41.6% is 2× the 5-category chance baseline (20%).
- 5-seed means/SDs from `1-research/nbs-bridge/multi-seed-report.md` §3.2; feature-match column reports seed-42 v2-validator value per category (validator: `verify_python_features.py`). v1 feat-match (9.4%) retracted — single-feature `has_loop`-only scoring mis-counted absence.
- Bridge is better at absence-signals than presence-signals: arithmetic/collection recover "no loop / no try-except" at 90–100%; loop presence 28%, conditional presence 9%. Strict AST parse rate is near-zero because 400-char windows are not standalone-parseable — validator uses AST-first / regex-fallback hybrid.
- See paper §4.3 and `mode-collapse-diagnostics.md` for cross-PoC context.
