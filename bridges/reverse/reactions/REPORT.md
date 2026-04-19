# reactions reverse-bridge PoC REPORT

Generated: 2026-04-19 17:46 (Spec 3 v2 multi-seed)

## Decision checkpoint

- Window size chosen: **400**
- Category distribution: `{elimination: 0.286, substitution: 0.286, rearrangement: 0.286, addition: 0.143}`
- Skipped categories (<5%): `[]`
- Fallback applied: `None`

## Corpus

- Pairs: **1280** (balanced 320 per category × 4)
- Dist after build: `{addition: 320, elimination: 320, substitution: 320, rearrangement: 320}`
- Skipped categories at build: `[]`
- Round-trip labeller accuracy: **100.00%**

## Training (reverse bridge v2, S-size fine-tuned)

- Best val loss: **0.674** (seed 42)
- Best epoch: **28**
- Splits: train=1024 val=128 eval=128
- Seeds: **3 (42, 7, 1337)**

## Held-out evaluation (temperature=0.8, top-k=40)

- Parse success: **4.4% ± 5.1** (3-seed mean ± SD; per-seed 0.102 / 0.000 / 0.031) — reaction-SMILES format; most generations malformed
- Category match: **3.4% ± 5.9** (3-seed mean ± SD; per-seed 0.102 / 0.000 / 0.000)
- Feature match (domain-specific): not applicable — bridge fails to emit well-formed reaction SMILES
- **Verdict**: **FAIL**
- Failure mode: **compositional-hierarchy**

### Per-category breakdown

| Category | n | parse | cat-match | feature-match | unique |
|---|---|---|---|---|---|
| addition | 32 | ~0% | 0.000 ± 0.000 | — | 8.3% ± 4.8 |
| elimination | 32 | 13-40% | 0.135 ± 0.235 | — | 15.6% ± 10.8 |
| rearrangement | 32 | ~0% | 0.000 ± 0.000 | — | 15.6% ± 9.4 |
| substitution | 32 | ~0% | 0.000 ± 0.000 | — | 8.3% ± 4.8 |

Per-seed diagnostic verdicts: **FAIL / FAIL / FAIL**. Prediction check: cat-match ≥ 0.45 NOT MET; ≥ 3/4 categories unique-gen ≥ 0.30 NOT MET (0/4 meet).

## Notes

- SS = 0.449. Char-level S-model cannot assemble reaction SMILES (`A.B>>C.D`) — the reaction grammar requires composing valid reactant SMILES, a `>>` separator, and valid product SMILES. The bridge produces near-zero parseable reaction strings except for a fraction of `elimination` (seed 42 reaches 40.6% parse there).
- Failure mode **compositional-hierarchy**: char-level training cannot assemble the higher-order structure (valid SMILES × 2 composed via `>>`). This is distinct from within-window-attractor (parse succeeds, diversity fails): here parse itself fails because the required structure spans multiple sub-sequences the S-model cannot jointly model.
- See paper §4.3 and `spec3-v2-report.md` §7–8 for full context.
