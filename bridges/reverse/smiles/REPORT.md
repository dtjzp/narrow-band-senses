# smiles reverse-bridge PoC REPORT

Generated: 2026-04-18 (sampled temp=0.8 top-k=40 seed=42)

## Decision checkpoint

- Window size chosen: **(SMILES character-level; smiles_semantic_balanced_1920)**
- Category distribution: `{aromatic-simple, heterocyclic, functional-rich, complex-polycyclic}` — 4 balanced categories
- Skipped categories (<5%): `[]`
- Fallback applied: `None`

## Corpus

- Pairs: **1920** (balanced 480 per category × 4)
- Dist after build: `{aromatic-simple: 480, heterocyclic: 480, functional-rich: 480, complex-polycyclic: 480}`
- Skipped categories at build: `[]`
- Round-trip labeller accuracy: **100.00%**

## Training (reverse bridge v2, S-size fine-tuned)

- Best val loss: **0.965**
- Best epoch: **28** (38-epoch training)
- Splits: train=1536 val=192 eval=192

## Held-out evaluation (temperature=0.8, top-k=40)

- Parse success: **78.9%** (n=384) — strict RDKit parse with longest-parseable-prefix walker
- Category match: **31.8%**
- Feature match (domain-specific): **45.1%** overall-fidelity aggregate (heteroatom-subset 40.6%, ring-count 32.6%, aromatic-ring-count 37.2%, functional-group 25.9%, MW 11.2%, logP 18.5%, H-donor 61.7%, H-acceptor 42.4%, heavy-atoms 14.3%)
- **Verdict**: **PASS-legacy**

### Per-category breakdown

| Category | n | parse | cat-match | feature-match | unique |
|---|---|---|---|---|---|
| aromatic-simple | 110 | 82% | 29% | rings 45%, aromatic-rings 48%, hetero-subset 58%, H-donor 74% | — |
| heterocyclic | 79 | 84% | 39% | rings 30%, functional-group 32%, H-donor 66% | — |
| functional-rich | 88 | 91% | 55% | rings 53%, aromatic-rings 57%, H-acceptor 49% | — |
| complex-polycyclic | 107 | 63% | 10% | rings 4%, heavy-atoms 6%, H-donor 44% | — |

Unique canonical SMILES: **290/303 = 95.7%** (structural diversity).

## Notes

- S-size domain model only. Top-k 40, temp 0.8, seed 42. Greedy decoding = 0% parse; temperature sampling is load-bearing.
- **PASS-legacy**: this PoC pre-dates the mandatory mode-collapse diagnostic gate (introduced in the RNA spec). Clears spec-own parse/sanitisation/diversity targets but not aggregate cat-match ≥ 45%; functional-rich alone reaches 55% cat-match. Not re-gated against the later diagnostic suite.
- Single-seed (seed=42); multi-seed not run. Numeric precision (MW 11%, logP 19%) is weak, which is consistent with framework expectation at S-scale.
- See paper §i 4.3 and `mode-collapse-diagnostics.md` for cross-PoC context.

_(Original long REPORT archived at `1-research/nbs-bridge/paper-reports/smiles/REPORT.md`.)_
