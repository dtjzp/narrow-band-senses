# dna_coding reverse-bridge PoC REPORT

Generated: 2026-04-19 06:32 (sampled temp=0.8 top-k=40 seed=42)

## Decision checkpoint

- Window size chosen: **200**
- Category distribution: `{with-ORF: 0.536, mid-GC: 0.222, high-GC: 0.183, low-GC: 0.060}`
- Skipped categories (<5%): `[homopolymer-rich]` (defined but 0 prevalence in data)
- Fallback applied: `None`

## Corpus

- Pairs: **1280** (balanced 320 per category × 4; homopolymer-rich dropped)
- Dist after build: `{low-GC: 320, mid-GC: 320, high-GC: 320, with-ORF: 320}`
- Skipped categories at build: `[homopolymer-rich]`
- Round-trip labeller accuracy: **100.00%**

## Training (reverse bridge v2, S-size fine-tuned)

- Best val loss: **1.305**
- Best epoch: **15** (30-epoch max, patience 10, stopped at 25)
- Splits: train=1024 val=128 eval=128

## Held-out evaluation (temperature=0.8, top-k=40)

- Parse success (alphabet-ok): **100.00%** (n=128)
- Category match: **37.5%** (aggregate; mode-collapse artefact — see Notes)
- Feature match (domain-specific): longest-run 100%, has-ORF 54%, GC±5% 27%, top-codon 14%
- **Verdict**: **FAIL**
- Failure mode: **low-SS**

### Per-category breakdown

| Category | n | parse | cat-match | feature-match | unique |
|---|---|---|---|---|---|
| high-GC | 32 | 100% | 100% | GC±5% 50%, top-codon 3%, longest-run 100%, has-ORF 100% | 46.9% |
| with-ORF | 32 | 100% | 41% | GC±5% 28%, top-codon 13%, longest-run 100%, has-ORF 41% | 75.0% |
| mid-GC | 32 | 100% | 0% | GC±5% 22%, top-codon 13%, longest-run 100%, has-ORF 66% | 56.3% |
| low-GC | 32 | 100% | 9% | GC±5% 9%, top-codon 28%, longest-run 100%, has-ORF 9% | 40.6% |

Cross-category leakage to dominant (`high-GC`): **43.0%**. Mode-collapse diagnostic verdict: **FAIL** (intra-cat-Hamming FAIL — high-GC 13.3 below 30 threshold; unique-rate passes).

## Notes

- SS = 0.033 (well below SS ≥ 0.15 detection threshold). Framework predicts detection-utility only — no capability claim expected.
- Diagnostic-gate FAIL: bridge finds 1–2 attractors for high-GC that happen to re-classify correctly; no meaningful conditioning on mid-GC / low-GC. Second low-SS validation of the SS-threshold prediction alongside network.
- Single-seed (seed=42).
- See paper §4.3 and `mode-collapse-diagnostics.md` for cross-PoC context.

_(Original long REPORT archived at `1-research/nbs-bridge/paper-reports/dna_coding/REPORT.md`.)_
