# Mode-Collapse Diagnostics

**The load-bearing methodology lesson of the reverse-bridge chapter.** Without these diagnostics, three of the eight PoCs would have been mis-reported as showing partial capability. The diagnostic gate forces honest framing.

## Why this exists

Early in the 10-PoC sequence, we reported:

- **Network (SS = 0.126)**: aggregate cat-match 46.4%. Looked like partial capability.
- **Python (SS = 0.523)**: aggregate feat-match 9.38%. Looked like weak feature-recovery.

On critical review:

- Network was **mode collapse**. The bridge generated one canonical "repeated-burst" output for every prompt. 46% of the eval set happened to *be* repeated-burst, giving the misleading aggregate. Per-category: 100% on repeated-burst, 0% on every other category.
- Python had a **single-feature validator** that scored 0% for categories that should legitimately have no loops. Cat-match 41% was real signal; feat-match 9% was a broken validator, not a capability failure.

Both issues were caught only by reading the per-category breakdowns and interrogating the validator. The diagnostics below are the formalised gate that catches them by default.

## The three criteria

For each held-out eval (typically 32-50 generations per category × 4-5 categories), compute:

### 1. Unique-generation rate per category

```python
unique_rate[c] = len(set(generations[c])) / len(generations[c])
```

Generations are compared as strings. Threshold: **≥ 30% unique per category** is soft pass. **< 30%** is FAIL.

If all generations in a category are byte-identical, `unique_rate = 1 / n_per_cat`. This is what happened to RNA's `simple-hairpin` (unique rate = 3.1% = 1/32; all 32 gens were byte-identical).

### 2. Intra-category Hamming distance

For each category:

```python
intra_hamming[c] = mean(
    hamming(g_i, g_j) for g_i, g_j in combinations(generations[c], 2)
)
```

Compared to a **domain-specific threshold**:

- **Small-alphabet domains** (DNA 4-char, RNA 3-char structural): threshold 30
- **Medium-alphabet domains** (G-code, MIDI, Python char-set): threshold 10-20
- **Large-alphabet / code domains** (Python AST-level): threshold 5-10

Intra-Hamming catches "looks diverse by string but really isn't" — e.g. DNA generations differing by 13 chars out of 200 (FAIL at threshold 30).

### 3. Cross-category leakage to dominant classification

For every generation in every category, run the per-domain classifier. Compute:

```python
classified = [classify(g) for c in cats for g in generations[c]]
dominant = mode(classified)
cross_leak = classified.count(dominant) / total
```

If `cross_leak > 0.80`, the bridge is classifying every generation as one dominant category — hard FAIL.

Typical pattern at FAIL: cross-leak 0.50-0.65. Network v2: 60.2% → nine-heavy. RNA: 59.4% → simple-hairpin. DNA coding: 43.0% → high-GC (below hard-FAIL threshold but triggers intra-Hamming FAIL on its own).

## JSON schema

Every PoC from 2026-04-19 writes `mode_collapse_diagnostics.json`:

```json
{
  "per_category": {
    "<cat_name>": {
      "n": 32,
      "unique_rate": 0.78,
      "intra_cat_distance": 23.4,
      "cat_match": 0.62
    },
    ...
  },
  "cross_category_leakage_to_dominant": 0.34,
  "dominant_classification": "<cat_name>",
  "intra_dist_threshold": 10,
  "verdict": "PASS" | "FAIL",
  "failure_reasons": {
    "unique_rate_fail": false,
    "intra_cat_distance_fail": false,
    "high_leak_fail": false
  }
}
```

`verdict = FAIL` if any of the three criteria triggers.

## Worked examples

### RNA (FAIL, simple-hairpin collapse)

```
per_category:
  simple-hairpin: unique=3.1%, intra=0.00, cat-match=100%   # total collapse
  nested-helical: unique=18.8%, intra=126, cat-match=50%
  bulge-rich:     unique=25%, intra=37, cat-match=100%
  multiloop:      unique=25%, intra=72, cat-match=12.5%
cross-leak-to-simple-hairpin: 59.4%
verdict: FAIL (unique_rate_fail AND intra_cat_distance_fail)
```

Lead headline: parse_ok_rate 100% is the real capability (bracket-balance grammar learned). Aggregate cat-match 65.6% is an artefact and **not** reported as capability.

### Network v2 (FAIL, nine-heavy collapse)

```
per_category:
  zero-dense:  unique=6.25%,  intra=0.94, cat-match=100%    # total collapse
  one-heavy:   unique=21.9%, intra=12.8, cat-match=0%
  nine-heavy:  unique=21.9%, intra=6.7,  cat-match=100%
  five-heavy:  unique=34.4%, intra=10.0, cat-match=0%
cross-leak-to-nine-heavy: 60.2%
verdict: FAIL (unique_rate_fail AND intra_cat_distance_fail)
```

The aggregate 50% cat-match is the sum of two attractors (nine-heavy and zero-dense; the bridge's nine-heavy attractor happens to score high zero-density too). Not a capability number.

### Quantum (PASS)

```
per_category:
  1-qubit:    unique=37.5%, intra=18.3, cat-match=56.25%
  entangling: unique=78.1%, intra=42.7, cat-match=84.4%
  deep:       unique=90.6%, intra=61.1, cat-match=87.5%
  random:     unique=93.8%, intra=58.2, cat-match=80.3%
cross-leak-to-dominant: 32%
verdict: PASS
```

Typical high-SS / high-capability pattern: unique rates 78–94% for three categories, 1-qubit drops to 37.5% (short programs repeat naturally — soft pass). Cat-match 77% aggregate is a real capability number because no single category dominates and diversity is real.

## What a FAIL headline means

The verdict is **not** "the paper's framework is wrong." For low-SS domains (Network 0.126, DNA coding 0.033), FAIL *validates* the §4.7 TRL playbook's prediction: reverse bridging at SS < 0.15 does not support generative use at S-size under the standard recipe. Report honestly as framework-validation.

For high-SS domains (RNA 0.675), FAIL is an informative counter-example: SS is **necessary but not sufficient**. Vocabulary dimensionality, category separability, and architecture/corpus/sampling all matter. RNA reruns are ongoing at time of writing.

## References

- Paper §4.5 (reverse-bridge cross-domain synthesis)
- Paper §6.2 (limitations — includes this diagnostic gate)
- Overnight summary v2 — reverse-bridge set to n=8
- Handover spec §3.1 Critical review protocol (methodology lesson)
