# Semantic Labeller Template

Every per-domain semantic labeller must satisfy the following contract. This template is the spec for a new domain's labeller + verifier pair.

## Input

A tokenised window as a `str` (character-level alphabet chosen per domain) or a list of events (e.g. MIDI note events). The labeller reads the window directly — no model calls, no external APIs, no randomness.

Constraint: the labeller is a pure function of the window. Given the same window twice, it emits the same claim dict.

## Output — claim dict

A dict of structured claims. Each claim is either:

- **categorical** — `"category": "perimeter"`, or `"category": "multiloop"`
- **count** — `"g2g3_arc_count": 7`, `"n_stems": 3`
- **range** — `"x_range_mm": [42.1, 78.3]`, `"max_stem_length": 12`
- **boolean** — `"has_orf": true`, `"uses_arithmetic": false`

One claim must be `category` (used for balanced-corpus stratification). The remaining claims are the per-category feature contract (see below).

## Category

Each window gets exactly one category. Categories are chosen per domain, typically 4–6, and the labeller's rule for assignment is fully deterministic (regex, threshold on a count, etc.). No ties, no "unknown" category.

Per-category feature contract: each category claims a **subset** of the available features as its relevant set. Feature-match is averaged only over the claimed subset. This is the category-aware validator lesson from the Python reverse-bridge PoC (see `../bridges/reverse/mode-collapse-diagnostics.md`).

## Template

A fixed-format natural-language description template that renders the claim dict to a `str`. Example for G-code:

```
A {category} toolpath with {g2g3_arc_count} arc moves and {m204_count}
acceleration changes. X range {x_range_mm[0]}-{x_range_mm[1]}mm,
Y range {y_range_mm[0]}-{y_range_mm[1]}mm, feedrate {feedrate_mean}
{feedrate_unit}. {notes_or_none}.
```

The template must be **deterministic**. No randomised wording, no synonyms. The bridge learns against a fixed surface form and the verifier parses that surface form.

## Verifier

The inverse operation: reads a generated description (that need not follow the template exactly) and extracts claim values using:

1. **AST-based extraction** for code domains (Python) — parse tree walk
2. **Regex extraction with keyword anchors** for numeric/count domains (G-code, SMILES)
3. **Grammar validation** for structured-string domains (RNA, SMILES) — parse-first, extract-second

Verifier outputs a dict with the same keys as the labeller, plus one additional key per claim: `_verified: true | false | "unextractable"`. `unextractable` means the description did not contain enough surface information to evaluate this claim — handle separately from `false`.

## Round-trip accuracy

On a held-out set of `N` windows:

1. Labeller produces claim dict + description for each window.
2. Verifier runs on the description, producing extracted claim dict.
3. Compute `match_rate = (extracted == labelled) / N` per claim.

Target: round-trip `match_rate ≥ 0.95` per claim for the labeller's own outputs.

Round-trip accuracy is **not** a capability number — it's a labeller correctness gate. A labeller that fails its own round-trip cannot be used for bridge training.

## Cross-domain specificity test

Apply the verifier for domain **X** to descriptions from domain **Y**. Match rate should be near-zero (≤ chance, typically 0.05). If cross-domain match is non-negligible, the description set is too generic — this is the natural-pair failure mode.

## Per-domain examples

See `semantic-labellers/` for concrete implementations:

- `gcode_semantic_labeller.py` (template-compliant reference)
- `smiles_semantic_labeller.py`, `midi_semantic_labeller.py`, etc.

And `claim-verifiers/` for the matching verifiers.

## Balance

When used to build a corpus, the labeller is called once per source window. The category distribution across a natural sample is typically skewed (some categories dominate). Balanced corpora (BAL-N) re-sample to equal per-category counts, truncating the dominant category. This is cheap (labeller is O(seconds per window)) and the decisive lesson: balance first, scale second.

## Failure modes to avoid

Pulled from real debugging of the Python reverse-bridge PoC:

- **Single-feature validators**: `has_loop` as the only boolean → scored 0% for arithmetic and collection categories (which correctly don't have loops). Fix: category-aware feature sets.
- **Template mismatch**: labeller emits "range 42-78mm" but verifier expects "range from 42mm to 78mm" → unextractable. Fix: use the same regex in both, ideally shared as a helper.
- **Non-deterministic labeller**: includes `datetime.now()` or a hash with seed variation → round-trip fails non-reproducibly. Fix: pure functions only.
