# Protocol 1 — Entropy Criterion: Structure Score Measurement

**Author**: Daniel Ziekenoppasser-Powell (Independent researcher, UK)
**Keywords**: information theory, narrow-band senses, entropy, structure score, machine learning, multimodal
**Licence**: CC-BY 4.0
**Intended DOI**: `10.17504/protocols.io.[TBD]` (issued on publish)
**Companion paper**: Ziekenoppasser-Powell 2026, *Nature Machine Intelligence* (under review) / arXiv preprint [TBD]

## Abstract

Compute a character-level structure score (SS) that predicts how compressible a tokenised data stream is for a standard neural language model. SS is computed offline in seconds without neural-network training, yet correlates with small-Transformer bits-per-character at Spearman ρ = −0.92 across 29 diverse domains. This protocol documents the canonical pipeline: input preparation, entropy estimation (Miller–Madow for unigrams, pooled n-gram counts for 3rd-order conditional), structure-score computation, and validation against a shuffled baseline.

**Intended audience**: researchers studying domain-specific compressibility, information-theoretic properties of non-linguistic signals, or multimodal AI representations.

## Guidelines and warnings

- The structure score is computed on a **character-level** tokenisation. Sub-word or byte-pair encoding changes SS values and breaks cross-domain comparison. Canonical encodings per domain are documented in the companion repository.
- Alphabet size strongly influences `H_0`. Reported SS values are stable across alphabet sizes from 4 to ~100, but cross-domain comparisons must use the same `n_chars` budget or apply Miller–Madow bias correction.
- Data windowing matters: overlapping n-gram counts within sequences; pooled counting across sequences (no cross-sequence concatenation). This preserves sequence-integrity for domains where sequence boundaries are meaningful (MIDI files, molecular sequences).
- This is not a "language complexity" metric. A natural-language text and a MIDI file can have similar SS without being linguistically comparable in any sense.

## Materials

### Hardware

- Modern laptop CPU. No GPU required. ~1–2 GB RAM sufficient for 1M-character corpora.

### Software

- Python 3.10 or later
- NumPy 1.24+ (for pooled counting)
- (optional) `ndd` 1.1.0 Python package for NSB entropy estimator — MI plots only, not required for SS
- (optional) Matplotlib 3.8+ for the validation figure

### Code

- `entropy.py` from the companion repository: canonical pure functions for unigram H₀ (Miller–Madow) and k-th order conditional H_k via pooled n-grams
- Per-domain tokenisation scripts: see repository `factor-a-domain-structure/per-domain-scripts/`

### Datasets

Per-domain canonical corpora are listed in `factor-a-domain-structure/data/SOURCES.md`. 29 sources spanning:

- Molecular biology: NCBI RefSeq, Rfam, UniProt, ZINC
- Sensor data: OpenBCI EEG, CICIDS network flows, IRIS seismological
- Symbolic: ClassicalPianoMIDI, Project Gutenberg, OpenQASM 3
- Numerical: UK Tidal Service, FAA ATC, NOAA ISD

Each has its own citation and (where required) registration.

## Before you begin

1. Clone the companion repository:
   ```bash
   git clone https://github.com/<user>/nbs-bridge-public
   cd nbs-bridge-public/factor-a-domain-structure
   pip install -r requirements.txt
   ```
2. Download per-domain raw data as documented in `data/SOURCES.md`. Some corpora require academic-use registration; budget ~30 min for this step.
3. Verify `entropy.py` tests pass (if any are shipped): `python -m pytest` from the `../nbs-experiment/` directory.

## Procedure

### Step 1 — Encode domain data to a character stream

Each domain has a canonical encoding from raw data to a printable-ASCII character stream:

- **MIDI**: note events encoded as `chr(33 + note % 94)` producing a ~60-char alphabet
- **SMILES**: strip stereo annotations, keep the structural string; ~40-char alphabet
- **DNA / RNA**: direct character stream (4 chars / 4 chars)
- **G-code**: raw G-code text, alphabet of ~30 chars
- **(Numerical domains — tidal, financial, SETI, weather)**: quantise into a discrete alphabet per domain's typical dynamic range

See the per-domain script for the exact encoding recipe.

```bash
python per-domain-scripts/midi_entropy.py  # produces results/midi_entropy.json
```

### Step 2 — Count unigrams and 3-grams (pooled)

For a stream `x` of length `N`:

- `C_1[s] = count of character s`
- `C_3[abc] = count of trigram abc`
- Pool counts across sequences if you have multiple sequences (do not concatenate).

Code reference: `entropy.compute_ngram_counts_pooled`.

### Step 3 — Compute Miller–Madow unigram entropy H₀

```python
from entropy import compute_entropy_miller_madow

H0 = compute_entropy_miller_madow(C_1, N)  # bits
```

Miller–Madow corrects the plug-in entropy estimator for finite-sample bias. For alphabets of ~10–100 characters at `N = 1e5`, the correction is small (~0.05 bits) but non-negligible.

### Step 4 — Compute 3rd-order conditional entropy H₃

```python
from entropy import compute_conditional_entropy_pooled

H3 = compute_conditional_entropy_pooled(C_3, C_2, N)  # bits
```

`H_3` represents the entropy of the 4th character given the preceding 3 characters.

### Step 5 — Compute structure score SS

```python
SS = 1 - H3 / H0
```

`SS ∈ [0, 1]` (modulo estimator bias). `SS = 0` means "no predictive structure beyond character frequencies." `SS = 1` means "perfectly deterministic given a short prefix."

### Step 6 — Emit per-domain JSON record

```json
{
  "domain": "midi",
  "h0": 5.872,
  "h3": 1.341,
  "structure_score": 0.772,
  "n_chars": 785000,
  "alphabet": 60
}
```

### Step 7 — Shuffled baseline validation

For each domain, compute SS on a randomly-permuted version of the character stream. Expected: SS should collapse to near-zero (≤ 0.05), confirming that SS measures sequential structure, not the marginal character distribution.

```python
import numpy as np
permuted = ''.join(np.random.permutation(list(stream)))
SS_shuffled = compute_structure_score(permuted)
assert SS_shuffled < 0.05
```

### Step 8 — Aggregate across domains

Concatenate per-domain JSON records into `canonical_training_entropy.json`. This is the authoritative per-domain SS file.

### Step 9 — Fit small Transformer for BPC (optional, for ρ validation)

To validate ρ(SS, BPC) = −0.92 against your own data, train a small Transformer (6-layer, 512-dim, ~50M params) on each domain's character stream for 20 epochs; record best validation BPC. Wall-clock: ~2.5 min per domain on A100 or ~30 min per domain on laptop CPU.

This is the path for "from-scratch" replication. Alternatively, the canonical repository ships a pre-computed `bpc_per_domain.json` in its Zenodo deposit.

### Step 10 — Compute Spearman ρ and confidence interval

```python
from scipy.stats import spearmanr
rho, p = spearmanr(ss_values, bpc_values)
```

Expected: `rho ≈ −0.92`, `p < 10⁻¹²` for `n ≥ 29`.

### Step 11 — Reproduce the headline figure

```bash
python factor-a-domain-structure/reproduce_rho.py
```

Writes `paper-figures/fig_main_ss_correlation.png` + `.pdf`.

### Step 12 — Leave-one-out sensitivity analysis

Drop each domain in turn; compute ρ on the remaining n−1. The range of ρ values is the sensitivity to any single domain. Expected: ρ ∈ [−0.95, −0.88] after leave-one-out (i.e. the effect is not driven by any single outlier).

## Expected results

- Spearman ρ(SS, BPC) ≈ **−0.92** with p < 10⁻¹² on n = 29 domains.
- Shuffled-baseline SS near 0 for every domain.
- Per-domain SS range: 0.001 (SETI) to 0.773 (whale).
- Leave-one-out stability: ρ range within ±0.05 of the full-n value.

## Troubleshooting

- **Unusually high SS on a synthetic domain**: check that your stream isn't inadvertently deterministic (e.g. a test generator leaking a known pattern). A synthetic random source should produce SS ≈ 0.
- **Unexpectedly low SS on a structured domain**: check the character encoding. If the raw data is tokenised at a sub-character level (BPE), the structure score is not what the paper measures.
- **Spearman ρ much weaker than −0.9**: verify per-domain BPC was measured at the same training budget. BPC is sensitive to training length; under-trained fits produce noisy BPC values that weaken the correlation.
- **Miller–Madow correction diverges**: happens if your alphabet size approaches `N` (too little data for the alphabet). Use a larger corpus, minimum recommended 20,000 characters.

## References

- Shannon 1948, *Bell System Technical Journal* — information theory and entropy
- Miller 1955, "Note on the bias of information estimates" — the Miller–Madow correction
- Ziekenoppasser-Powell 2026 — the NBS paper (forthcoming)

## Acknowledgements

This protocol is the canonical structure-score pipeline. The Miller–Madow correction and pooled-counting design choices were informed by extensive work on entropy estimation in molecular biology and information theory; see references.

No specific funding is acknowledged for this protocol; the underlying research was supported by Colab Pro+ compute credits (self-funded).
