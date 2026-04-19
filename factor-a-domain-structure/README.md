# Factor A — Domain Structure

**Definition**: the intrinsic redundancy of a tokenised signal, measured as a character-level structure score.

The structure score is cheap (seconds on a laptop), deterministic (no neural training involved), and fully specified by two ideas: plug-in Miller–Madow entropy of character unigrams, and 3rd-order conditional entropy computed via pooled n-gram counts.

$$
\text{SS}(x) \;=\; 1 \;-\; \frac{H_3(x)}{H_0(x)}
$$

where `H_0` is the Miller–Madow plug-in estimator of unigram entropy and `H_3` is the 3rd-order conditional entropy. Both are in bits. SS is bounded in `[0, 1]` (modulo estimator bias), with 0 meaning "no predictive structure beyond character frequencies" and 1 meaning "perfectly deterministic given a short prefix."

## Headline result

**Spearman ρ(SS, bits-per-character of a small Transformer) = −0.92 (p < 10⁻¹², n = 29 domains).**

The structure score predicts neural compressibility across 29 domains spanning seven modality classes — molecular biology (DNA, protein, SMILES), music (MIDI), navigation (tidal, ATC), biological sensing (EEG, hormonal, whale calls), financial time series, astronomy (SETI), traffic, network flows, and quantum circuits. This is the paper's central empirical contribution.

Figure: [`../paper-figures/fig_main_ss_correlation.png`](../paper-figures/fig_main_ss_correlation.png).

## Reproduce the headline in 30 minutes

```bash
cd factor-a-domain-structure
python reproduce_rho.py             # ≈25 min CPU, prints rho + regenerates fig
```

`reproduce_rho.py` reads the 29-domain tokenised windows from `data/`, computes SS per domain via `entropy.py`, fits a small Transformer per domain to measure bits-per-character, and computes Spearman ρ. Requires: Python 3.10+, PyTorch, scipy, matplotlib.

If you already have `results/canonical_training_entropy.json` and `results/bpc_per_domain.json`, just the correlation + figure step takes <30 seconds:

```bash
python reproduce_rho.py --skip-bpc
```

## What's in this directory

| File | Purpose |
|---|---|
| `entropy.py` | Core entropy functions: unigram H₀ via Miller-Madow, conditional H_k via pooled n-gram counts, structure score. Pure functions, no I/O. 270 lines. |
| `per-domain-scripts/*_entropy.py` | 19 domain-specific wrappers that encode each domain's raw data into a character stream and invoke `entropy.py`. |
| `reproduce_rho.py` | One-shot reproduction: SS → fit small Transformer → BPC → ρ → figure. |
| `results/canonical_training_entropy.json` | Per-domain H₀, H₃, SS, n_chars, alphabet. 29 entries. |
| `results/bpc_per_domain.json` | Bits-per-character after training a small Transformer on each domain's stream. |
| `results/shuffle_baseline.json` | Shuffled-stream control: SS collapses to noise, verifying the score is structural not statistical. |
| `data/` | Tokenised windows per domain. ~20 MB. Source attributions in `data/SOURCES.md`. |

## Key subsidiary findings

- **Shuffle baseline**: random-permutation of each character stream reduces SS to ≈0 for every domain, confirming SS measures sequential structure, not marginal character distribution.
- **Out-of-training tidal verification**: tidal-prediction SS (pre-registered before measurement) landed at 0.657, within the range of music-like domains — confirming the structure score generalises to a domain whose SS was pre-registered.
- **Alphabet-size robustness**: SS values are stable across alphabet sizes from 4 (DNA coding) to 94 (MIDI). No spurious high-SS at small alphabets.

## What this factor does NOT do

- SS does not predict *which* patterns the domain contains — only how predictable the next character is given a short prefix.
- SS is a signal-side property, not a model-side property. It does not account for tokeniser choices, sub-word vocabulary, or model architecture.
- SS correlates with bridge-difficulty but is **necessary, not sufficient**: see Factor B/C/D for capacity and data-quality effects that determine whether a bridge can actually be built.

## References in the paper

- Introduction: SS as the operational definition of "narrow-band" sense.
- §3.4 Entropy Measurement and Neural Compression: ρ = −0.92 result.
- §3.4.2 Tidal verification: pre-stated SS holds out-of-training.
- S9 Per-Channel Canonical Structure Scores: full 29-domain table.

## Data sources

See [`data/SOURCES.md`](data/SOURCES.md) for per-domain provenance, licences, and preprocessing steps. Briefly: all data is publicly available or permissively licensed; hormonal, EEG and quantum sources have their own citation requirements reproduced there.
