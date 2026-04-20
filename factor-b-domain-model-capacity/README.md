# Factor B — Domain-Model Capacity

**Definition**: the capacity of the encoder (domain model) trained on the target signal. Varied in this work across three scales: S, M, XL.

The domain model is the frozen front-end that turns a tokenised domain sequence into a hidden representation. A forward bridge reads from this representation; a reverse bridge writes into it. Factor B asks: **how much does encoder capacity matter, given that Factors A / C / D are fixed?**

## Key finding

**At constrained LM scale (GPT-2 small), domain-model capacity is binding.** Going from S to M to XL on the same paired-description corpus meaningfully reduces forward-bridge validation loss for most domains. At Pythia-1B LM scale, the effect compresses: XL-size domain models still help, but the marginal improvement from M → XL is smaller than the improvement from S → M.

**At S-size domain models, Factor D (paired-description quality) is the binding constraint — not capacity.** This is the negative finding that made the four-factor framework necessary: you cannot scale your way out of bad paired-description data by growing the encoder.

See paper §4.4 for the full treatment and supplementary S13 (S-Medium intermediate bridge deltas) for the intermediate-scale data.

## Figures

| Figure | Shows |
|---|---|
| [`../paper-figures/fig_s_model_scaling.png`](../paper-figures/fig_s_model_scaling.png) | Validation loss vs domain-model scale, per domain |
| [`../paper-figures/fig_bridge_contribution.png`](../paper-figures/fig_bridge_contribution.png) | Bridge bits-vs-baseline at each S/M/XL scale |
| [`../paper-figures/fig_bridge_contribution_xl_n16.png`](../paper-figures/fig_bridge_contribution_xl_n16.png) | XL-only bridge contribution at n=16 |

## What's in this directory

| File | Purpose |
|---|---|
| `train_s.py` | Canonical S-size domain-model training script (~50M params, char-level). CLI flags for `--domains`, `--epochs`, `--data-dir`, `--ckpt-dir`. |
| `architecture.md` | Architectural details: layer count, dims, dropout, optimiser, stop criteria. Defines S/M/XL + XS (dev-only). |
| `results/scaling_summary.md` | Per-domain S/M/XL best-val tables + deltas. |
| `results/xl_training_diagnostics.md` | XL train-val gap patterns (paper §S14). |

M-size and XL-size training scripts are trivial variants of `train_s.py` — they differ only in `MODEL_SIZE = 'M'` / `'XL'` and a correspondingly larger `--batch-size`. Rather than duplicate the 150-line trainer, the M / XL variants live Drive-resident pending Zenodo deposit; for a fresh reviewer, the minimal reproduction is to edit `MODEL_SIZE` at the top of `train_s.py` and adjust batch size per `architecture.md`. Runs were all on Colab Pro+ A100.

## Reproduce

S-size training for one domain (~2.5 min A100 per domain):

```bash
cd factor-b-domain-model-capacity
python train_s.py --domain gcode --epochs 20
```

Produces `<domain>_S_s42.pt` (checkpoint) and a training log. For the M/XL-size runs see `architecture.md`; those require an A100-class GPU.

## How Factor B interacts with the other factors

- **A × B**: For a high-SS domain (Factor A high), even an S-size domain model captures most of the signal. For a low-SS domain, B is dominated by A — no amount of encoder capacity creates structure that isn't there. Evidenced by financial (SS=0.011) and SETI (SS=0.001) showing near-zero improvement from S → XL.
- **B × C**: At GPT-2 small (constrained C), B matters. At Pythia-1B (abundant C), B compresses. The LM can recover missing encoder capacity through its own depth.
- **B × D**: Biggest surprise. S + high-quality semantic pairs **beats** XL + low-quality natural pairs on reverse-bridge fidelity. This is the Factor D dominance claim — see Factor D README.

## Reverse-bridge rule (from Factor B × D interaction)

**Reverse bridges use S-size only.** XL-size reverse bridges mode-collapse, documented in:

- MIDI XL: 24% unique generations, all classified tonal (paper §S21)
- G-code BAL-600 XL: 2% unique generations (paper §4.6)
- G-code BAL-2400 XL: OOM'd — see `../archive/xl-reverse-bridge-collapse/README.md`

This is a pure-B finding: the encoder's capacity interacts destructively with the soft-prefix reverse-bridge recipe. The Factor D dominance finding (balanced + small > imbalanced + big) is the validated recipe; reverse bridges stay at S.
