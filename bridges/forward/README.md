# Forward Bridges: Domain → Text

A **forward bridge** projects a frozen domain encoder's hidden representation into a frozen language model's input space, producing a natural-language description of the domain sequence. The bridge is the only component trained; both encoder and LM are frozen throughout.

## Architecture at a glance

```
  domain sequence
        │
        ▼
 ┌────────────────┐
 │ domain encoder │  (frozen; S / M / XL)
 └───────┬────────┘
         │  hidden state (d_enc)
         ▼
 ┌────────────────┐
 │ MLP projection │  (trainable; ~1-10M params depending on d_enc × d_lm)
 └───────┬────────┘
         │  n_soft soft-prefix embeddings (d_lm)
         ▼
 ┌────────────────┐
 │  language model│  (frozen; GPT-2 small / GPT-2 medium / Pythia-1B)
 └───────┬────────┘
         │
         ▼
    description
```

Full details: [`architecture.md`](architecture.md).

## Key finding

**Forward bridges are broadly constructable.** At Pythia-1B with XL-size domain encoders and balanced semantic pair corpora, the trained bridge reduces validation loss by **3.0–4.5 bits** versus a zero-conditioning baseline, **uniformly across 16 domains spanning SS = 0.001 to 0.773**.

The bridge is doing real information transfer — it is not confabulating from the LM's prior. This is demonstrated by the zero-conditioning + random-prompt baselines, which both produce higher val loss than the trained bridge at every scale × every domain. See `diagnostics/` for the baseline control scripts.

## Training recipe (canonical)

- **Encoder**: frozen, XL-size, checkpoint from Factor B training
- **LM**: frozen, Pythia-1B for the headline recipe (GPT-2 small for LM-scaling experiments)
- **Bridge**: 2-layer MLP projection, `n_soft = 8`, trained 20 epochs max with patience 3
- **Pairs**: semantic, BAL-1600 per domain
- **Optimiser**: AdamW, lr=5e-4, cosine decay to 10%, 200-step warmup
- **Precision**: bf16 mixed-precision on A100

Training time: ~30 min / domain at Pythia-1B × XL encoder on A100.

## Reproduce one domain

The canonical forward-bridge trainer (`phase34_bridge.py`) is Drive-resident pending Zenodo deposit — it's a ~200-line script that instantiates a frozen Pythia-1B, a frozen domain encoder, and a 2-layer MLP bridge, then trains the bridge with AdamW per the hyperparameters above. The intended invocation is:

```bash
python train_mlp_bridge.py --domain gcode --lm pythia1b --s_size XL
```

Expected output: bridge checkpoint (`*_mlp.pt`) + per-run training JSON that aggregates into `../factor-c-language-model-capacity/aggregate_all.py`'s input.

Val loss converges near 0.24 for G-code (domain-dependent; see Factor B scaling summary).

## Degeneracy controls

Every forward-bridge headline run is accompanied by two baseline controls, per paper §4.8:

### Zero-conditioning baseline
The MLP output is replaced with `n_soft` zero embeddings; LM is then asked to produce the same description with no domain conditioning. Val loss is always higher than trained bridge. _Baseline-control script is Drive-resident pending Zenodo deposit._

### Random-prompt baseline
The MLP output is replaced with a random draw from the LM's own embedding matrix (5 seeds averaged). Ensures the LM is not just using "any prefix as a hint" — random embeddings should perform worse than targeted ones, and they do. _Baseline-control script is Drive-resident pending Zenodo deposit._

Both baselines form the `zero − best` and `random − best` bits numbers reported in the paper's bridge-contribution figures.

## What to look for when debugging a new domain

1. **Zero baseline not converging**: forward-bridge training without a bridge shouldn't need convergence — you're just measuring LM-only loss on the description. If it "doesn't converge," something is wrong with the LM inference pipeline.
2. **Trained bridge matching zero baseline**: the encoder is contributing no information. Check the MLP head for collapse (`torch.allclose(output, output[0])`); check the encoder's final-token hidden state is non-degenerate; check that the training batches aren't all one category.
3. **Trained bridge beating zero baseline by only 0.1 bits**: pair corpus may be too small, too unbalanced, or too generic (classic Factor D symptom). Try BAL-1600 balanced semantic.

See paper §6.2 Limitations for a catalogue of forward-bridge failure modes documented during development.

## Architecture alternatives tested

- **Linear projection** (no MLP): baseline, always worse
- **Q-Former** (BLIP-2 style): within noise of MLP at this scale (paper §S11)
- **MLP with `n_soft = 32`**: marginal improvement; `n_soft = 8` chosen as default for compute efficiency

Full comparison in [`../../paper-figures/fig_arch_comparison.png`](../../paper-figures/fig_arch_comparison.png).

## References in the paper

- §4.1 Experimental setup
- §4.2 Forward bridges are constructable across the structure-score spectrum (headline)
- §4.3 LM capacity compresses SS dependence
- §4.4 Domain-model capacity is a binding constraint at constrained LM scale
- §4.8 Methodology: zero-baseline degeneracy control
- §S11 Bridge Architecture Details
