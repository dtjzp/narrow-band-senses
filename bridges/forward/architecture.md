# Forward Bridge Architecture — Detail

See also: [`../../factor-c-language-model-capacity/architecture.md`](../../factor-c-language-model-capacity/architecture.md) for the full LM-side projection detail.

## MLP projection

```python
class ForwardBridge(nn.Module):
    def __init__(self, d_enc: int, d_lm: int, n_soft: int = 8):
        super().__init__()
        self.n_soft = n_soft
        self.d_lm = d_lm
        self.mlp = nn.Sequential(
            nn.Linear(d_enc, 4 * d_lm),
            nn.GELU(),
            nn.Linear(4 * d_lm, n_soft * d_lm),
        )

    def forward(self, encoder_hidden):        # (B, d_enc)
        flat = self.mlp(encoder_hidden)        # (B, n_soft * d_lm)
        return flat.view(-1, self.n_soft, self.d_lm)  # (B, n_soft, d_lm)
```

Parameter count: `d_enc * 4*d_lm + 4*d_lm * n_soft * d_lm ≈ 3M for S-size encoder → GPT-2 small; ≈ 16M for XL → Pythia-1B`.

## Encoder-side reduction

The encoder hidden state is reduced from sequence-length to a single vector by taking the **final non-pad token's top-layer hidden state**. This is canonical and has been stable across all runs.

Alternative reductions tested during development (not adopted):
- mean-pooling over non-pad tokens: slightly lower val loss but ablated by decision to match BLIP-2 convention
- weighted-pool using a learned attention over tokens: adds ~1M bridge params, marginal improvement

## Prefix-conditioning in the LM

The `n_soft` projected embeddings are prepended to the LM's description-token embeddings. The LM's positional embedding table is extended to length `n_soft + description_length`; the first `n_soft` positions use freshly-initialised position vectors (zero-init works equally well in practice). LM weights remain frozen except the position table extension, which is trainable (`n_soft × d_lm` additional params — negligible).

## Loss

Standard per-token cross-entropy over the description tokens only (soft-prefix positions are excluded from loss). Reduction: `mean`, per batch.

## Validation protocol

- 80 / 10 / 10 train / val / test split, stratified by category (when labelled) or by sequence-id (otherwise)
- val loss is reported at the best-epoch checkpoint (patience 3 on val-loss plateau)
- test-set numbers are held back for the paper's final tables; val-loss is used for scaling and LM-comparison analyses

## Reproducibility

All runs use seed 42. Per-domain bootstrap (resampling train pairs with replacement) is used to estimate confidence intervals where quoted in the paper (ρ 95% CIs in the correlation tables).

## Compute

At XL encoder + Pythia-1B LM, a single domain's forward-bridge run takes ≈ 30 min on A100. Full n=16 grid at this scale: ≈ 8 hours wall-clock.
