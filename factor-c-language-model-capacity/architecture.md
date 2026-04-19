# Forward Bridge: Projection into LM Embedding Space

## Target LMs and tokenisers

| LM | Params | Vocab size | Embedding dim | HuggingFace ID |
|---|---|---|---|---|
| GPT-2 small  | 124 M  | 50257 | 768  | `gpt2` |
| GPT-2 medium | 355 M  | 50257 | 1024 | `gpt2-medium` |
| Pythia-1B    | 1.01 B | 50304 | 2048 | `EleutherAI/pythia-1b` |

All three LMs are kept **frozen** throughout bridge training. Tokenisers are the off-the-shelf BPE tokenisers shipped with each checkpoint.

## Projection architecture

Two variants were tested (paper §4.1):

### MLP projection (canonical)

Two-layer MLP:

```
encoder_hidden (d_enc)  →  GELU  →  Linear(d_enc, 4 * d_lm)
                        →  GELU  →  Linear(4 * d_lm, n_soft * d_lm)
                        →  reshape to (n_soft, d_lm)
```

`n_soft = 8` query embeddings prepended to the target description's BPE-tokenised prefix in the LM's embedding space. The LM is conditioned on these 8 soft tokens + the textual description; loss is the standard per-token cross-entropy over the description tokens.

### Q-Former (alternate, paper §S11)

Following BLIP-2: a small Transformer (2 layers, 8 heads, 512 dim) with 32 learnable queries that cross-attend to the encoder's hidden states, then project to (n_soft=32, d_lm).

**Empirical conclusion**: Q-Former ≈ MLP on bridge-val loss. Architecture is not the capacity bottleneck at this scale (consistent with LLaVA > BLIP-2 lesson). MLP is the canonical forward-bridge architecture in the paper.

## Training recipe

- Optimiser: AdamW, lr=5e-4, β=(0.9, 0.95), weight decay 0.0 (bridge-only)
- LR schedule: cosine decay to 10%, 200-step linear warmup
- Batch size: 8 (limited by LM memory at Pythia-1B)
- Precision: bf16 mixed-precision
- Stop criterion: val loss plateau (patience=3) or max 20 epochs
- Seed: 42

## Zero-conditioning baseline (degeneracy control)

A matched run where the MLP projection output is replaced with a zero vector (`n_soft` zero embeddings). The LM is then asked to produce the same description with no domain conditioning. Baseline val loss is always higher than bridge val loss — this is the 3.0–4.5 bits contribution.

## Random-prompt baseline (degeneracy control)

A second matched run where the MLP output is replaced with a random draw from the LM's own embedding matrix (5 seeds, averaged). Random prompts also produce higher val loss than the trained bridge. This rules out the "the LM reads any prefix as a useful hint" confound.

## Results schema

Each training run writes a JSON record with:

```json
{
  "domain": "gcode",
  "source": "semantic",
  "arch": "mlp",          // or "qformer" or "linear"
  "lm": "pythia1b",       // or "gpt2", "gpt2medium"
  "s_size": "XL",         // or "S", "M"
  "metric_name": "bridge_val",
  "metric": 0.27,
  "zero_baseline": 4.12,
  "random_baseline": 3.98,
  "improvement": "computed as (zero - best) in newer runs",
  "n_soft": 8
}
```

`aggregate_all.py` reads these into a flat table used for all correlation analyses.
