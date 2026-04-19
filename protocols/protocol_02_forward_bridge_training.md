# Protocol 2 — Forward Bridge: Domain → Text

**Author**: Daniel Ziekenoppasser-Powell (Independent researcher, UK)
**Keywords**: multimodal AI, language models, BLIP-2, soft prompts, bridge learning, narrow-band senses
**Licence**: CC-BY 4.0
**Intended DOI**: `10.17504/protocols.io.[TBD]`
**Companion paper**: Ziekenoppasser-Powell 2026, *Nature Machine Intelligence* (under review)

## Abstract

Train a **forward bridge** that projects a frozen domain-model encoder's hidden representation into a frozen language model's input space, producing a natural-language description of the domain sequence. Architecture: BLIP-2-style MLP projection. Training the bridge (not the encoder or LM) yields 3.0–4.5 bits of validation-loss reduction over a zero-conditioning baseline, uniformly across 16 domains spanning SS = 0.001 to 0.773. This protocol specifies the canonical forward-bridge recipe used in the paper.

**Intended audience**: researchers building translation-style bridges from non-linguistic data streams (sensor, molecular, musical, numerical) to natural-language descriptions, using frozen public LMs.

## Guidelines and warnings

- This protocol assumes the paired-description corpus is **semantic-balanced** per Protocol 3's pair-generation pattern (see also `factor-d-paired-description-quality/README.md`). Natural or synthetic pairs produce null results; do not use them.
- The protocol is specified for three LM scales (GPT-2 small, GPT-2 medium, Pythia-1B) and three domain-model scales (S, M, XL). Canonical recipe: XL encoder + Pythia-1B LM + BAL-1600 semantic pairs. Other combinations are for scaling-analysis experiments.
- Zero-conditioning and random-prompt baselines are **mandatory** for every bridge run (see Steps 9 + 10). Reporting bridge val-loss without the baseline pair invites the confabulation critique (paper §4.8).

## Materials

### Hardware

- NVIDIA A100 (40 GB or 80 GB) or equivalent. Tested on Colab Pro+ A100 80 GB.
- Alternative: multi-GPU configuration with ≥ 24 GB VRAM per GPU for Pythia-1B.

### Software

- Python 3.10+, PyTorch 2.3+, Transformers 4.41+, HuggingFace Datasets 2.18+
- bitsandbytes (for fp16/bf16 mixed-precision)
- Weights & Biases or equivalent for training logs (optional)

### Model checkpoints

- Domain model: S-size (50 M) or XL-size (380 M) autoregressive character-level Transformer trained on the target domain. Training code in the companion repo `factor-b-domain-model-capacity/train_s.py`.
- Language model (frozen):
  - `gpt2` (124 M, HuggingFace)
  - `gpt2-medium` (355 M, HuggingFace)
  - `EleutherAI/pythia-1b` (1.01 B, HuggingFace)

### Data

- Semantic-balanced paired corpus for the target domain. Typical shape: 1600 pairs, stratified 320 pairs × 5 categories, 80/10/10 train/val/test split. See `factor-d-paired-description-quality/labeller_template.md` for labeller design.
- Tokeniser: use the off-the-shelf BPE tokeniser for each HuggingFace LM checkpoint.

## Before you begin

1. Clone companion repo and install requirements.
2. Download or generate the paired-description corpus (Step 6 of Protocol 3).
3. Train the S-size domain model per `factor-b-domain-model-capacity/README.md` or obtain a pre-trained checkpoint from the Zenodo deposit.
4. Verify the domain model's val-loss plateau and save its checkpoint.

## Procedure

### Step 1 — Load frozen encoder and frozen LM

```python
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AutoModelForCausalLM
encoder = torch.load("gcode_XL_s42.pt")  # your domain checkpoint
encoder.eval()
for p in encoder.parameters(): p.requires_grad = False

lm = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-1b")
lm.eval()
for p in lm.parameters(): p.requires_grad = False

tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-1b")
```

Verify: encoder forward pass produces `(batch, seqlen, d_enc)` hidden states; LM forward pass produces logits of shape `(batch, seqlen, vocab_size)`.

### Step 2 — Define the MLP projection bridge

```python
class ForwardBridge(torch.nn.Module):
    def __init__(self, d_enc, d_lm, n_soft=8):
        super().__init__()
        self.n_soft = n_soft
        self.d_lm = d_lm
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(d_enc, 4 * d_lm),
            torch.nn.GELU(),
            torch.nn.Linear(4 * d_lm, n_soft * d_lm),
        )
    def forward(self, hidden):
        return self.mlp(hidden).view(-1, self.n_soft, self.d_lm)

bridge = ForwardBridge(d_enc=1024, d_lm=2048, n_soft=8).cuda()
```

`n_soft = 8` is canonical. Larger `n_soft` (32) produces marginal improvement at ~4× memory cost.

### Step 3 — Reduce encoder hidden to single vector

For each input sequence, take the **final non-pad token's top-layer hidden state**:

```python
with torch.no_grad():
    enc_hidden = encoder(input_ids).last_hidden_state  # (B, L, d_enc)
final_token_idx = attention_mask.sum(dim=1) - 1        # (B,)
pooled = enc_hidden[torch.arange(len(enc_hidden)), final_token_idx]  # (B, d_enc)
```

### Step 4 — Project into soft-prefix

```python
soft = bridge(pooled)        # (B, n_soft, d_lm)
```

### Step 5 — Prepend soft-prefix to description embedding

```python
desc_ids = tok(descriptions, padding=True, return_tensors="pt").input_ids.cuda()
desc_emb = lm.get_input_embeddings()(desc_ids)
inputs = torch.cat([soft, desc_emb], dim=1)
```

Create matching attention mask and position IDs; the soft-prefix positions get `attention_mask = 1` and position IDs `[0, 1, …, n_soft - 1]`.

### Step 6 — Forward through frozen LM and compute loss over description tokens

```python
out = lm(inputs_embeds=inputs, attention_mask=attn_mask)
logits = out.logits[:, n_soft:, :]  # description-token logits only
loss = torch.nn.functional.cross_entropy(
    logits.reshape(-1, logits.size(-1)),
    desc_ids.reshape(-1),
    ignore_index=tok.pad_token_id,
)
```

### Step 7 — Optimise bridge parameters only

```python
opt = torch.optim.AdamW(bridge.parameters(), lr=5e-4, betas=(0.9, 0.95), weight_decay=0.0)
```

Cosine decay to 10% of peak LR over training. 200-step linear warmup.

### Step 8 — Train for up to 20 epochs with patience 3

Log validation loss every epoch. Save the best-val checkpoint. Stop if val plateaus for 3 epochs.

Expected time: ≈ 30 min / domain on A100 at Pythia-1B × XL encoder.

### Step 9 — Zero-conditioning baseline (mandatory)

Re-run training with the MLP output replaced by zero embeddings:

```python
soft_zero = torch.zeros_like(soft)
```

Report: the **zero-baseline best-val-loss**. Should always be > bridge best-val-loss.

### Step 10 — Random-prompt baseline (mandatory)

Draw `n_soft` vectors randomly from the LM's input-embedding matrix, with 5 seeds:

```python
emb_matrix = lm.get_input_embeddings().weight  # (V, d_lm)
for seed in range(5):
    torch.manual_seed(seed)
    idx = torch.randint(0, emb_matrix.shape[0], (n_soft,))
    soft_random = emb_matrix[idx].unsqueeze(0).expand(B, -1, -1)
    # re-compute loss as in Step 6
```

Average the 5 random baselines. Should be **slightly higher** than zero baseline (the LM sometimes gets lucky on random prefixes).

### Step 11 — Report three headline numbers

- Bridge best-val-loss
- Zero-baseline best-val-loss
- Random-baseline (5-seed mean) best-val-loss
- `zero − bridge` in bits (the paper's main reported number — "bridge contribution")

### Step 12 — Held-out test-set evaluation

Using the best-val checkpoint, evaluate on the held-out 10% test split. Report final-table numbers from test (not val) to avoid overfitting-to-val confound.

### Step 13 — Cross-LM-scale replication

To check that the bridge is not an artefact of a specific LM's prior, replicate Steps 1–12 with:

- `gpt2` (same domain encoder)
- `gpt2-medium` (same domain encoder)

Expected: bridge contribution persists at every LM scale; uniformity of 3.0–4.5 bits at Pythia-1B is the paper's headline claim.

### Step 14 — Cross-encoder-scale replication (optional)

Replicate Steps 1–12 with S-size and M-size domain encoders at fixed LM. Factor B results documented in paper §4.4.

### Step 15 — Save outputs

```
results/
  {domain}_semantic_{lm}_{s_size}_training.json  # training trace + best-val
  {domain}_semantic_{lm}_{s_size}_zero.json      # zero-baseline result
  {domain}_semantic_{lm}_{s_size}_random.json    # 5-seed random baseline
  {domain}_semantic_{lm}_{s_size}_bridge.pt      # bridge checkpoint
```

Aggregate into `bridge_results_v2.json` via the repo's `aggregate_all.py` for cross-domain analyses.

## Expected results

- Bridge best-val-loss: ranges from 0.24 (SMILES, S) to 1.36 (tidal, S) at GPT-2 small; lower at larger LM.
- Zero-baseline: ~3–5 bits higher than bridge at every (LM, encoder) combination.
- Random-baseline: 0.1–0.3 bits higher than zero-baseline.
- Bridge contribution (zero − bridge) ≈ **3.0–4.5 bits uniformly at Pythia-1B** across 16 domains.
- Spearman ρ(SS, zero − bridge) ≈ **+0.30** at Pythia-1B, n = 16 (non-monotonic with LM scale — see paper §4.3).

## Troubleshooting

- **Bridge val-loss plateaus at or near zero-baseline**: encoder contributes no information. Check encoder final-token hidden state is non-degenerate (`hidden[0] != hidden[1]`). Check training batches aren't all one category. Try a different domain-model checkpoint.
- **Bridge val-loss diverges**: LR too high. Reduce to 2e-4. Check for NaN in bridge weights (bf16 precision issues).
- **Zero-baseline similar to random-baseline**: LM is insensitive to the prefix at all (unusual). Verify the `inputs_embeds` pathway by comparing to standard generation with the same prefix.
- **Per-domain results don't aggregate into the paper's figures**: ensure all domains use the same `n_soft`, batch size, and random seed. Minor variations can shift ρ by ±0.05.

## References

- Li et al. 2023, *BLIP-2* — the MLP-projection architecture that inspired this bridge
- Liu et al. 2023, *LLaVA* — MLP-vs-Q-Former comparison (informs Step 2's "use MLP" choice)
- Ziekenoppasser-Powell 2026 — the NBS paper

## Acknowledgements

Architecture choices follow the LLaVA > BLIP-2 lesson (MLP equals Q-Former at this scale, and is cheaper). Degeneracy controls (zero + random baselines) adopted from ablation-study conventions in the soft-prompt literature.
