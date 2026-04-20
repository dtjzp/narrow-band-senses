# Reverse Bridge Architecture — Detail

## Pipeline

```
  text prompt
       │
       ▼
 ┌──────────────┐
 │ LM tokeniser │  (frozen)
 └──────┬───────┘
        │ tokens (variable length)
        ▼
 ┌──────────────┐
 │ LM encoder   │  (frozen, used only to produce embedding of [BOS] + tokens)
 └──────┬───────┘
        │ pooled prompt embedding (d_lm)
        ▼
 ┌──────────────┐
 │ MLP projection│ (trainable)
 └──────┬───────┘
        │ 64 soft-prefix embeddings (d_enc)
        ▼
 ┌──────────────┐
 │ domain model │ (trainable at 0.1x bridge LR; S-size only)
 └──────┬───────┘
        │
        ▼
   sampled tokens (temp=0.8, top-k=40) → domain sequence
```

## Key decisions

### Fine-tune, don't freeze

Unlike forward bridges (where the domain model is frozen), reverse bridges **fine-tune** the domain model jointly with the projection. The fine-tune LR is 0.1× the projection LR — aggressive enough to adapt to soft-prefix conditioning, gentle enough to preserve the domain priors learned during S-size training.

### 64 soft tokens

Longer than forward-bridge's 8 because the prompt carries more conditional information than the encoder's final-token hidden state did. Tested `n_soft ∈ {16, 32, 64, 128}`; 64 is the plateau. Beyond 64, val loss flat and generations no better.

### S-size only — XL mode-collapses

See [`../../archive/xl-reverse-bridge-collapse/README.md`](../../archive/xl-reverse-bridge-collapse/README.md). This is the key Factor B × D interaction: at XL + reverse-bridge recipe, the joint capacity produces a collapsed attractor landscape and 2–24% unique generations.

### Temperature sampling, not greedy

MIDI greedy: 36.4% overall fidelity. MIDI sampled (temp 0.8, top-k 40, seed 42): 60.3% on same checkpoint + prompts. See paper §S21. The recipe used for every PoC from 2026-04-18 onward.

## Loss

Standard causal-LM cross-entropy over the domain model's output. The soft-prefix positions are excluded from loss.

## Training

- Optimiser: AdamW, lr=5e-4 (bridge) + 5e-5 (domain fine-tune), β=(0.9, 0.95), weight decay 0.1
- LR schedule: cosine decay to 10%, 200-step warmup
- Batch size: 8 (A100 memory at S-size + 64 soft tokens + domain-length contexts)
- Precision: bf16 mixed-precision
- Stop criterion: val loss plateau (patience=10) or max 30 epochs
- Seed: 42

Training time: ~2-5 min on A100 per PoC at BAL-1600 semantic pairs.

## Corpus shape

Each PoC uses a balanced paired corpus. Canonical target: **1280-2400 pairs**, balanced across **4-5 categories** (~320 per category). Categories are domain-specific and data-driven — the per-domain labellers produce the category-level structure; these are the categories as they appear in the committed `decision_checkpoint.json` / `scorecard_heldout_temp08.json` for each PoC:

- **G-code**: `extrusion, travel, mixed, retraction, accel` (BAL-2400)
- **MIDI**: `tonal, ascending-run, descending-run, leap-dominant, mixed` (BAL-1600)
- **SMILES**: chemical-class categories (drug-like, simple, polycyclic, etc.); legacy PoC, see `smiles/REPORT.md`
- **Python**: `arithmetic, loop, conditional, collection, complex` (BAL-1600, window=400)
- **Network**: `zero-dense, nine-heavy, five-heavy, one-heavy` (BAL-1280; data-driven from observed 0-9 digit distribution)
- **Quantum**: `parameterised, highly-entangled, measurement-heavy, entangling, single-qubit` (BAL-1468; `entangling` at ~12% of samples)
- **RNA**: `simple-hairpin, nested-helical, bulge-rich, multiloop` (pseudoknot dropped as < 5%)
- **DNA coding**: `high-GC, mid-GC, low-GC, with-ORF` (homopolymer-rich dropped)
- **ATC**: `climbing, descending, mid-cruise, low-altitude, high-cruise` (BAL-1353; Spec 3 v2)
- **Reactions**: `addition, elimination, substitution, rearrangement` (BAL-1280; Spec 3 v2)

Category rule is a deterministic classifier over a 200-character window (except Python + Reactions at 400). See each per-domain labeller (templates in `../../factor-d-paired-description-quality/labeller_template.md`; domain-specific labellers pending Zenodo deposit).

## Decoding

```python
domain_tokens = model.generate(
    input_embeds=soft_prefix,
    do_sample=True,
    temperature=0.8,
    top_k=40,
    max_new_tokens=window_length,
    seed=42,
)
```

For each held-out prompt, 1 sample is generated. `n = 160-200` held-out prompts per PoC (32-50 per category × 4-5 cats).

## What "capability" means here

A reverse bridge has **capability** on a domain if its generations are:

1. Syntactically valid (parse, balanced brackets, legal tokens)
2. Category-conditional — generations for category A prompts classify as A at > chance rate
3. Feature-match — the generated sequence exhibits the features claimed by the prompt (arc counts, GC content, stem-loop depth, etc.)
4. **Diverse within a category** — no mode collapse

Point 4 is the hardest. Aggregate cat-match can look high when the bridge collapses to a few canonical outputs that happen to re-classify correctly. Per-category unique-generation rate and intra-category Hamming distance are the checks that catch this. See [`mode-collapse-diagnostics.md`](mode-collapse-diagnostics.md).
