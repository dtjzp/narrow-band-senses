# Protocol 3 — Reverse Bridge: Text → Domain

**Author**: Daniel Ziekenoppasser-Powell (Independent researcher, UK)
**Keywords**: multimodal AI, language models, soft prompts, fine-tuning, mode collapse, text-to-domain generation
**Licence**: CC-BY 4.0
**Intended DOI**: `10.17504/protocols.io.[TBD]`
**Companion paper**: Ziekenoppasser-Powell 2026, *Nature Machine Intelligence* (under review)

## Abstract

Train a **reverse bridge** that generates tokenised domain sequences (G-code, MIDI, SMILES, Python, quantum circuits, RNA structures, network flows, DNA-coding strings) from natural-language prompts. Architecture: a fine-tuned S-size domain model conditioned on 64 learnable soft-prefix tokens projected from the prompt. This protocol also specifies the **mandatory mode-collapse diagnostic suite** — three criteria that must be computed on every held-out evaluation to catch a failure mode that aggregate metrics hide. Across 8 domains spanning SS = 0.033 to 0.698, the protocol produces verifiable generations in high-SS domains (≥ 0.3) and framework-validated failures in low-SS domains (< 0.15).

**Intended audience**: researchers building generative text-to-domain systems for scientific notations, experimental controls, or programmatic domains.

## Guidelines and warnings

- **Use S-size domain models only.** XL mode-collapses under this recipe (see paper §S21 and companion `archive/xl-reverse-bridge-collapse/`). This is the single most consequential parameter choice.
- **Use temperature sampling (temp = 0.8, top-k = 40) for all generation**, not greedy decoding. Greedy collapses to a single canonical output per prompt — paper §S21 documents MIDI greedy at 36.4% overall fidelity vs sampled at 60.3% on the same checkpoint.
- **Run the mode-collapse diagnostics before reporting any capability numbers.** Three PoCs in the paper (Network, RNA, DNA coding) have aggregate cat-match values that would have been reported as "partial capability" if not gated on the diagnostics. The diagnostics caught the mode collapse; honest framing followed.
- Balanced paired corpora (BAL-1280–1600, 320 pairs × 4–5 categories) are canonical. Imbalanced corpora produce mode-collapsed outputs at scale — this is the Factor D × Factor B interaction documented in paper §4.6.

## Materials

### Hardware

- NVIDIA A100 (40 GB+). CPU-only is infeasible — reverse-bridge training involves autoregressive decoder fine-tuning at scale.

### Software

- Python 3.10+, PyTorch 2.3+, Transformers 4.41+
- bitsandbytes (bf16)
- For per-domain validators: as required by the domain (rdkit for SMILES, ast module for Python, mido for MIDI, etc.)

### Model checkpoints

- Domain model: S-size (~50 M) autoregressive character-level Transformer trained on the target domain. Same checkpoint as used in Protocol 2 forward bridge.
- Language-model tokeniser: for prompt tokenisation. Typically `gpt2` tokeniser (any BPE tokeniser works; choose the one matching your training prompts' provenance).

### Data

- Semantic-balanced paired corpus. BAL-1280 or BAL-1600 target: 320 pairs × 4 or 5 categories.
- Per-domain labeller: deterministic function mapping sequence window → claim dict → natural-language description.
- Per-domain verifier: complementary function mapping description → extracted claim dict. Used for evaluation (Step 14).

Templates: `factor-d-paired-description-quality/labeller_template.md`. 8 concrete examples across the paper's 8 reverse-bridge PoCs.

## Before you begin

1. Clone the companion repo and install requirements.
2. Train or download the S-size domain model for the target domain.
3. Generate the semantic-balanced paired corpus (BAL-1280–1600). See `factor-d-paired-description-quality/semantic-labellers/` for per-domain labeller reference.
4. Verify labeller round-trip accuracy ≥ 95% (labeller output → verifier → extracted claims; match rate against input claims).
5. Plan one A100-hour per PoC.

## Procedure

### Step 1 — Decision checkpoint: category distribution

On 2000-window sample from the domain's natural distribution, classify windows by category using the labeller. If the dominant category exceeds 70%, either:

- Retry with a larger window size (e.g. 400 chars instead of 200) — windows that span more of the sequence tend to have more balanced classifications
- Subdivide the dominant category
- Drop categories below 5% prevalence

Record all decisions in `decision_checkpoint.json` for audit trail.

### Step 2 — Build BAL-1280 or BAL-1600 balanced corpus

Per category, randomly select 320 windows. Concatenate across categories. Shuffle. 80/10/10 train/val/test split, stratified by category.

Each pair: `{"sequence": "<window>", "prompt": "<labeller(window) rendered to NL>"}`.

Verify: held-out round-trip accuracy of the verifier on the training descriptions is 100% by construction.

### Step 3 — Load S-size domain model

```python
import torch
domain_model = torch.load("gcode_S_s42.pt")
domain_model.train()
for p in domain_model.parameters(): p.requires_grad = True
```

Note: domain model is **NOT frozen**. Fine-tuned at 0.1× the bridge LR.

### Step 4 — Load LM tokeniser for prompts

```python
from transformers import GPT2Tokenizer
tok = GPT2Tokenizer.from_pretrained("gpt2")
```

### Step 5 — Define the reverse bridge

```python
class ReverseBridge(torch.nn.Module):
    def __init__(self, d_lm_tok, d_enc, n_soft=64):
        super().__init__()
        self.n_soft = n_soft
        self.d_enc = d_enc
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(d_lm_tok, 4 * d_enc),
            torch.nn.GELU(),
            torch.nn.Linear(4 * d_enc, n_soft * d_enc),
        )
    def forward(self, prompt_emb):  # (B, d_lm_tok)
        return self.mlp(prompt_emb).view(-1, self.n_soft, self.d_enc)
```

`n_soft = 64` is canonical. Tested values: 16, 32, 64, 128; 64 is the plateau.

### Step 6 — Pool prompt tokens into a single embedding

```python
from transformers import GPT2Model
prompt_enc = GPT2Model.from_pretrained("gpt2").eval()
for p in prompt_enc.parameters(): p.requires_grad = False

with torch.no_grad():
    prompt_hidden = prompt_enc(prompt_ids, attention_mask=prompt_mask).last_hidden_state
    # mean-pool over non-pad tokens
    pooled = (prompt_hidden * prompt_mask.unsqueeze(-1)).sum(dim=1) / prompt_mask.sum(dim=1, keepdim=True)
```

### Step 7 — Project to soft prefix

```python
soft = bridge(pooled)  # (B, n_soft, d_enc)
```

### Step 8 — Concatenate soft prefix + sequence embedding

```python
seq_emb = domain_model.get_input_embeddings()(sequence_ids)
inputs = torch.cat([soft, seq_emb], dim=1)
```

The attention mask and position IDs are extended accordingly; first `n_soft` positions have `attention_mask = 1` and positions `[0, …, n_soft - 1]`.

### Step 9 — Forward through domain model and compute loss over sequence tokens

```python
out = domain_model(inputs_embeds=inputs)
logits = out.logits[:, n_soft:, :]
loss = torch.nn.functional.cross_entropy(
    logits.reshape(-1, logits.size(-1)),
    sequence_ids.reshape(-1),
    ignore_index=domain_model.pad_token_id,
)
```

### Step 10 — Optimise: bridge at 5e-4, domain model at 5e-5

```python
bridge_params = list(bridge.parameters())
domain_params = list(domain_model.parameters())
opt = torch.optim.AdamW(
    [{"params": bridge_params, "lr": 5e-4},
     {"params": domain_params, "lr": 5e-5}],
    betas=(0.9, 0.95), weight_decay=0.1,
)
```

Cosine decay to 10% of peak over training; 200-step linear warmup.

### Step 11 — Train for up to 30 epochs with patience 10

Expected time: 2-5 min on A100. The domain model's val-loss should plateau around domain-specific values (e.g. 0.63 for Network, 0.50 for RNA, 1.30 for DNA coding).

### Step 12 — Generate held-out set with temperature sampling

```python
torch.manual_seed(42)
generations = []
for prompt in test_prompts:
    pooled = pool_prompt(prompt)
    soft = bridge(pooled)
    gen = domain_model.generate(
        inputs_embeds=soft,
        max_new_tokens=window_length,
        do_sample=True,
        temperature=0.8,
        top_k=40,
    )
    generations.append(gen)
```

32 or 50 generations per category; 160-200 total.

### Step 13 — Per-category scorecard

Using the per-domain verifier, compute for each generation:

- Did it parse (syntactically valid)?
- Does the classifier put it in the expected category?
- Per-feature match rate (against the claim dict of the ground-truth sequence it was paired with)

Aggregate into a scorecard JSON:

```json
{
  "overall": {"cat_match": 0.77, "feat_match": 0.82, "parse_rate": 1.0},
  "per_category": {
    "simple_tonal": {"cat_match": 0.56, "feat_match": 0.45, "unique_rate": 0.38},
    "entangling": {"cat_match": 0.84, "feat_match": 0.91, "unique_rate": 0.78},
    "deep": {"cat_match": 0.88, "feat_match": 0.92, "unique_rate": 0.91}
  }
}
```

### Step 14 — MANDATORY mode-collapse diagnostics

Compute three criteria on the held-out generations:

**Criterion 1 — Unique-generation rate per category**:

```python
unique_rate[c] = len(set(generations[c])) / len(generations[c])
```

Threshold: ≥ 30% unique per category. < 30% is FAIL.

**Criterion 2 — Intra-category Hamming distance**:

```python
from itertools import combinations
intra[c] = mean(hamming(g_i, g_j) for g_i, g_j in combinations(generations[c], 2))
```

Threshold: domain-dependent. 30 for small-alphabet (DNA, RNA), 10-20 for medium alphabet (G-code, MIDI), 5-10 for code (Python). Below threshold is FAIL.

**Criterion 3 — Cross-category leakage**:

```python
classified = [classify(g) for c in cats for g in generations[c]]
dominant = mode(classified)
leak = classified.count(dominant) / total
```

Threshold: leak > 80% is hard FAIL.

Any criterion triggering → VERDICT = FAIL. Write to:

```json
{
  "verdict": "PASS" | "FAIL",
  "per_category": {...},
  "cross_category_leakage_to_dominant": 0.34,
  "dominant_classification": "...",
  "failure_reasons": {
    "unique_rate_fail": false,
    "intra_cat_distance_fail": false,
    "high_leak_fail": false
  }
}
```

### Step 15 — Honest framing of the result

- **If VERDICT = PASS**: the bridge has capability. Report cat-match, feat-match, and unique-gen rates per category. Aggregate is a valid headline.
- **If VERDICT = FAIL and SS < 0.15**: framework-validation outcome. Low-SS domains are predicted to not support generative reverse bridging at S-size. Report the FAIL as **validated framework prediction** (paper §4.7 TRL playbook).
- **If VERDICT = FAIL and SS ≥ 0.3**: surprising high-SS failure. Report parseability as a separate capability (if applicable — RNA bracket-balance is the canonical example). Do **not** report aggregate cat-match as capability. Flag for follow-up (larger n_soft, per-category conditioning ablation, larger corpus).

### Step 16 — Cross-PoC aggregation

Across domains, tabulate: SS, verdict, per-category unique-gen, aggregate cat-match (caveated by verdict), honest headline. Paper §4.5 Table is the canonical cross-PoC table.

## Expected results

- **Quantum** (SS = 0.698): cat-match 77%, feat-match 82%, VERDICT = PASS.
- **Python** (SS = 0.523): cat-match 41% (2× chance), feat-match 43% with category-aware validator, VERDICT = PASS.
- **G-code** (SS = 0.323): hybrid pipeline scorecard near-100% (largely tautological); M1 source fidelity 67.8%, paraphrase robustness 40%. Tests A + B are the genuine capability numbers.
- **MIDI** (SS = 0.340) at S-size sampled: cat-match 30%, tonal 86%, directional 70-87%. At XL: FAIL (mode collapse). Documented exemplar of Factor B × D interaction.
- **Network** (SS = 0.126): VERDICT = FAIL. Framework-validation Outcome A.
- **DNA coding** (SS = 0.033): VERDICT = FAIL. Second low-SS framework validation.
- **RNA** (SS = 0.675): parse 100% (bracket balance — a real capability), but VERDICT = FAIL on prompt-conditional diversity. Surprising high-SS counter-example to naive SS monotonicity.

## Troubleshooting

- **All generations byte-identical within a category**: total mode collapse. Try: larger n_soft (128), different random seed, larger corpus, per-category class balance in labelling. Report as FAIL if remediation fails.
- **Generations are alphabet-invalid** (wrong characters): labeller–verifier mismatch on the alphabet. Verify domain model vocabulary aligns with the window encoding used in training.
- **Loss diverges**: LR too high for the fine-tune. Drop bridge LR to 2e-4 and domain-LR to 2e-5.
- **Greedy decoding looks great, temp-sampling looks bad**: unusual. Check greedy isn't collapsing to a single token that happens to match one class's ground truth. Report both and investigate.
- **XL domain model produces 2–24% unique generations regardless of prompt**: this is the documented XL failure mode. Drop to S-size; this is canonical. See `archive/xl-reverse-bridge-collapse/`.

## References

- Lester et al. 2021, "The Power of Scale for Parameter-Efficient Prompt Tuning" — soft-prompt conditioning
- Liu et al. 2023 — temperature sampling reference (used here at temp 0.8, top-k 40)
- Ziekenoppasser-Powell 2026 — the NBS paper

## Acknowledgements

The mandatory mode-collapse diagnostic gate emerged from the Network v1 / v2 debugging cycle in 2026-04. Without the gate, three of the eight PoCs would have been mis-reported as showing partial capability. This protocol codifies the gate so that future reverse-bridge work does not repeat the mistake.
