# Domain-Model Architecture — S / M / XL

Three scales, all character-level, all autoregressive (GPT-2-like). Trained from scratch per domain on the domain's canonical tokenised stream. All checkpoints are frozen during bridge training.

## Size comparison

| Scale | n_layer | n_head | d_model | d_ff | Params (approx) | Context | Training time per domain (A100) |
|---|---|---|---|---|---|---|---|
| XS | 2  | 2  | 128  | 512  | ~1 M       | 512 tokens | 30 s    |
| S  | 6  | 8  | 512  | 2048 | **~50 M**  | 512 tokens | 2.5 min |
| M  | 12 | 12 | 768  | 3072 | ~130 M     | 512 tokens | ~10 min |
| XL | 24 | 16 | 1024 | 4096 | ~380 M     | 512 tokens | ~45 min |

`MODEL_CONFIGS` in [`../experiment/code/config.py`](../experiment/code/config.py) matches this table for S / M / XL. XS is retained for dev / prototype work (e.g. the 2026-04-14 initial-release entropy survey ran at XS for early compute-budget reasons) but is **not** the scale used for any paper result.

Vocabulary: per-domain character set + 3 special tokens (`<pad>`, `<bos>`, `<eos>`). Typical alphabet size is 4 (DNA) to ~100 (MIDI).

## Training recipe

- Optimiser: AdamW, lr=3e-4 (S), 2e-4 (M), 1e-4 (XL), β=(0.9, 0.95), weight decay 0.1
- LR schedule: cosine decay to 10%, 500-step linear warmup
- Batch size: 256 (S), 128 (M), 64 (XL)
- Dropout: 0.1 on attention + residual streams
- Precision: bf16 mixed-precision on A100
- Stop criterion: val loss plateau (patience=3 epochs) or max 20 epochs
- Seed: 42

## Deployment during bridge training

**Forward bridge**: domain model frozen entirely. Bridge reads the top-layer hidden state at the final non-pad token and projects it into the LM's input embedding space (or, for BLIP-2 style, into a small bank of learned query embeddings — see `../bridges/forward/architecture.md`).

**Reverse bridge**: domain model is fine-tuned at 0.1× the bridge's LR. The bridge learns 64 soft prefix tokens which are prepended to the domain model's input; the full stack is trained jointly. Reverse bridges use S-size domain models only (see Factor B README for why).

## Validation loss reference points

See [`results/scaling_summary.md`](results/scaling_summary.md) for per-domain S best-val and S→XL deltas. Authoritative per-run numbers aggregate into `bridge_results_v2.json` (Drive-resident, pending Zenodo DOI on paper acceptance).

## Why ~50M for S

~50M is the operating point adopted as the S scale. Exact layer/dim values above are the canonical recipe; per-domain variation is documented in `train_s.py` arguments.

## Checkpoints

All S/M/XL checkpoints (per domain, per seed) are Drive-resident (~250 GB total) and will be mirrored to Zenodo on paper acceptance. Until the Zenodo DOI is live, reviewers needing specific checkpoints can contact the corresponding author ([danielzp.com](https://danielzp.com)).
