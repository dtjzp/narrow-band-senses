"""Character-transformer training stack.

Modules:
- config: MODEL_CONFIGS (XS/S/M/L), TRAIN_CONFIG, domain lists
- model: CharTransformer architecture
- dataset: CharDataset, build_vocab, encode_text
- train: train_one_run training loop with early stopping
- evaluate: compute_bpc evaluation

Used by:
- factor-b-domain-model-capacity/train_s.py (S-size domain checkpoints)
- factor-c-language-model-capacity/aggregate_all.py (scale comparisons)
- factor-a-domain-structure/per-domain-scripts/*.py (BPC evaluation)
"""
