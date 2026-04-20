"""Central configuration for NBS Experiment 2.

All constants in one place: domains, model configs, training hyperparams.
"""

from pathlib import Path

# ── Paths ──

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
EXP1_DIR = PROJECT_ROOT.parent / "nbs-experiment"
EXP1_RESULTS = EXP1_DIR / "results"

# ── Domains ──

DOMAINS = [
    "protein", "dna_coding", "dna_noncoding", "smiles", "crispr",
    "english", "greek", "python_code", "gcode", "financial", "weather",
]

HOLDOUT_DOMAIN = "weather"  # Pre-registered holdout for prediction validation

# Domains where sequences are discrete (use SEP token, mask during eval)
DISCRETE_DOMAINS = {"protein", "smiles", "crispr", "dna_coding", "dna_noncoding"}

# ── Data ──

TARGET_CHARS = 1_000_000  # Non-SEP characters per domain
TRAIN_FRAC = 0.8
VAL_FRAC = 0.1
TEST_FRAC = 0.1
RANDOM_SEED = 42

# ── Model Configs ──

# Sizes S / M / XL are the canonical scales used in the paper (matches
# factor-b-domain-model-capacity/architecture.md). XS is retained as a
# small prototype size for dev runs; not referenced in the paper.
MODEL_CONFIGS = {
    "XS": {"n_layers": 2,  "d_model": 128,  "n_heads": 2},   # ~1M  (dev only)
    "S":  {"n_layers": 6,  "d_model": 512,  "n_heads": 8},   # ~50M (paper "S")
    "M":  {"n_layers": 12, "d_model": 768,  "n_heads": 12},  # ~130M (paper "M")
    "XL": {"n_layers": 24, "d_model": 1024, "n_heads": 16},  # ~380M (paper "XL")
}

DEFAULT_CONTEXT_LEN = 512

# ── Training ──

TRAIN_CONFIG = {
    "lr": 3e-4,
    "weight_decay": 0.1,
    "betas": (0.9, 0.95),
    "max_epochs": 10,
    "patience": 3,
    "batch_size": 64,
    "dropout": 0.1,
    "grad_clip": 1.0,
}

# ── Seeds ──

# 3 seeds for most configs, 5 for model M
SEEDS_DEFAULT = [42, 137, 256]
SEEDS_MODEL_M = [42, 137, 256, 512, 1024]

# ── Experiment 1 H0 values (for normalisation) ──

H0_VALUES = {
    "protein": 4.1798,
    "dna_coding": 1.9948,
    "dna_noncoding": 1.9969,
    "smiles": 3.5964,
    "crispr": 1.9922,
    "english": 4.0810,
    "greek": 4.1059,
    "python_code": 4.7016,
    "gcode": 3.8060,
    "financial": 3.3225,
    "weather": 3.4542,
}

# Sequential structure scores from Experiment 1 (for Marchetti correlation)
# Formula: 1 - (H3 / H2_shuffled) — isolates sequential dependencies from alphabet effects
# Values from Experiment 1 entropy_table.csv "sequential_score" column
SEQ_STRUCTURE_SCORES = {
    "protein": 0.0063,
    "dna_coding": 0.0381,
    "dna_noncoding": 0.0368,
    "smiles": 0.4704,
    "crispr": 0.0003,
    "english": 0.3487,
    "greek": 0.2939,
    "python_code": 0.4575,
    "gcode": 0.3136,
    "financial": 0.0285,
    "weather": 0.2011,
}
