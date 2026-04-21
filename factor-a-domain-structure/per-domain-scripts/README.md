# Per-domain entropy scripts

Each script in this directory computes Shannon entropy and the derived
*structure score* on one domain's character stream, and writes
`results/{domain}_entropy.json`.

## Running one

```bash
python factor-a-domain-structure/per-domain-scripts/quantum_entropy.py
```

Each script is self-contained (imports shared helpers from `../entropy.py`)
and writes its own result JSON under `results/`. The headline ρ correlation
is then reconstructed from the canonical aggregate at
`factor-a-domain-structure/results/canonical_training_entropy.json`.

## Canonical structure scores vs `primary_structure_score`

Some per-domain scripts compute structure scores on **multiple sub-corpora**
and emit a top-level `primary_structure_score` that may differ from the
value reported in `canonical_training_entropy.json`. This is by design —
not a bug — but the mismatch has been a source of confusion for fresh
reviewers, so it is documented explicitly here.

### Quantum

`quantum_entropy.py` generates three corpora and reports all three:

| Field in `quantum_entropy.json`          | Corpus                                     | Structure score |
|------------------------------------------|--------------------------------------------|-----------------|
| `random_circuits.structure_score`        | Random circuits only (width 2–10, d 2–20)  | ~0.696          |
| `structured_circuits.structure_score`    | Algorithmic circuits (QFT, VQE, Grover, …) | ~0.637          |
| `pooled.structure_score`                 | Random + structured                        | ~0.651          |
| `primary_structure_score` *(top-level)*  | Alias for `pooled.structure_score`         | ~0.651          |

`canonical_training_entropy.json` lists `quantum.structure_score ≈ 0.698`,
which corresponds to the **random-circuits sub-corpus**, computed on the
training-data stream actually used by `factor-b-domain-model-capacity/train_s.py`
(hence the small numerical drift between 0.696 and 0.698 — the canonical
computation uses the exact training stream the models saw, while the
per-domain script's `random_circuits` entry uses the concatenated corpus
before train/val/test split).

### Other domains

- *[Document here as audited. Most single-corpus domains
  (e.g. `english_entropy.py`, `python_code_entropy.py`) have no
  sub-corpus split and `primary_structure_score ==
  canonical_training_entropy.json[domain].structure_score` up to rounding.]*

## Which value should I cite?

- **To reproduce the paper's headline correlation** (ρ across 29 domains),
  use `canonical_training_entropy.json`. These are the values the paper's
  figures and regressions are computed from.
- **To validate a per-domain pipeline end-to-end**, compare the per-domain
  script's output against the matching sub-corpus field (for quantum, that
  is `random_circuits.structure_score`, *not* `primary_structure_score`).
