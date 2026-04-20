# Post-overhaul fixes (2026-04-19)

Companion to commit `95df28f` ("Overhaul to current paper state"). After the overhaul landed on branch `overhaul-to-paper-state-2026-04`, a static integrity audit (Spec 1) was run against the branch and found 11 blockers. This file summarises the fix-up pass: what was fixed, what's deferred, and why.

The full audit report lives in the author's private workspace at `docs/superpowers/reports/2026-04-19-nbs-repo-integrity-audit-report.md`.

## Fixed in this pass

### Artefacts shipped

- **`bridges/reverse/multi-seed-report.md`** — added. Source of the 5-seed Stable PASS numbers for Quantum + Python cited in their REPORTs.
- **`bridges/reverse/atc/*.json`** — 8 files added (decision_checkpoint, summary, scorecards × 3, mode_collapse_diagnostics × 3). Previously only `REPORT.md` was present, leaving the per-seed audit trail empty.
- **`bridges/reverse/reactions/*.json`** — 8 files added, same structure as ATC.
- **`bridges/reverse/bioreactor/README.md`** — added. Bioreactor is cited in the three-branch failure typology but was not previously represented as a directory. The README documents why it's FAIL-with-no-full-PoC (Spec 3 v1 legacy) and points to paper §6.2 + H_win/H₀ data.
- **`factor-d-paired-description-quality/semantic-labellers/README.md`** and **`claim-verifiers/README.md`** — added. Those directories were previously empty. The READMEs explain that the labellers live in the author's Drive workspace pending Zenodo deposit and point to the labeller template + corresponding-author contact.

### Broken paths / imports fixed

- **`factor-a-domain-structure/per-domain-scripts/*_entropy.py`** (16 files) — all `sys.path.insert(0, ...nbs-experiment...)` replaced with `sys.path.insert(0, str(Path(__file__).resolve().parents[1]))` so `entropy.py` (at `factor-a-domain-structure/entropy.py`) is importable from the per-domain scripts. `nbs-experiment/` does not exist in the public repo; the old pattern was a workspace-layout assumption from the private tree.
- **`bridges/reverse/compute_window_entropy.py`** — hardcoded `parents[2] / "1-research" / "nbs-survey" / "data" / "processed"` read paths and `... / "nbs-bridge" / "window_entropy_results.json"` write path all replaced with script-local defaults + `--data-dir` / `--out` / `--check-only` CLI args. The `--check-only` mode loads the pre-computed results JSON (which ships alongside the script) and prints the table without needing the raw streams.
- **`factor-b-domain-model-capacity/train_s.py`** — three hardcoded `/content/drive/MyDrive/...` Colab paths replaced. `sys.path.insert` now points at repo root; imports changed from bare `config`/`model`/`dataset` to `experiment.code.config`/`experiment.code.model`/`experiment.code.dataset`. `DATA_DIR` and `CKPT_DIR` are now CLI args with local defaults.
- **`experiment/__init__.py`** and **`experiment/code/__init__.py`** — added. Makes `experiment.code` a proper Python package so `from experiment.code.config import ...` resolves when called from anywhere under the repo root.

### Documentation corrections

- **`CITATION.cff`** — `url:` field changed from literal placeholder `https://github.com/<user>/nbs-bridge-public` to `https://github.com/dtjzp/narrow-band-senses`.
- **`bridges/reverse/mode-collapse-diagnostics.md`** — "8-PoC sequence" → "10-PoC sequence" (the gate was introduced mid-8-PoC but the current repo reflects 10 PoCs).
- **`paper-figures/README.md`** — two corrections: §4.2 forward bridge line said "8 domains" → corrected to "16 domains"; §4.5 reverse-bridge line said "8 domains" → corrected to "10 domains". Added a prominent paper-caption → repo-filename mapping at the top, flagging the three "pending generation" figures (fig3_reverse_bridge_10poc, fig4_three_branch_typology, fig5_ss_trl_playbook).
- **`factor-d-paired-description-quality/README.md`** — "beyond 8 domains" → "beyond 10 domains".
- **`README.md`** (top-level) — the "30-minute reproduce" section rewritten to be honest about what's shipped vs pending Zenodo. Script requires `bpc_per_domain.json` which is NOT shipped (pending Zenodo deposit — each BPC value is ~1 A100-hour of training). When both inputs are present the correlation is ~30 s; until Zenodo is live, BPC values available from corresponding author. Added the `python bridges/reverse/compute_window_entropy.py --check-only` one-liner for the H_win/H₀ result which IS fully reproducible locally.
- **10 `bridges/reverse/*/REPORT.md`** — the "_(Original long REPORT archived at `1-research/nbs-bridge/paper-reports/...`)_" lines removed from all 7 REPORTs that had them (leaked internal paper-development paths into the public artefact).

### Dependencies

- **`requirements.txt`** — added `h5py`, `qiskit`, `requests`, `librosa`, `mido` (all needed by various `factor-a/per-domain-scripts/*_entropy.py` scripts for the `--recompute-ss` path). `mido` was previously commented out as "optional" but `midi_entropy.py` imports it unguarded at module top level — would ImportError on any default install. `rdkit` remains commented (no script currently imports it).

## Deferred — needs user resolution

### Substantive discrepancies

- **S-model architecture mismatch**: `factor-b-domain-model-capacity/architecture.md` states S = n_layer=6, d_model=512, ~50M params. `experiment/code/config.py` defines S = n_layers=4, d_model=256 (~4M params). The paper text uses "S ~50M." **The doc and the code disagree about what "S" means.** One of them is wrong; someone needs to check the actual checkpoints used for the paper's Factor B results. Did NOT touch either file in this pass — this is a judgement call the author needs to make.

- **`bpc_per_domain.json` is not shipped.** The 30-minute reproduce path is predicated on pre-computed BPC values, and they're not in the repo. Three options:
  1. Ship the JSON (requires extracting BPC numbers from paper Table 1 or equivalent, creating a JSON manually).
  2. Ship a script that computes BPC from raw streams + character-transformer checkpoints (requires shipping the checkpoints too).
  3. Leave as-is and wait for Zenodo deposit.
  The README has been reworded honestly in this pass; the actual reproducibility gap remains pending.

### Three figures pending generation

`fig3_reverse_bridge_10poc.pdf`, `fig4_three_branch_typology.pdf`, `fig5_ss_trl_playbook.pdf` — referenced in the paper's Results section but not yet composed. Flagged in `paper-figures/README.md`. Generation spec lives in the author's workspace (`docs/superpowers/specs/2026-04-19-nbs-figure-composition-spec.md`); the figure-composition work is a ~1-day effort on its own.

### experiment/code/train.py bare imports

`experiment/code/train.py` uses bare `from config import ...` / `from model import ...` / `from dataset import ...`. These work when CWD is `experiment/code/` but not otherwise. For consistency with the package-ification done here, these should become `from .config import ...` (or `from experiment.code.config import ...`). Not touched in this pass to minimise surface of changes; the script is still runnable with `cd experiment/code && python train.py`.

### Documentation still pointing at `G:/My Drive/...`

- `factor-b/architecture.md:33, 41` — authoritative per-run numbers live on Drive.
- `factor-b/README.md:32` — M/XL training scripts on Drive.
- `factor-c/README.md:40` — `bridge_results_v2.json` on Drive.
- `factor-c/aggregate_all.py:19` — RESULTS path hardcoded to `G:/My Drive/nbs-bridge/results`.
- `factor-d/README.md:40` — labeller code on Drive (now has a semantic-labellers/README.md pointing at this).
- `factor-d/results/source_comparison_summary.md:7` — numbers regenerate from Drive.
- `archive/phase-3-4-*/*.py` (20 files) — archived scripts contain hardcoded G: paths. The archive READMEs are honest that these weren't ported.

All of these represent artefacts that live on Drive and will be mirrored to Zenodo on acceptance. A single Zenodo DOI in each referenced location would close the loop. Not done in this pass — one placeholder DOI would need to apply everywhere, and we don't have it yet.

### Stale worked examples

- `bridges/reverse/mode-collapse-diagnostics.md` Quantum worked example lists categories `1-qubit/entangling/deep/random` — the actual quantum PoC has `parameterised/highly-entangled/measurement-heavy/entangling/single-qubit`. Cosmetic; not touched.
- `bridges/reverse/architecture.md` MIDI category list `simple_tonal/complex_tonal/atonal/percussive` — actual MIDI categories are `tonal/ascending-run/descending-run/leap-dominant/mixed`. Cosmetic; not touched.

## What the audit rated clean (not touched)

- Reverse-bridge REPORT numerical consistency with shipped scorecards (quantum single-seed, python v2 single-seed, gcode, midi, smiles, rna, network, dna_coding all match; ATC + reactions now have shipped scorecards too).
- `canonical_training_entropy.json` internal consistency (H₃/H₀ ratios match SS column to 4 decimals).
- `window_entropy_results.json` — consistent with repo README's three-branch typology description.
- LICENSE dual-licensing (MIT code + CC-BY 4.0 docs/figures).
- `archive/` READMEs honestly describe each failed-approach branch.
- AI tool disclosure in top-level README matches paper disclosure.

## For the next session

Before this branch merges to `main` and before the GitHub URL is sent to Brown:

1. Resolve the S-model architecture discrepancy (paper claim vs `config.py` — see "Substantive discrepancies" above).
2. Either ship `bpc_per_domain.json` or confirm the README's Zenodo-pending framing is acceptable to stand.
3. Consider whether the three pending figures need to be composed before submission or can follow as a separate commit.
4. Spec 2 (execution reproducibility) is running in parallel — its findings may surface additional blockers that this pass didn't address.

---

## Second pass — 2026-04-20

All four "deferred" items above now resolved in commit on branch `post-overhaul-fixes-2026-04-20`:

### Item 1 — S-model architecture mismatch: FIXED in code

Git history confirmed: `experiment/code/config.py` was last touched in the 2026-04-14 initial release and still carried the pre-bridge-era model sizes. `factor-b-domain-model-capacity/architecture.md` is the authoritative version (matches paper text).

`MODEL_CONFIGS` updated:

| Scale | Before (stale) | After (paper) |
|---|---|---|
| XS | n_layers=2, d_model=128, n_heads=2 (~1M) | unchanged (retained as dev/prototype size, not in paper) |
| S  | n_layers=4, d_model=256, n_heads=4 (~4M) | **n_layers=6, d_model=512, n_heads=8 (~50M)** |
| M  | n_layers=8, d_model=512, n_heads=8 (~50M) | **n_layers=12, d_model=768, n_heads=12 (~130M)** |
| L/XL | L: n_layers=12, d_model=768, n_heads=12 (~130M) | **renamed to XL: n_layers=24, d_model=1024, n_heads=16 (~380M)** |

Param-count comments added inline. "L" key removed — paper uses "XL" for the 380M scale.

### Item 2 — `bpc_per_domain.json`: SHIPPED

File existed at `1-research/nbs-survey/shared-denominator-defence/bpc_per_domain.json` (in the author's private tree) — transcribed from paper Tables S2a, S2b, and §S6. Copied to `factor-a-domain-structure/results/bpc_per_domain.json`. `_source_note` field in the JSON documents the derivation from paper sources.

`reproduce_rho.py` had a minor mismatch — it expected a `bpc` key; the JSON uses `norm_bpc` and `raw_bpc`. Script updated to prefer `norm_bpc` (matches paper's ρ = −0.92 claim), fall back to `bpc`, and skip `_source_note` metadata keys.

**Verified end-to-end**: `python factor-a-domain-structure/reproduce_rho.py` now prints:

```
Spearman rho(SS, BPC) = -0.9236
p-value                = 9.293e-13
n domains              = 29
[ok] wrote .../paper-figures/fig_main_ss_correlation.png + .pdf
```

Matches paper's ρ = −0.92 to two decimals. **30-second reproduce path now works out of a fresh clone with no external dependencies.**

### Item 3 — fig3/4/5 pending generation: SHIPPED

All five paper-caption figures exist in the author's scaffold at `1-research/nbs-bridge-public/paper-figures/`. Copied into the clone:

- `figure1_entropy_compressibility.{pdf,png}` (also from `3-drafts/`)
- `fig2_forward_bridge_universality.{pdf,png}`
- `fig3_reverse_bridge_10poc.{pdf,png}`
- `fig4_three_branch_typology.{pdf,png}`
- `fig5_ss_trl_playbook.{pdf,png}`

Plus the `compose_fig{1..5}.py` generation scripts — so each paper-caption figure is reproducible from its source data.

Note: earlier figure copies (`fig_main_ss_correlation`, `fig_bridge_contribution`, etc.) are retained because other docs (`factor-c/README.md`, `paper-figures/README.md` section headings) reference them. `paper-figures/README.md` paper-caption-mapping table updated to reflect the dual naming.

### Item 4 — `experiment/code/train.py` bare imports: FIXED

Bare `from config import ...`, `from model import ...`, `from dataset import ...`, `from evaluate import ...` replaced with `from experiment.code.{config,model,dataset,evaluate} import ...`. Script is now importable / runnable from the repo root, consistent with `factor-b/train_s.py`'s usage. Other files in `experiment/code/` don't have bare imports that need the same fix (verified via grep).

## Still deferred (post-merge work)

- `factor-b/architecture.md` documentation uses "S/M/XL" naming; `experiment/code/config.py` now matches for S/M/XL but retains "XS" as a dev size. Minor cosmetic alignment possible but not load-bearing.
- Per-domain entropy JSONs (12 of 29) still missing individually under `factor-a/results/per-domain/` — the aggregate `canonical_training_entropy.json` has all 29, so this is cosmetic completeness rather than a blocker.
- Archive `phase-2-cka/run_phase2_local.py` still imports a `phase2_cka` module not in the repo. Archive READMEs should flag this but don't yet. Low priority — archive is explicitly historical.
- Stale worked-example categories in `mode-collapse-diagnostics.md` + `architecture.md` (MIDI category list). Cosmetic; not blocking.
