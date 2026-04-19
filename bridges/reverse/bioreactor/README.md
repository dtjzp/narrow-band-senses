# Bioreactor reverse-bridge PoC — not fully shipped

This PoC is cited in the three-branch failure typology as an
exemplar of the **within-window-attractor** failure mode at extreme
SS. H_win/H₀ = 0.29 is the lowest in the 11-domain set, reflecting
near-constant within-window vocabulary.

- SS: **0.931** (highest of the 11 PoCs)
- Verdict: **FAIL** (mode-collapse diagnostic rejected on all 3 seeds)
- Failure mode: **within-window-attractor**
- Corpus: 199 pairs (below the 960 floor that was adopted after this run)
- Per-category result: 3/4 categories byte-identical across seeds
- H_win/H₀: 0.292 — see `../window_entropy_results.json` line for `bioreactor`

## Why there's no full REPORT here

The bioreactor PoC was Spec 3 v1 (pre-Spec 3 v2 methodology
refinement). Its result confounded data-scarcity (199 pairs) with
high-SS saturation. After the H_win/H₀ metric settled the question
(0.292 is consistent with attractor collapse, not data-scarcity
confound), the rescue-spec was withdrawn and bioreactor is reported
as an informative FAIL in the paper's §6.2 Limitations — *not* as a
full PoC in §4.3.

The scorecards + diagnostics artefacts live in the author's private
tree (`1-research/nbs-bridge/poc/bioreactor/`) pending Zenodo deposit.
They are referenced in paper-development records but are not paper-
headline data.

See:
- Paper §4.3 Pattern 3 (within-window-attractor branch)
- Paper §6.2 Limitations #2
- `../../README.md` §Three-branch failure typology
- `spec3-v2-report.md` in the author's private tree for the
  comparison with ATC + reactions
