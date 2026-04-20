# Semantic labellers — pending Zenodo deposit

The per-domain semantic labellers used to construct the BAL-1280 to
BAL-2400 semantic paired corpora are **not shipped in the public
repository at this stage**. They live in the author's Drive workspace
at `G:/My Drive/nbs-bridge/scripts/{domain}_semantic_labeller.py` and
will be mirrored to Zenodo on paper acceptance.

Labellers are per-domain (gcode, midi, smiles, python, quantum, rna,
network, dna_coding, atc, reactions) and follow the template in
[`../labeller_template.md`](../labeller_template.md). Each labeller:

1. Reads a raw domain stream (`{domain}_1M.txt`)
2. Segments into windows
3. Assigns each window to one of N categories (domain-specific, 4–5)
4. Writes a paired-description corpus (one category label per window)
5. Is self-consistent (round-trip ≥ 95% gate)

Reviewers needing a labeller for a specific domain prior to the
Zenodo release can contact the corresponding author
([danielzp.com](https://danielzp.com)).

The labeller template at `../labeller_template.md` specifies the
interface contract and methodology; any domain-specific labeller can
be reconstructed from the template + a spot-check against the
domain's committed per-PoC `decision_checkpoint.json` (which shows
category distributions).
