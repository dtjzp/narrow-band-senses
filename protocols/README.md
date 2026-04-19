# Protocols for protocols.io

Three protocols covering the core NBS methods, drafted in protocols.io format for publication with CC-BY licence and DOI-on-publish.

## Background on protocols.io

protocols.io is an open-access protocol repository. Protocols are published step-by-step, with a mandatory abstract, and get a DOI immediately on "Publish" (no external peer review unless co-submitted with a partner journal like Nature Protocols or GigaByte). Licensing is CC-BY.

Versioning: each version has its own DOI suffix (v1, v2, ...). Once published, a version cannot be deleted — only superseded. This means we publish the v1 after the paper is accepted, to avoid ratcheted corrections against pre-print claims.

Citation model: the paper's Methods section cites these protocols by DOI rather than duplicating full method text. This keeps the Methods section within Nature's 3000-word limit without sacrificing reproducibility.

## The three protocols

| # | Title | Draft file | Length (est. steps) |
|---|---|---|---|
| 1 | Entropy criterion measurement (structure score) | [`protocol_01_entropy_criterion.md`](protocol_01_entropy_criterion.md) | 12 |
| 2 | Forward bridge training (BLIP-2-style MLP projection) | [`protocol_02_forward_bridge_training.md`](protocol_02_forward_bridge_training.md) | 15 |
| 3 | Reverse bridge training (fine-tuned domain model + soft prefix) | [`protocol_03_reverse_bridge_training.md`](protocol_03_reverse_bridge_training.md) | 16 |

Each draft follows the canonical protocols.io template:

- Abstract
- Materials (ML-adapted: hardware, software versions, datasets with hashes, HuggingFace IDs)
- Before you begin
- Procedure (numbered steps, grouped into sections)
- Expected results
- Troubleshooting
- References

## Publishing timeline

These protocols are drafts. Actual publication will happen **after paper acceptance** so that:

1. DOI locks against a stable version of the method
2. The paper's Methods section can cite the protocol DOIs directly
3. Any clarifications emerging from peer review are incorporated into v1 (not retrofitted into a supposedly-locked v1)

Estimated time from draft → published: ~1 week (editor at protocols.io, pre-submission review for formatting, one round of light revision).

## Collection

The three protocols will be linked as a protocols.io **collection** so reviewers see them as a coherent pipeline. Collection title (proposed): "Narrow-Band Senses: Information-Theoretic Framework for Multimodal AI Perception — Protocol Set"

## Authorship

- First author: Daniel Ziekenoppasser-Powell
- Additional authors added if/when collaborator outreach converts (see `docs/superpowers/specs/2026-04-19-nbs-academic-collaboration-pitch-spec.md`).

## Pre-publication checklist

- [ ] Screenshots of expected results (loss curves, sample outputs) rendered at 2× resolution
- [ ] Code-pointer URLs resolving (Zenodo DOI for the repo snapshot; GitHub URL stable)
- [ ] Dataset availability: Zenodo DOI for paired corpora + checkpoints (pending)
- [ ] Ethics and licensing notes: no human-subject data (all signals are machine-generated or public sensor data)
- [ ] Collaborator additions before locking v1 (if any)
- [ ] Proofread: mandatory — CC-BY published forever
