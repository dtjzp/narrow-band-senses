# Data Sources for Factor A (29 Domains)

All data is publicly available or permissively licensed. Where a dataset has its own citation requirement, it is reproduced below.

## Biological / molecular

| Domain | Source | Licence | Citation |
|---|---|---|---|
| dna_coding, dna_noncoding | NCBI RefSeq | Public domain (US) | O'Leary et al. 2016, *Nucleic Acids Res.* |
| crispr | CRISPR-Cas gRNA library (public) | CC-BY | (per-library attribution in script header) |
| rna | Rfam 14.x WUSS notation | CC-BY | Kalvari et al. 2021, *Nucleic Acids Res.* |
| protein | UniProt SwissProt | CC-BY | UniProt Consortium 2023 |
| smiles | ZINC 20 | Free academic | Sterling & Irwin 2015, *JCIM* |
| bioreactor | Public bioreactor feed log (filtered) | CC-BY | (see per-domain script) |

## Sensor / signal

| Domain | Source | Licence | Citation |
|---|---|---|---|
| eeg | OpenNeuro ds003061 | CC-BY | Aricò et al. 2020 |
| hormonal | Public chronobiology time series | CC-BY | (see script) |
| das | Distributed acoustic sensing open release | CC-BY | (site-specific attribution) |
| seismo | IRIS seismological data | Public domain | IRIS/GEOFON |
| wifi_csi | SignFi WiFi CSI | CC-BY | Ma et al. 2018 |
| satellite | Copernicus Sentinel-2 | Free access (EU) | Copernicus Sentinel data |
| network | CICIDS2017 | Academic-use only | Sharafaldin et al. 2018 |
| traffic | Public city traffic log | CC-BY | (per-city attribution) |

## Audio / symbolic

| Domain | Source | Licence | Citation |
|---|---|---|---|
| whale | Watkins Marine Mammal Sound Database | CC-BY-NC | Watkins et al. 1998 (WHOI) |
| midi | ClassicalPianoMIDI (piano-midi.de) | Public domain | piano-midi.de |
| english | Project Gutenberg (Brown corpus subset) | Public domain | — |
| greek | Ancient Greek Papyri Corpus | CC-BY | (PHI 7) |

## Scientific-instrument / numerical

| Domain | Source | Licence | Citation |
|---|---|---|---|
| seti | Breakthrough Listen GBT data | Free academic | Lebofsky et al. 2019 |
| atc | FAA System Wide Information Management (filtered log) | Government work | FAA |
| quantum | Random-circuit corpus (OpenQASM 3) | MIT | (author-generated, reproducible via `gen_quantum.py`) |
| tidal | UK Tidal Prediction Service | Open Government | BODC |
| weather | NOAA ISD / MetOffice surface stations | Public domain (US) / Open Government (UK) | — |
| financial | Public historical equity time series | Public domain | — |
| game_scenes | SceneGraph open release | CC-BY | — |
| reactions | USPTO reaction SMILES | Public domain | Lowe 2017 |

## Domain subset used for bridges

Only a subset of the 29 domains appears in the bridge experiments (n=16 for forward, n=8 for reverse). Domain choice for bridges was driven by availability of a meaningful semantic description vocabulary (see Factor D), not by SS rank.

## Preprocessing

All domains are tokenised at character-level after domain-specific canonicalisation (e.g. MIDI note events encoded as printable ASCII via `chr(33 + note % 94)`; SMILES stripped of stereo annotations for the SS subset). Each canonical pipeline is documented in the per-domain script header in `../per-domain-scripts/`.

## Reproducibility

To regenerate the canonical corpora from scratch:

```bash
python per-domain-scripts/midi_entropy.py --regen   # example
```

(Data downloads may require API keys or manual registration for academic-use corpora.)
