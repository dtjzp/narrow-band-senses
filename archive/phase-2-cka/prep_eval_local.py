"""Local runner for phase34_prepare_eval.py, adapting paths from Colab to G: drive.

=== ARCHIVED — NOT RUNNABLE AS SHIPPED ===================================
Depends on `phase34_prepare_eval` (lived at `G:/My Drive/nbs-bridge/scripts/`)
and data/corpora paths under `G:/My Drive/` — none preserved in the public
repo. Kept alongside run_phase2_local.py as a historical record of the
Phase 2 CKA null result. See README.md in this directory.
==========================================================================
"""
import sys, json, pathlib

raise SystemExit(
    "archive/phase-2-cka/prep_eval_local.py is an archived script; the "
    "phase34_prepare_eval module and the G:/My Drive data paths it uses "
    "were not preserved in the public repo. See README.md for context."
)

BASE = pathlib.Path('G:/My Drive/nbs-bridge')
DATA_DIR = pathlib.Path('G:/My Drive/nbs-survey/data')
EXP2_DATA = pathlib.Path('G:/My Drive/nbs-experiment-2/data')
CONTINUATION_LEN = 50

DOMAINS = ['whale', 'tidal', 'smiles', 'english', 'gcode', 'dna_coding', 'financial', 'seti']
SOURCES = ['natural', 'synthetic']

def load_text(domain):
    p = EXP2_DATA / f'{domain}_1M.txt'
    if not p.exists():
        p = DATA_DIR / f'{domain}_1M.txt'
    return p.read_text(encoding='utf-8')

for arch in ['mlp']:
    for d in DOMAINS:
        for s in SOURCES:
            gen = BASE / f'results/{arch}/{d}_{s}_generated.jsonl'
            if not gen.exists():
                print(f'{arch}/{d}/{s}: MISSING')
                continue
            text = load_text(d)
            out_records = []
            with open(gen, encoding='utf-8') as f:
                for line in f:
                    r = json.loads(line)
                    start = r['seq_start_idx']
                    end = start + len(r['sequence'])
                    cont = text[end:end + CONTINUATION_LEN]
                    out_records.append({
                        'domain': d, 'source': s, 'arch': arch,
                        'seq_idx': r['seq_idx'],
                        'sequence': r['sequence'],
                        'continuation': cont,
                        'target_description': r['target_description'],
                        'generated_description': r['generated_description'],
                    })
            outp = BASE / f'results/{arch}/{d}_{s}_eval_prepared.jsonl'
            with open(outp, 'w', encoding='utf-8') as f:
                for rec in out_records:
                    f.write(json.dumps(rec, ensure_ascii=False) + '\n')
            print(f'{arch}/{d}/{s}: {len(out_records)} records → {outp.name}')
