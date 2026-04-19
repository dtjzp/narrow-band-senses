"""Run Phase 2 CKA locally on CPU (adapts paths from Colab to G: drive)."""
import sys
from pathlib import Path

# Point at G: drive paths (Drive Desktop mount)
G = Path('G:/My Drive')

# Add survey code directory to path
sys.path.insert(0, str(G / 'nbs-survey/code'))
# Add nbs-bridge/scripts for phase2_cka imports
sys.path.insert(0, str(G / 'nbs-bridge/scripts'))

import phase2_cka as p2

# Monkey-patch paths before running
p2.BASE = G / 'nbs-bridge'
p2.CKPT_DIR = p2.BASE / 'checkpoints'
p2.PAIRS_DIR = p2.BASE / 'paired_data'
p2.OUT_DIR = p2.BASE / 'results/cka'
p2.OUT_DIR.mkdir(parents=True, exist_ok=True)

import torch
p2.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'device: {p2.DEVICE}')

if __name__ == '__main__':
    import argparse, time
    parser = argparse.ArgumentParser()
    parser.add_argument('--domains', nargs='+', default=None)
    parser.add_argument('--sources', nargs='+', default=None)
    args = parser.parse_args()

    print('loading GPT-2 small ...', flush=True)
    from transformers import GPT2Model, GPT2TokenizerFast
    gpt2_tok = GPT2TokenizerFast.from_pretrained('gpt2')
    gpt2_model = GPT2Model.from_pretrained('gpt2').to(p2.DEVICE)
    gpt2_model.eval()

    domains = args.domains if args.domains else p2.DOMAINS
    sources = args.sources if args.sources else p2.SOURCES

    results = []
    print(f'\n{"domain":<12} {"source":<10} {"n":>4} {"CKA":>8} {"time":>7}')
    for d in domains:
        for s in sources:
            t0 = time.time()
            r = p2.run(d, s, gpt2_model, gpt2_tok)
            dt = time.time() - t0
            if 'error' in r:
                print(f'{d:<12} {s:<10} ERROR: {r["error"]}')
                continue
            print(f'{d:<12} {s:<10} {r["n_samples"]:>4} {r["cka_score"]:>8.4f} {dt:>6.1f}s', flush=True)
            results.append(r)

    print('\n=== summary ===')
    for src in sources:
        print(f'\n{src}:')
        for r in sorted([x for x in results if x['data_source'] == src], key=lambda x: -x['structure_score']):
            print(f'  SS={r["structure_score"]:.3f}  {r["domain"]:<12}  CKA={r["cka_score"]:.4f}')
