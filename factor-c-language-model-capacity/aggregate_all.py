"""Phase 5 aggregation — flat results JSON + correlation tables.

Extended (2026-04-18) to include:
- Baseline controls (zero + random-seed, spec §3.3)
- S-model-size dimension (spec §3.5)
- Pythia-1B LM scaling (spec §3.1)

Outputs:
- G:/My Drive/nbs-bridge/results/bridge_results_v2.json — flat records
- Prints Spearman rho tables per (arch, source, lm, s_size, metric)
- Prints the cleanest new metric: rho(SS, zero - best) per LM scale
"""
from __future__ import annotations

import json, statistics
from pathlib import Path
from collections import defaultdict

RESULTS = Path('G:/My Drive/nbs-bridge/results')
# SS values from 1-research/nbs-survey/results/canonical_training_entropy.json (rounded to 3dp).
# Original 8-domain set used for S-bridge experiments; 8 new domains added 2026-04-18
# for n-extension to n=16 at XL-size (spec: 2026-04-18-nbs-n-extension-spec.md).
STRUCTURE_SCORES = {
    # Original 8 (S-bridge + XL-bridge + delta available)
    'whale': 0.773, 'tidal': 0.657, 'smiles': 0.553, 'english': 0.362,
    'gcode': 0.323, 'dna_coding': 0.033, 'financial': 0.011, 'seti': 0.001,
    # New 8 (XL-bridge only; no S-bridge — delta NOT available for these)
    'protein': 0.014, 'crispr': 0.362, 'dna_noncoding': 0.037, 'greek': 0.291,
    'python_code': 0.523, 'weather': 0.185, 'midi': 0.340, 'network': 0.126,
}

# Map filename LM-suffix fragments to canonical LM names.
LM_SUFFIX_TO_LM = {
    '': 'gpt2-small',
    'gpt2': 'gpt2-small',       # auto-derived suffix from --lm-model gpt2
    'gpt2m': 'gpt2-medium',
    'gpt2medium': 'gpt2-medium', # auto-derived
    'gpt2l': 'gpt2-large',
    'gpt2large': 'gpt2-large',
    'pythia1b': 'pythia-1b',
    'EleutherAIpythia1b': 'pythia-1b',  # paranoid
}

# Map lm_slug from baselines filenames to canonical LM names.
LM_SLUG_TO_LM = {
    'gpt2': 'gpt2-small',
    'gpt2medium': 'gpt2-medium',
    'gpt2large': 'gpt2-large',
    'pythia1b': 'pythia-1b',
}


def spearman(xs, ys):
    n = len(xs)
    if n < 3:
        return float('nan')
    def ranks(v):
        s = sorted(range(n), key=lambda i: v[i])
        r = [0.0]*n
        for i, idx in enumerate(s): r[idx] = i+1
        return r
    rx, ry = ranks(xs), ranks(ys)
    mx, my = sum(rx)/n, sum(ry)/n
    cov = sum((rx[i]-mx)*(ry[i]-my) for i in range(n))
    vx = sum((r-mx)**2 for r in rx); vy = sum((r-my)**2 for r in ry)
    return cov / ((vx*vy)**0.5 + 1e-12)


def parse_training_filename(stem):
    """Parse '{domain}_{source}[_lm_suffix][_sM]_training' → (domain, source, lm, s_size)."""
    stem = stem.replace('_training', '')
    s_size = 'S'
    # Detect s_size suffix at end. Order: longest suffixes first to avoid ambiguity.
    for sz in ('XL', 'XS', 'M', 'L'):
        if stem.endswith('_s' + sz):
            s_size = sz
            stem = stem[:-(len('_s' + sz))]
            break
    domain = None
    for d in STRUCTURE_SCORES:
        if stem.startswith(d + '_'):
            domain = d
            rest = stem[len(d) + 1:]
            break
    if domain is None:
        return None, None, None, None
    parts = rest.split('_')
    source = parts[0]
    # LM suffix may be empty or may include underscores (e.g. gpt2m or pythia1b)
    # In the existing suffix convention used by phase_large_lm_bridge.py, suffix starts with '_'
    # after the source. So rest of parts are joined.
    lm_suffix = '_'.join(parts[1:]) if len(parts) > 1 else ''
    lm = LM_SUFFIX_TO_LM.get(lm_suffix, lm_suffix or 'gpt2-small')
    return domain, source, lm, s_size


def collect_bridges():
    rows = []

    # CKA
    for f in (RESULTS / 'cka').glob('*.json'):
        try:
            r = json.loads(f.read_text(encoding='utf-8'))
            rows.append({
                'arch': 'cka', 'domain': r['domain'], 'source': r['data_source'],
                'ss': r['structure_score'], 'metric': r['cka_score'],
                'metric_name': 'cka_score', 'lm': 'n/a', 's_size': 'S',
            })
        except Exception as e:
            print(f'warn cka/{f.name}: {e}')

    # Linear + MLP + Q-Former training JSONs
    for arch_dir, arch in (('linear', 'linear'), ('mlp', 'mlp'), ('qformer', 'qformer')):
        for f in (RESULTS / arch_dir).glob('*_training.json'):
            try:
                r = json.loads(f.read_text(encoding='utf-8'))
                domain, source, lm, s_size = parse_training_filename(f.stem)
                if domain is None:
                    continue
                rows.append({
                    'arch': arch, 'domain': domain, 'source': source,
                    'ss': STRUCTURE_SCORES.get(domain, float('nan')),
                    'metric': r['best_val_loss'],
                    'metric_name': 'best_val_loss',
                    'improvement': r['training_history'][0]['val_loss'] - r['best_val_loss'],
                    'init_val': r['training_history'][0]['val_loss'],
                    'lm': lm,
                    's_size': s_size,
                    'best_epoch': r.get('best_epoch'),
                    'stopping_epoch': r.get('stopping_epoch'),
                    'bridge_params': r.get('bridge_params'),
                })
            except Exception as e:
                print(f'warn {arch_dir}/{f.name}: {e}')

    # Semantic claim-verification
    for arch in ('linear', 'mlp', 'qformer'):
        for f in (RESULTS / arch).glob('*_semantic_result.json'):
            try:
                r = json.loads(f.read_text(encoding='utf-8'))
                rows.append({
                    'arch': arch, 'domain': r['domain'], 'source': 'semantic',
                    'ss': r['structure_score'],
                    'metric': r['bridge_quality'],
                    'metric_name': 'bridge_quality',
                    'ceiling': r.get('ceiling_score'),
                    'efficiency': r.get('bridge_efficiency'),
                    'lm': 'gpt2-small', 's_size': 'S',
                })
            except Exception as e:
                print(f'warn {arch}/{f.name}: {e}')

    # Reverse bridge
    for f in (RESULTS / 'reverse').glob('*_reverse_training.json'):
        try:
            r = json.loads(f.read_text(encoding='utf-8'))
            # stem like: gcode_semantic_reverse_training OR gcode_semantic_v2ft_reverse_training
            stem = f.stem.replace('_reverse_training', '')
            parts = stem.split('_')
            variant = 'v1'
            if 'v2ft' in stem:
                variant = 'v2ft'
            rows.append({
                'arch': f'reverse_mlp_{variant}', 'domain': r['domain'],
                'source': r.get('source', 'semantic'),
                'ss': STRUCTURE_SCORES.get(r['domain'], float('nan')),
                'metric': r['best_val_loss'],
                'metric_name': 'best_val_loss',
                'improvement': r['training_history'][0]['val_loss'] - r['best_val_loss'],
                'init_val': r['training_history'][0]['val_loss'],
                'direction': 'text_to_domain',
                'lm': 'gpt2-small', 's_size': 'S',
                'variant': variant,
            })
        except Exception as e:
            print(f'warn reverse/{f.name}: {e}')

    return rows


def collect_baselines():
    """Load zero/random baselines and aggregate random seeds into mean/std."""
    raw = []
    base = RESULTS / 'baselines'
    if not base.exists():
        return raw
    for f in base.glob('*.json'):
        try:
            r = json.loads(f.read_text(encoding='utf-8'))
            raw.append(r)
        except Exception as e:
            print(f'warn baselines/{f.name}: {e}')
    # Aggregate random-mode: group by (domain, source, lm_model, s_size), mean/std over seeds.
    aggregated = []
    by_key = defaultdict(list)
    for r in raw:
        lm_canon = LM_SLUG_TO_LM.get(r['lm_model'].split('/')[-1].replace('-', '').replace('.', '').lower(),
                                      r['lm_model'])
        r['_lm_canon'] = lm_canon
        key = (r['domain'], r['source'], lm_canon, r.get('s_model_size', 'S'), r['mode'])
        by_key[key].append(r)
    for (domain, source, lm, s_size, mode), rs in by_key.items():
        vals = [x['val_loss'] for x in rs]
        row = {
            'arch': f'baseline_{mode}',
            'domain': domain, 'source': source,
            'ss': STRUCTURE_SCORES.get(domain, float('nan')),
            'lm': lm, 's_size': s_size, 'mode': mode,
            'val_loss': statistics.mean(vals),
            'val_loss_std': statistics.stdev(vals) if len(vals) > 1 else 0.0,
            'n_seeds': len(vals),
            'metric': statistics.mean(vals),
            'metric_name': f'baseline_{mode}_val',
        }
        aggregated.append(row)
    return aggregated


def joined_bridge_minus_baseline(bridges, baselines):
    """For each trained bridge row, attach matching zero/random baseline val_loss.

    Produces a 'bridge_contribution' = baseline - best_val metric.
    Only operates on MLP forward bridges with semantic source.
    """
    # Index baselines by (domain, source, lm, s_size, mode)
    bl = {}
    for b in baselines:
        bl[(b['domain'], b['source'], b['lm'], b['s_size'], b['mode'])] = b

    contributions = []
    for row in bridges:
        if row.get('arch') != 'mlp': continue
        if row.get('source') != 'semantic': continue
        if row.get('metric_name') != 'best_val_loss': continue
        key_zero = (row['domain'], row['source'], row['lm'], row['s_size'], 'zero')
        key_rand = (row['domain'], row['source'], row['lm'], row['s_size'], 'random')
        zero_b = bl.get(key_zero)
        rand_b = bl.get(key_rand)
        if zero_b:
            contributions.append({
                'arch': 'mlp_bridge_contrib',
                'domain': row['domain'], 'source': row['source'],
                'ss': row['ss'], 'lm': row['lm'], 's_size': row['s_size'],
                'metric_name': 'zero_minus_best',
                'metric': zero_b['val_loss'] - row['metric'],
                'zero_baseline': zero_b['val_loss'],
                'best_val': row['metric'],
            })
        if rand_b:
            contributions.append({
                'arch': 'mlp_bridge_contrib',
                'domain': row['domain'], 'source': row['source'],
                'ss': row['ss'], 'lm': row['lm'], 's_size': row['s_size'],
                'metric_name': 'random_minus_best',
                'metric': rand_b['val_loss'] - row['metric'],
                'random_baseline_mean': rand_b['val_loss'],
                'random_baseline_std': rand_b.get('val_loss_std'),
                'best_val': row['metric'],
            })
    return contributions


def summarise(rows):
    groups = defaultdict(list)
    for r in rows:
        if r.get('ss') != r.get('ss'): continue
        key = (r['arch'], r.get('source', '-'), r.get('lm', '-'), r.get('s_size', 'S'), r.get('metric_name', '?'))
        groups[key].append(r)

    print(f'\n{"arch":<22} {"source":<10} {"lm":<14} {"s":<3} {"metric":<22} {"n":>3} {"rho(SS,m)":>11}')
    print('-' * 100)
    for (arch, source, lm, s_size, metric_name), rs in sorted(groups.items()):
        if len(rs) < 3: continue
        xs = [r['ss'] for r in rs]
        ys = [r['metric'] for r in rs]
        rho = spearman(xs, ys)
        print(f'{arch:<22} {source:<10} {lm:<14} {s_size:<3} {metric_name:<22} {len(rs):>3} {rho:>+11.3f}')

        # Sub-metrics
        for m in ('improvement',):
            ym = [r[m] for r in rs if m in r]
            xm = [r['ss'] for r in rs if m in r]
            if len(ym) == len(xm) and len(ym) >= 3:
                rho_m = spearman(xm, ym)
                print(f'{" ":<22} {" ":<10} {" ":<14} {" ":<3} {"  " + m:<22} {len(ym):>3} {rho_m:>+11.3f}')


def save_summary(rows):
    out_path = RESULTS / 'bridge_results_v2.json'
    payload = {'rows': rows, 'n_rows': len(rows)}
    out_path.write_text(json.dumps(payload, indent=2, default=str), encoding='utf-8')
    print(f'\nwrote {len(rows)} rows to {out_path}')


if __name__ == '__main__':
    bridges = collect_bridges()
    baselines = collect_baselines()
    contributions = joined_bridge_minus_baseline(bridges, baselines)
    all_rows = bridges + baselines + contributions
    print(f'collected {len(bridges)} bridge rows, {len(baselines)} baseline rows, {len(contributions)} contribution rows')
    summarise(all_rows)
    save_summary(all_rows)
