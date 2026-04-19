"""
Hormonal Cycle Entropy Analysis
================================
NBS Survey domain: hormonal_cycle

Data: iurteaga/hmc realistic_dataset.pickle
- Shape: (60 cycles, 300 time steps, 5 hormones)
- Channels: LH, FSH, E2 (estradiol), P4 (progesterone), Ih (inhibin)
- Source: Clark ODE model of HPO axis (validated mechanistic simulation)
- Reference: Urteaga et al. MLHC 2019 (arXiv:1908.10226)

This file is the MLHC 2019 supplementary dataset from the hormonal menstrual
cycle ML project. It contains 60 menstrual cycle segments, each 300 time steps,
simulated from the Clark et al. differential delay equations describing the
HPO axis feedback loop. The model is validated against real clinical hormone data.

Note on data availability:
Real daily urinary hormone datasets (Marquette NFP, PhysioNet mcPHASES)
require institutional registration or DUA agreements. The Clark ODE simulation
used here is the standard reference for HPO axis dynamics research.

Time resolution: 300 steps per cycle. A typical menstrual cycle is ~28 days;
at daily resolution this would be ~300/28 ≈ 10.7 cycles per 300-step segment.
The simulation runs at sub-daily resolution (likely 12-hour or 6-hour steps)
based on the LH surge pattern spanning ~2-3 steps.

Encoding:
- Each hormone channel treated separately (per-channel analysis)
- 20 equal-frequency quantile bins -> chars A-T
- Each cycle treated as separate sequence (pooled counting, no cross-cycle concat)
- Also combined: concatenate all cycles per channel into one stream
"""

import sys
import os
import json
import pickle
import numpy as np
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from entropy import (
    compute_ngram_counts,
    compute_entropy_miller_madow,
    compute_conditional_entropy,
    compute_structure_score,
    compute_sequential_score,
    shuffle_control,
    compute_mutual_information,
    compute_ngram_counts_pooled,
    compute_conditional_entropy_pooled,
)

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data', 'hormonal')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), 'data', 'processed')
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

PICKLE_PATH = os.path.join(DATA_DIR, 'hmc_realistic_dataset.pickle')
RESULTS_PATH = os.path.join(RESULTS_DIR, 'hormonal_entropy.json')
TEXT_LOCAL = os.path.join(PROCESSED_DIR, 'hormonal_1M.txt')
TEXT_GDRIVE = 'G:/My Drive/nbs-survey/data/hormonal_1M.txt'

CHANNEL_NAMES = ['LH', 'FSH', 'E2', 'P4', 'Ih']
N_BINS = 20
ALPHABET = 'ABCDEFGHIJKLMNOPQRST'  # 20 chars for 20 bins


# ── Download data if needed ────────────────────────────────────────────────────
def download_data():
    if os.path.exists(PICKLE_PATH):
        print(f'Data already exists: {PICKLE_PATH}')
        return
    url = 'https://raw.githubusercontent.com/iurteaga/hmc/master/data/mlhc_2019/realistic_dataset.pickle'
    print(f'Downloading from {url} ...')
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=60) as resp:
        raw = resp.read()
    with open(PICKLE_PATH, 'wb') as f:
        f.write(raw)
    print(f'  Saved {len(raw):,} bytes to {PICKLE_PATH}')


# ── Load and inspect ───────────────────────────────────────────────────────────
def load_data():
    with open(PICKLE_PATH, 'rb') as f:
        data = pickle.load(f)
    # Shape: (n_cycles, n_timesteps, n_channels)
    assert data.ndim == 3, f'Expected 3D array, got {data.ndim}D'
    n_cycles, n_steps, n_ch = data.shape
    print(f'Dataset shape: {n_cycles} cycles x {n_steps} steps x {n_ch} channels')
    print(f'Channels: {CHANNEL_NAMES}')
    for i, name in enumerate(CHANNEL_NAMES):
        vals = data[:, :, i].flatten()
        print(f'  {name}: min={vals.min():.2f}, max={vals.max():.2f}, mean={vals.mean():.2f}')
    return data


# ── Discretisation ─────────────────────────────────────────────────────────────
def discretise_channel(cycles_arr, n_bins=N_BINS):
    """
    cycles_arr: shape (n_cycles, n_steps) — one hormone channel
    Returns: list of strings, one per cycle
    """
    # Compute global quantile bin edges across all cycles and time steps
    flat = cycles_arr.flatten()
    # Equal-frequency (quantile) bins
    percentiles = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.percentile(flat, percentiles)
    # Make bin edges unique (handle degenerate case)
    bin_edges = np.unique(bin_edges)
    actual_bins = len(bin_edges) - 1

    encoded_cycles = []
    for cycle in cycles_arr:  # (n_steps,)
        # np.digitize: bin index 1..actual_bins -> map to 0..actual_bins-1
        idx = np.digitize(cycle, bin_edges[1:-1], right=True)  # 0..actual_bins-1
        idx = np.clip(idx, 0, actual_bins - 1)
        # Map to chars (if actual_bins < 20, use subset of alphabet)
        chars = ''.join(ALPHABET[i] for i in idx)
        encoded_cycles.append(chars)
    return encoded_cycles, actual_bins


# ── Per-channel entropy analysis ───────────────────────────────────────────────
def analyse_channel(cycles_arr, name):
    """Compute entropy metrics for one hormone channel."""
    print(f'\n  Channel: {name}')
    encoded_cycles, actual_bins = discretise_channel(cycles_arr)

    n_chars_total = sum(len(s) for s in encoded_cycles)
    print(f'    Cycles: {len(encoded_cycles)}, total chars: {n_chars_total:,}, bins: {actual_bins}')

    # Pooled H0, H2, H3 (no cross-cycle concatenation)
    counts1 = compute_ngram_counts_pooled(encoded_cycles, 1)
    h0 = compute_entropy_miller_madow(counts1)

    h2 = compute_conditional_entropy_pooled(encoded_cycles, 2)
    h3 = compute_conditional_entropy_pooled(encoded_cycles, 3)

    structure_score = max(0.0, min(1.0, 1 - h3 / h0)) if h0 > 0 else 0.0
    print(f'    H0={h0:.4f}, H2={h2:.4f}, H3={h3:.4f}, structure={structure_score:.4f}')

    # Concatenated stream (for MI and sequential score)
    concat = ''.join(encoded_cycles)

    # Sequential score
    seq_score = compute_sequential_score(concat)
    print(f'    Sequential score: {seq_score:.4f}')

    # MI decay
    lags = [1, 2, 5, 10, 50]
    mi_profile = {}
    for lag in lags:
        mi = compute_mutual_information(concat, lag)
        mi_profile[str(lag)] = round(mi, 6)
        print(f'    MI(lag={lag})={mi:.4f}')

    # Shuffled control
    shuf = shuffle_control(concat)
    shuf_h3 = compute_conditional_entropy(shuf, 3)
    shuf_struct = max(0.0, min(1.0, 1 - shuf_h3 / h0)) if h0 > 0 else 0.0
    print(f'    Shuffled structure: {shuf_struct:.4f}')

    return {
        'h0': round(h0, 6),
        'h2': round(h2, 6),
        'h3': round(h3, 6),
        'structure_score': round(structure_score, 6),
        'sequential_score': round(seq_score, 6),
        'mi_profile': mi_profile,
        'n_chars': n_chars_total,
        'alphabet_size': actual_bins,
        'shuffled_structure_score': round(shuf_struct, 6),
    }


# ── Combined multi-channel analysis ───────────────────────────────────────────
def analyse_combined(data):
    """
    Interleave all 5 channels into a single stream per cycle, then pool.
    Each time step contributes 5 chars (one per hormone channel).
    """
    print('\n  Combined (interleaved 5-channel) stream...')
    n_cycles, n_steps, n_ch = data.shape

    # Encode each channel with global quantile bins
    all_encoded = []  # all_encoded[ch][cycle] = encoded string
    for ch_i in range(n_ch):
        encoded, _ = discretise_channel(data[:, :, ch_i])
        all_encoded.append(encoded)

    # Interleave: for each cycle, zip chars across channels at each time step
    interleaved_cycles = []
    for cyc_i in range(n_cycles):
        chars = []
        for t in range(n_steps):
            for ch_i in range(n_ch):
                chars.append(all_encoded[ch_i][cyc_i][t])
        interleaved_cycles.append(''.join(chars))

    n_chars_total = sum(len(s) for s in interleaved_cycles)
    print(f'    Total chars: {n_chars_total:,}')

    counts1 = compute_ngram_counts_pooled(interleaved_cycles, 1)
    h0 = compute_entropy_miller_madow(counts1)
    h2 = compute_conditional_entropy_pooled(interleaved_cycles, 2)
    h3 = compute_conditional_entropy_pooled(interleaved_cycles, 3)
    structure_score = max(0.0, min(1.0, 1 - h3 / h0)) if h0 > 0 else 0.0
    alphabet_size = len(counts1)

    concat = ''.join(interleaved_cycles)
    seq_score = compute_sequential_score(concat)

    lags = [1, 2, 5, 10, 50]
    mi_profile = {}
    for lag in lags:
        mi = compute_mutual_information(concat, lag)
        mi_profile[str(lag)] = round(mi, 6)

    shuf = shuffle_control(concat)
    shuf_h3 = compute_conditional_entropy(shuf, 3)
    shuf_struct = max(0.0, min(1.0, 1 - shuf_h3 / h0)) if h0 > 0 else 0.0

    print(f'    H0={h0:.4f}, H2={h2:.4f}, H3={h3:.4f}')
    print(f'    Structure: {structure_score:.4f}, Sequential: {seq_score:.4f}')
    print(f'    Shuffled structure: {shuf_struct:.4f}')

    return {
        'h0': round(h0, 6),
        'h2': round(h2, 6),
        'h3': round(h3, 6),
        'structure_score': round(structure_score, 6),
        'sequential_score': round(seq_score, 6),
        'mi_profile': mi_profile,
        'n_chars': n_chars_total,
        'alphabet_size': alphabet_size,
        'shuffled_structure_score': round(shuf_struct, 6),
    }


# ── Save processed text ────────────────────────────────────────────────────────
def save_processed_text(data):
    """Save LH channel encoded as 1M char text (pad/truncate as needed)."""
    n_cycles, n_steps, n_ch = data.shape
    encoded, _ = discretise_channel(data[:, :, 0])  # LH channel
    concat = ''.join(encoded)

    # Tile to 1M if needed
    target = 1_000_000
    if len(concat) < target:
        repeats = target // len(concat) + 1
        concat = (concat * repeats)[:target]
    else:
        concat = concat[:target]

    with open(TEXT_LOCAL, 'w') as f:
        f.write(concat)
    print(f'  Saved {len(concat):,} chars to {TEXT_LOCAL}')

    # Try Google Drive path
    gdrive_dir = os.path.dirname(TEXT_GDRIVE)
    if os.path.exists(gdrive_dir):
        with open(TEXT_GDRIVE, 'w') as f:
            f.write(concat)
        print(f'  Also saved to {TEXT_GDRIVE}')
    else:
        print(f'  Google Drive not accessible ({gdrive_dir} not found), skipping')


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print('=== Hormonal Cycle Entropy Analysis ===\n')

    # 1. Get data
    download_data()
    data = load_data()
    n_cycles, n_steps, n_ch = data.shape

    # 2. Per-channel analysis
    print('\n--- Per-Channel Analysis ---')
    per_channel = {}
    for ch_i, name in enumerate(CHANNEL_NAMES):
        result = analyse_channel(data[:, :, ch_i], name)
        per_channel[name] = result

    # 3. Combined interleaved stream
    print('\n--- Combined Analysis ---')
    combined = analyse_combined(data)

    # 4. Canonical metrics: per-channel mean (primary, matching bioreactor/ATC approach)
    mean_struct = np.mean([v['structure_score'] for v in per_channel.values()])
    mean_h0 = np.mean([v['h0'] for v in per_channel.values()])
    mean_h2 = np.mean([v['h2'] for v in per_channel.values()])
    mean_h3 = np.mean([v['h3'] for v in per_channel.values()])
    mean_seq = np.mean([v['sequential_score'] for v in per_channel.values()])
    mean_n_chars = sum(v['n_chars'] for v in per_channel.values())

    print(f'\n--- Per-Channel Means ---')
    print(f'  Mean structure score: {mean_struct:.4f}')
    print(f'  Mean H0: {mean_h0:.4f}, H2: {mean_h2:.4f}, H3: {mean_h3:.4f}')

    # Use LH MI profile as the canonical profile (most biologically relevant)
    lh_mi = per_channel['LH']['mi_profile']
    # Add lag 100 if data is long enough
    concat_lh = ''.join(discretise_channel(data[:, :, 0])[0])
    if len(concat_lh) > 100:
        mi_100 = compute_mutual_information(concat_lh, 100)
        lh_mi['100'] = round(mi_100, 6)

    # 5. Save processed text
    print('\n--- Saving Processed Text ---')
    save_processed_text(data)

    # 6. Build output JSON (mandatory schema)
    # Use per-channel mean as canonical h0/h2/h3/structure/sequential
    # (matching bioreactor precedent where per-channel is the primary measure)
    output = {
        'domain': 'hormonal_cycle',
        'h0': round(float(mean_h0), 6),
        'h2': round(float(mean_h2), 6),
        'h3': round(float(mean_h3), 6),
        'structure_score': round(float(mean_struct), 6),
        'sequential_score': round(float(mean_seq), 6),
        'mi_profile': {k: v for k, v in sorted(lh_mi.items(), key=lambda x: int(x[0]))},
        'n_chars': int(mean_n_chars),
        'alphabet_size': N_BINS,
        'data_source': (
            'iurteaga/hmc realistic_dataset.pickle (MLHC 2019). '
            'Clark ODE model simulation of HPO axis (60 cycles x 300 steps x 5 channels). '
            'Channels: LH, FSH, E2, P4, Ih. '
            'Validated mechanistic simulation; real data (Marquette NFP, PhysioNet mcPHASES) '
            'requires institutional DUA agreements not available for automated download.'
        ),
        'encoding_notes': (
            '20 equal-frequency quantile bins per channel (global bins across all cycles). '
            'Each cycle encoded as separate sequence; pooled counting without cross-cycle concatenation. '
            'Canonical metric = per-channel mean structure score (matching bioreactor precedent). '
            'Combined interleaved stream (5 chars/timestep) also computed for comparison.'
        ),
        'per_channel': {
            name: {
                'h0': v['h0'],
                'h3': v['h3'],
                'structure_score': v['structure_score'],
                'sequential_score': v['sequential_score'],
                'shuffled_structure_score': v['shuffled_structure_score'],
                'mi_lag1': v['mi_profile'].get('1', None),
            }
            for name, v in per_channel.items()
        },
        'combined_interleaved': {
            'h0': combined['h0'],
            'h3': combined['h3'],
            'structure_score': combined['structure_score'],
            'sequential_score': combined['sequential_score'],
            'n_chars': combined['n_chars'],
            'alphabet_size': combined['alphabet_size'],
        },
        'n_cycles': int(n_cycles),
        'n_timesteps_per_cycle': int(n_steps),
    }

    with open(RESULTS_PATH, 'w') as f:
        json.dump(output, f, indent=2)
    print(f'\nResults saved to: {RESULTS_PATH}')

    # Summary
    print('\n=== SUMMARY ===')
    print(f'Per-channel structure scores:')
    for name, v in per_channel.items():
        print(f'  {name}: {v["structure_score"]:.4f}  (shuffled: {v["shuffled_structure_score"]:.4f})')
    print(f'\nCanonical per-channel mean structure score: {mean_struct:.4f}')
    print(f'Combined interleaved stream structure score: {combined["structure_score"]:.4f}')
    print(f'\nExpected: 0.4-0.7 (HIGH, similar to tidal 0.658 / bioreactor 0.749)')

    return output


if __name__ == '__main__':
    main()
