"""
DAS (Distributed Acoustic Sensing) Entropy Analysis
=====================================================
NBS Survey domain: das_seismic

Data: PoroTomo Brady's Geothermal Field vertical DAS array
- Shape: (30000 time samples, 384 channels)
- Sample rate: 1000 Hz, duration: 30 seconds
- Records M3.4 earthquake at Brady Hot Springs, NV (March 2016)
- Silixa iDAS interrogator, 1.021 m channel spacing, 10 m gauge length
- Source: DOE Geothermal Data Repository (https://gdr.openei.org/submissions/848)

This is a pre-registered prediction experiment for the proxy sensor diagnostic:
  P1: Per-channel structure score will be moderate-to-high (physical waveform smoothness)
  P2: Pooled-vs-per-channel gap will exist (cross-channel spatial structure)
  P3: Transformer scaling will be flat (learning carrier, not earthquake source)
"""

import sys
import os
import json
import numpy as np
import h5py

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
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data', 'das')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), 'data', 'processed')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

H5_PATH = os.path.join(DATA_DIR, 'porotomo_dasv_eq1.h5')
RESULTS_PATH = os.path.join(RESULTS_DIR, 'das_entropy.json')
TEXT_LOCAL = os.path.join(PROCESSED_DIR, 'das_1M.txt')
TEXT_GDRIVE = 'G:/My Drive/nbs-survey/data/das_1M.txt'

N_BINS = 20
ALPHABET = 'ABCDEFGHIJKLMNOPQRST'
# Sample a subset of channels for per-channel analysis (all 384 is fine, they're short)
N_SAMPLE_CHANNELS = 384  # use all


def load_data():
    """Load DAS data from HDF5."""
    f = h5py.File(H5_PATH, 'r')
    data = f['DasRawData/RawData'][:]  # (30000, 384)
    f.close()
    print(f'Loaded DAS data: {data.shape} (samples x channels)')
    print(f'  Value range: [{data.min():.4f}, {data.max():.4f}]')
    return data


def discretise_channel(trace, global_bin_edges=None):
    """Quantise a single channel trace into N_BINS characters."""
    if global_bin_edges is None:
        percentiles = np.linspace(0, 100, N_BINS + 1)
        bin_edges = np.percentile(trace, percentiles)
        bin_edges = np.unique(bin_edges)
    else:
        bin_edges = global_bin_edges

    actual_bins = len(bin_edges) - 1
    idx = np.digitize(trace, bin_edges[1:-1], right=True)
    idx = np.clip(idx, 0, actual_bins - 1)
    chars = ''.join(ALPHABET[i] for i in idx)
    return chars, actual_bins


def analyse_channel(trace, ch_id, global_bin_edges=None):
    """Compute entropy metrics for one DAS channel."""
    encoded, actual_bins = discretise_channel(trace, global_bin_edges)

    counts1 = compute_ngram_counts(encoded, 1)
    h0 = compute_entropy_miller_madow(counts1)
    h2 = compute_conditional_entropy(encoded, 2)
    h3 = compute_conditional_entropy(encoded, 3)

    structure_score = max(0.0, min(1.0, 1 - h3 / h0)) if h0 > 0 else 0.0
    seq_score = compute_sequential_score(encoded)

    # Shuffled control
    shuf = shuffle_control(encoded)
    shuf_h3 = compute_conditional_entropy(shuf, 3)
    shuf_struct = max(0.0, min(1.0, 1 - shuf_h3 / h0)) if h0 > 0 else 0.0

    # MI at a few lags
    mi_lag1 = compute_mutual_information(encoded, 1)

    return {
        'h0': round(h0, 6),
        'h2': round(h2, 6),
        'h3': round(h3, 6),
        'structure_score': round(structure_score, 6),
        'sequential_score': round(seq_score, 6),
        'shuffled_structure_score': round(shuf_struct, 6),
        'mi_lag1': round(mi_lag1, 6),
        'n_chars': len(encoded),
        'actual_bins': actual_bins,
    }


def analyse_pooled(data, global_bin_edges):
    """Encode all channels, then pool for aggregate metrics."""
    print('\n--- Pooled Stream Analysis ---')

    n_samples, n_channels = data.shape
    encoded_channels = []
    for ch in range(n_channels):
        enc, _ = discretise_channel(data[:, ch], global_bin_edges)
        encoded_channels.append(enc)

    # Pooled entropy (treating each channel as a separate sequence)
    counts1 = compute_ngram_counts_pooled(encoded_channels, 1)
    h0 = compute_entropy_miller_madow(counts1)
    h2 = compute_conditional_entropy_pooled(encoded_channels, 2)
    h3 = compute_conditional_entropy_pooled(encoded_channels, 3)

    structure_score = max(0.0, min(1.0, 1 - h3 / h0)) if h0 > 0 else 0.0

    # Concatenated stream for MI and sequential score
    concat = ''.join(encoded_channels)
    seq_score = compute_sequential_score(concat)

    # MI profile
    lags = [1, 2, 5, 10, 50, 100, 500]
    mi_profile = {}
    for lag in lags:
        if lag < len(concat) // 2:
            mi = compute_mutual_information(concat, lag)
            mi_profile[str(lag)] = round(mi, 6)

    # Shuffled
    shuf = shuffle_control(concat)
    shuf_h3 = compute_conditional_entropy(shuf, 3)
    shuf_struct = max(0.0, min(1.0, 1 - shuf_h3 / h0)) if h0 > 0 else 0.0

    n_chars = sum(len(s) for s in encoded_channels)

    print(f'  H0={h0:.4f}, H2={h2:.4f}, H3={h3:.4f}')
    print(f'  Pooled structure score: {structure_score:.4f}')
    print(f'  Sequential score: {seq_score:.4f}')
    print(f'  Shuffled structure: {shuf_struct:.4f}')
    print(f'  Total chars: {n_chars:,}')

    return {
        'h0': round(h0, 6),
        'h2': round(h2, 6),
        'h3': round(h3, 6),
        'structure_score': round(structure_score, 6),
        'sequential_score': round(seq_score, 6),
        'shuffled_structure_score': round(shuf_struct, 6),
        'mi_profile': mi_profile,
        'n_chars': n_chars,
    }, encoded_channels


def analyse_interleaved(data, global_bin_edges):
    """Interleave all channels at each timestep into a single stream."""
    print('\n--- Interleaved Stream Analysis ---')

    n_samples, n_channels = data.shape

    # Encode each channel
    encoded_channels = []
    for ch in range(n_channels):
        enc, _ = discretise_channel(data[:, ch], global_bin_edges)
        encoded_channels.append(enc)

    # Interleave: at each timestep, emit one char per channel
    interleaved = []
    for t in range(n_samples):
        for ch in range(n_channels):
            interleaved.append(encoded_channels[ch][t])
    interleaved = ''.join(interleaved)

    counts1 = compute_ngram_counts(interleaved, 1)
    h0 = compute_entropy_miller_madow(counts1)
    h2 = compute_conditional_entropy(interleaved, 2)
    h3 = compute_conditional_entropy(interleaved, 3)

    structure_score = max(0.0, min(1.0, 1 - h3 / h0)) if h0 > 0 else 0.0
    seq_score = compute_sequential_score(interleaved)

    print(f'  H0={h0:.4f}, H2={h2:.4f}, H3={h3:.4f}')
    print(f'  Interleaved structure score: {structure_score:.4f}')
    print(f'  Total chars: {len(interleaved):,}')

    return {
        'h0': round(h0, 6),
        'h2': round(h2, 6),
        'h3': round(h3, 6),
        'structure_score': round(structure_score, 6),
        'sequential_score': round(seq_score, 6),
        'n_chars': len(interleaved),
    }


def save_processed_text(encoded_channels):
    """Save concatenated per-channel text, tiled to 1M chars."""
    concat = ''.join(encoded_channels)
    target = 1_000_000
    if len(concat) < target:
        repeats = target // len(concat) + 1
        concat = (concat * repeats)[:target]
    else:
        concat = concat[:target]

    with open(TEXT_LOCAL, 'w') as f:
        f.write(concat)
    print(f'  Saved {len(concat):,} chars to {TEXT_LOCAL}')

    gdrive_dir = os.path.dirname(TEXT_GDRIVE)
    if os.path.exists(gdrive_dir):
        with open(TEXT_GDRIVE, 'w') as f:
            f.write(concat)
        print(f'  Also saved to {TEXT_GDRIVE}')


def main():
    print('=== DAS Seismic Entropy Analysis ===')
    print('Pre-registered predictions:')
    print('  P1: Per-channel structure moderate-to-high (waveform smoothness)')
    print('  P2: Pooled > per-channel gap (cross-channel structure)')
    print('  P3: Transformer scaling flat (proxy sensor effect)')
    print()

    # 1. Load data
    data = load_data()
    n_samples, n_channels = data.shape

    # 2. Compute global bin edges (across all channels)
    flat = data.flatten()
    percentiles = np.linspace(0, 100, N_BINS + 1)
    global_bin_edges = np.percentile(flat, percentiles)
    global_bin_edges = np.unique(global_bin_edges)
    actual_global_bins = len(global_bin_edges) - 1
    print(f'\nGlobal quantile bins: {actual_global_bins} (target: {N_BINS})')

    # 3. Per-channel analysis
    print('\n--- Per-Channel Analysis ---')
    per_channel = {}
    struct_scores = []
    for ch in range(n_channels):
        result = analyse_channel(data[:, ch], ch, global_bin_edges)
        per_channel[str(ch)] = result
        struct_scores.append(result['structure_score'])
        if ch < 5 or ch % 50 == 0:
            print(f'  Ch {ch}: struct={result["structure_score"]:.4f}, '
                  f'shuffled={result["shuffled_structure_score"]:.4f}, '
                  f'MI(1)={result["mi_lag1"]:.4f}')

    struct_scores = np.array(struct_scores)
    mean_struct = struct_scores.mean()
    print(f'\n  Per-channel mean structure: {mean_struct:.4f}')
    print(f'  Per-channel min: {struct_scores.min():.4f}, max: {struct_scores.max():.4f}')
    print(f'  Per-channel std: {struct_scores.std():.4f}')

    # 4. Pooled analysis
    pooled, encoded_channels = analyse_pooled(data, global_bin_edges)

    # 5. Interleaved analysis
    interleaved = analyse_interleaved(data, global_bin_edges)

    # 6. Save processed text
    print('\n--- Saving Processed Text ---')
    save_processed_text(encoded_channels)

    # 7. Canonical metrics (per-channel mean, matching bioreactor/WiFi CSI approach)
    mean_h0 = np.mean([v['h0'] for v in per_channel.values()])
    mean_h3 = np.mean([v['h3'] for v in per_channel.values()])
    mean_seq = np.mean([v['sequential_score'] for v in per_channel.values()])

    # 8. Build output JSON
    output = {
        'domain': 'das_seismic',
        'h0': round(float(mean_h0), 6),
        'h3': round(float(mean_h3), 6),
        'structure_score': round(float(mean_struct), 6),
        'sequential_score': round(float(mean_seq), 6),
        'mi_profile': pooled['mi_profile'],
        'n_chars': pooled['n_chars'],
        'alphabet_size': actual_global_bins,
        'data_source': (
            'PoroTomo Brady Hot Springs vertical DAS array (DOE GDR submission 848). '
            'Silixa iDAS025, 384 channels, 1.021 m spacing, 1000 Hz, 30 s recording. '
            'M3.4 earthquake, March 2016. Public domain DOE data.'
        ),
        'encoding_notes': (
            f'{actual_global_bins} equal-frequency quantile bins (global across all channels). '
            'Each channel encoded as separate sequence; pooled counting without cross-channel concatenation. '
            'Canonical metric = per-channel mean structure score (matching WiFi CSI / bioreactor precedent). '
            'Interleaved stream (384 chars/timestep) also computed for cross-channel structure comparison.'
        ),
        'per_channel_summary': {
            'mean_structure': round(float(mean_struct), 6),
            'min_structure': round(float(struct_scores.min()), 6),
            'max_structure': round(float(struct_scores.max()), 6),
            'std_structure': round(float(struct_scores.std()), 6),
            'n_channels': n_channels,
        },
        'pooled': pooled,
        'interleaved': interleaved,
        'predictions': {
            'P1_perchannel_moderate_to_high': None,  # fill after measurement
            'P2_pooled_gt_perchannel': None,
            'P3_scaling_flat': 'pending_colab',
        },
        'n_samples': int(n_samples),
        'n_channels': int(n_channels),
        'sample_rate_hz': 1000,
        'duration_s': 30,
    }

    # Evaluate predictions
    output['predictions']['P1_perchannel_moderate_to_high'] = (
        f'{"CONFIRMED" if mean_struct > 0.3 else "DISCONFIRMED"}: '
        f'mean per-channel structure = {mean_struct:.4f}'
    )
    output['predictions']['P2_pooled_gt_perchannel'] = (
        f'{"CONFIRMED" if pooled["structure_score"] > mean_struct else "DISCONFIRMED"}: '
        f'pooled={pooled["structure_score"]:.4f} vs per-channel mean={mean_struct:.4f}, '
        f'gap={pooled["structure_score"] - mean_struct:.4f}'
    )

    with open(RESULTS_PATH, 'w') as f:
        json.dump(output, f, indent=2)
    print(f'\nResults saved to: {RESULTS_PATH}')

    # Summary
    print('\n=== SUMMARY ===')
    print(f'Per-channel mean structure: {mean_struct:.4f}')
    print(f'Pooled structure: {pooled["structure_score"]:.4f}')
    print(f'Interleaved structure: {interleaved["structure_score"]:.4f}')
    print(f'Gap (pooled - per-channel): {pooled["structure_score"] - mean_struct:.4f}')
    print()
    print('Prediction results:')
    for k, v in output['predictions'].items():
        print(f'  {k}: {v}')

    return output


if __name__ == '__main__':
    main()
