"""Traffic flow entropy analysis — UK National Highways WebTRIS API.

Dataset: A1M/2259B (Southbound), Active site, hourly vehicle count
Source: https://webtris.highwaysengland.co.uk/api/v1
Period: January 2023 - December 2023 (12 months, ~8,760 hourly observations)

Steps:
1. Download 15-minute interval "Total Volume" data from WebTRIS month by month.
2. Aggregate to hourly counts; interpolate small gaps.
3. Discretise to 20 quantile bins → character stream A-T.
4. Run NBS entropy analysis.
5. Save results to results/traffic_entropy.json.
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT   = Path(__file__).resolve().parents[2]
DATA_DIR    = REPO_ROOT / "1-research" / "nbs-survey" / "data" / "traffic"
RESULTS_DIR = REPO_ROOT / "1-research" / "nbs-survey" / "results"
RAW_CSV     = DATA_DIR / "webtris_a1m_2023_raw.csv"
HOURLY_CSV  = DATA_DIR / "webtris_a1m_2023_hourly.csv"
OUT_PATH    = RESULTS_DIR / "traffic_entropy.json"

DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Entropy module
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from entropy import (
    compute_ngram_counts,
    compute_entropy_miller_madow,
    compute_conditional_entropy,
    compute_structure_score,
    compute_sequential_score,
    compute_structure_scores_bootstrap,
    mi_decay_profile,
    shuffle_control,
)

# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

BASE_URL = "https://webtris.highwaysengland.co.uk/api/v1"
SITE_ID  = "2"   # A1M/2259B Southbound, Active

# Month ranges: (start_ddmmyyyy, end_ddmmyyyy) — full year 2023
MONTHS = [
    ("01012023", "31012023"),
    ("01022023", "28022023"),
    ("01032023", "31032023"),
    ("01042023", "30042023"),
    ("01052023", "31052023"),
    ("01062023", "30062023"),
    ("01072023", "31072023"),
    ("01082023", "31082023"),
    ("01092023", "30092023"),
    ("01102023", "31102023"),
    ("01112023", "30112023"),
    ("01122023", "31122023"),
]


def fetch_month(site: str, start: str, end: str) -> list[dict]:
    """Fetch one month of 15-min interval data. page_size=4000 covers 31*96=2976 rows."""
    url = (
        f"{BASE_URL}/reports/daily"
        f"?sites={site}&start_date={start}&end_date={end}"
        f"&page=1&page_size=4000"
    )
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    data = r.json()
    return data.get("Rows", [])


def download_data() -> pd.DataFrame:
    """Download or load cached raw data."""
    if RAW_CSV.exists():
        print(f"    [cache] Loading raw data from {RAW_CSV.name}")
        return pd.read_csv(RAW_CSV)

    print(f"    Downloading WebTRIS site {SITE_ID} (A1M/2259B), Jan-Dec 2023 ...")
    all_rows = []
    for i, (start, end) in enumerate(MONTHS):
        print(f"      Month {i+1:02d}/12: {start[:2]}/{start[2:4]}/{start[4:]} ...", end=" ")
        rows = fetch_month(SITE_ID, start, end)
        nonempty = [r for r in rows if r.get("Total Volume", "") not in ("", None)]
        print(f"{len(rows)} rows, {len(nonempty)} with data")
        all_rows.extend(rows)
        time.sleep(0.3)

    df = pd.DataFrame(all_rows)
    df.to_csv(RAW_CSV, index=False)
    print(f"    Saved {len(df):,} rows to {RAW_CSV.name}")
    return df


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def preprocess_to_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """Parse WebTRIS DataFrame into clean hourly vehicle counts."""
    # Build datetime: combine Report Date + Time Period Ending
    # Report Date format: '2023-06-01T00:00:00'
    # Time Period Ending: '00:14:00'
    df = df.copy()

    date_part = pd.to_datetime(df["Report Date"], errors="coerce").dt.date
    time_part = pd.to_datetime(df["Time Period Ending"], format="%H:%M:%S", errors="coerce")
    if time_part.isna().all():
        time_part = pd.to_datetime(df["Time Period Ending"], format="%H:%M", errors="coerce")

    df["datetime"] = pd.to_datetime(
        date_part.astype(str) + " " + df["Time Period Ending"].astype(str),
        errors="coerce",
    )

    # Parse Total Volume
    df["volume"] = pd.to_numeric(df["Total Volume"], errors="coerce")

    df = df[["datetime", "volume"]].dropna(subset=["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)

    n_raw = len(df)
    n_valid = df["volume"].notna().sum()
    print(f"    Raw 15-min intervals: {n_raw:,} (valid volume: {n_valid:,})")

    # Resample to hourly: sum 4 x 15-min intervals per hour
    df = df.set_index("datetime")
    # Use minimum 2 valid observations per hour to accept the sum
    hourly_sum   = df["volume"].resample("h").sum(min_count=2)
    hourly_count = df["volume"].resample("h").count()

    hourly = pd.DataFrame({
        "datetime": hourly_sum.index,
        "count":    hourly_sum.values,
        "n_obs":    hourly_count.values,
    })
    # Mark as NaN if fewer than 2 out of 4 observations
    hourly.loc[hourly["n_obs"] < 2, "count"] = np.nan

    return hourly.reset_index(drop=True)


def fill_gaps(df: pd.DataFrame, max_gap_hours: int = 3) -> pd.DataFrame:
    """Reindex to full hourly grid and interpolate small gaps."""
    full_idx = pd.date_range(df["datetime"].min(), df["datetime"].max(), freq="h")
    df_full = df.set_index("datetime")[["count"]].reindex(full_idx)
    df_full.index.name = "datetime"

    n_missing = df_full["count"].isna().sum()
    print(f"    Missing hourly slots before interpolation: {n_missing:,} / {len(df_full):,}")

    df_full["count"] = df_full["count"].interpolate(
        method="linear", limit=max_gap_hours, limit_direction="both"
    )
    n_remaining = df_full["count"].isna().sum()
    print(f"    Missing after interpolation:               {n_remaining:,}")

    df_full = df_full.dropna().reset_index()
    return df_full


# ---------------------------------------------------------------------------
# Discretisation
# ---------------------------------------------------------------------------

def discretise_to_20bins(series: pd.Series) -> str:
    """Quantile-discretise into 20 bins → chars A-T."""
    CHARS = "ABCDEFGHIJKLMNOPQRST"

    # qcut with 20 equal-occupancy bins; handle ties at 0 (quiet hours)
    q, bins = pd.qcut(series, q=20, labels=False, duplicates="drop", retbins=True)
    n_actual = q.nunique()

    if n_actual < 20:
        print(f"    Note: {n_actual} distinct quantile bins (ties expected at zero-count hours)")
        # Re-rank so values are dense 0..(n_actual-1)
        mapping = {v: i for i, v in enumerate(sorted(q.dropna().unique()))}
        q = q.map(mapping)
        q = (q * 19 / max(n_actual - 1, 1)).round().astype(int).clip(0, 19)

    q = q.fillna(0).astype(int).clip(0, 19)
    stream = "".join(CHARS[i] for i in q)
    return stream


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    t0 = time.time()

    # ---- Step 1: Download / load ----
    print("[1] Downloading / loading WebTRIS data ...")
    raw_df = download_data()
    print(f"    Raw rows loaded: {len(raw_df):,}")

    # ---- Step 2: Preprocess ----
    print("[2] Preprocessing to hourly counts ...")
    hourly = preprocess_to_hourly(raw_df)
    print(f"    Hourly rows: {len(hourly):,}")
    hourly = fill_gaps(hourly)
    hourly.to_csv(HOURLY_CSV, index=False)
    print(f"    Final hourly rows: {len(hourly):,}")
    print(f"    Date range: {hourly['datetime'].min()} to {hourly['datetime'].max()}")
    count_stats = hourly["count"].describe()
    print(f"    Count stats: min={count_stats['min']:.0f}, "
          f"max={count_stats['max']:.0f}, "
          f"mean={count_stats['mean']:.1f}, "
          f"std={count_stats['std']:.1f}")

    # ---- Step 3: Discretise ----
    print("[3] Discretising to 20 quantile bins ...")
    stream = discretise_to_20bins(hourly["count"])
    n_chars = len(stream)
    alphabet = sorted(set(stream))
    n_alpha = len(alphabet)
    print(f"    Stream length: {n_chars:,} chars ({n_chars/24/7:.1f} weeks)")
    print(f"    Alphabet size: {n_alpha} unique chars")
    print(f"    First 72 chars: {stream[:72]}")

    # Bin distribution
    bin_dist = pd.Series(list(stream)).value_counts().sort_index()
    print(f"    Bin distribution: {dict(bin_dist)}")

    # ---- Step 4: Entropy analysis ----
    print("[4] Computing entropy metrics ...")

    uni_counts = compute_ngram_counts(stream, 1)
    h0 = compute_entropy_miller_madow(uni_counts)
    h_max = math.log2(20)
    print(f"    H0 (unigram): {h0:.4f} bits  (max = {h_max:.4f})")

    h2 = compute_conditional_entropy(stream, order=2)
    h3 = compute_conditional_entropy(stream, order=3)
    h4 = compute_conditional_entropy(stream, order=4)
    print(f"    H(X|1-gram):  {h2:.4f} bits")
    print(f"    H(X|2-gram):  {h3:.4f} bits")
    print(f"    H(X|3-gram):  {h4:.4f} bits")

    struct_score = compute_structure_score(stream)
    seq_score    = compute_sequential_score(stream)
    print(f"    Structure score:  {struct_score:.4f}")
    print(f"    Sequential score: {seq_score:.4f}")

    # Shuffled control
    shuffled_stream = shuffle_control(list(stream))
    shuffled_h0     = compute_entropy_miller_madow(compute_ngram_counts(shuffled_stream, 1))
    shuffled_h2     = compute_conditional_entropy(shuffled_stream, order=2)
    shuffled_struct = compute_structure_score(shuffled_stream)
    print(f"    Shuffled H0:     {shuffled_h0:.4f}  (~= H0 expected)")
    print(f"    Shuffled H2:     {shuffled_h2:.4f}  (~= H0 expected)")
    print(f"    Shuffled struct: {shuffled_struct:.4f}  (~= 0 expected)")

    # MI decay — key lags: 1h (autocorr), 24h (daily period), 168h (weekly period)
    print("    Computing MI decay profile ...")
    lags = [1, 2, 6, 12, 24, 48, 72, 168, 336]
    mi_profile = mi_decay_profile(stream, lags=lags)
    for lag, mi in mi_profile.items():
        note = ""
        if int(lag) == 24:
            note = "  <- daily period"
        elif int(lag) == 168:
            note = "  <- weekly period"
        elif int(lag) == 336:
            note = "  <- 2-week"
        print(f"      MI(lag={lag:3d}h): {mi:.4f} bits{note}")

    # Bootstrap CIs
    print("    Bootstrap CIs (200 samples, 30% subsamples) ...")
    boot = compute_structure_scores_bootstrap(
        stream, n_bootstrap=200, subsample_frac=0.30, seed=42
    )
    print(f"    Structure score: {boot['mean']:.4f} [{boot['ci_low']:.4f}, {boot['ci_high']:.4f}]")
    print(f"    Sequential:      {boot['seq_mean']:.4f} [{boot['seq_ci_low']:.4f}, {boot['seq_ci_high']:.4f}]")

    # ---- Per-weekday breakdown ----
    print("[5] Per-weekday entropy breakdown ...")
    hourly["bin"] = list(stream)
    hourly["datetime"] = pd.to_datetime(hourly["datetime"])
    weekday_results = {}
    for wd in range(7):
        sub_df = hourly[hourly["datetime"].dt.dayofweek == wd]
        sub_stream = "".join(sub_df["bin"].tolist())
        if len(sub_stream) < 48:
            continue
        sub_h0 = compute_entropy_miller_madow(compute_ngram_counts(sub_stream, 1))
        sub_h2 = compute_conditional_entropy(sub_stream, order=2)
        sub_struct = compute_structure_score(sub_stream)
        day_name = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][wd]
        print(f"    {day_name}: n={len(sub_stream):,}h, H0={sub_h0:.4f}, H2={sub_h2:.4f}, struct={sub_struct:.4f}")
        weekday_results[day_name] = {
            "n_hours": len(sub_stream),
            "H0_bits": round(sub_h0, 6),
            "H_cond_bigram_bits": round(sub_h2, 6),
            "structure_score": round(sub_struct, 6),
        }

    # ---- Hourly profile ----
    print("[6] Mean traffic count by hour-of-day:")
    hourly["hour"] = pd.to_datetime(hourly["datetime"]).dt.hour
    hourly_profile = hourly.groupby("hour")["count"].mean()
    for h, v in hourly_profile.items():
        bar = "#" * int(v / hourly_profile.max() * 30)
        print(f"    {h:02d}:00  {v:6.0f}  {bar}")

    elapsed = time.time() - t0
    h_ratio_2 = h2 / h0 if h0 > 0 else 1.0
    h_ratio_3 = h3 / h0 if h0 > 0 else 1.0

    # ---- Save results ----
    results = {
        "dataset": "UK National Highways WebTRIS",
        "site_id": SITE_ID,
        "site_description": "MIDAS site at A1M/2259B, Southbound",
        "source": "https://webtris.highwaysengland.co.uk/api/v1",
        "period": "January 2023 - December 2023",
        "sampling_resolution": "hourly (aggregated from 15-min intervals)",
        "channel": "Total vehicle count per hour",
        "n_hours": n_chars,
        "date_range": {
            "start": str(hourly["datetime"].min()),
            "end": str(hourly["datetime"].max()),
        },
        "count_stats": {
            "min": float(count_stats["min"]),
            "max": float(count_stats["max"]),
            "mean": round(float(count_stats["mean"]), 1),
            "std": round(float(count_stats["std"]), 1),
        },
        "discretisation": {
            "method": "quantile (equal-occupancy)",
            "n_bins": 20,
            "alphabet": "ABCDEFGHIJKLMNOPQRST",
            "actual_bins_used": n_alpha,
        },
        "entropy": {
            "H0_unigram_bits": round(h0, 6),
            "H_max_bits": round(h_max, 6),
            "H_cond_bigram_bits": round(h2, 6),
            "H_cond_trigram_bits": round(h3, 6),
            "H_cond_4gram_bits": round(h4, 6),
            "structure_score": round(struct_score, 6),
            "sequential_score": round(seq_score, 6),
            "compression_proxy_H2_over_H0": round(h_ratio_2, 6),
            "compression_proxy_H3_over_H0": round(h_ratio_3, 6),
        },
        "shuffled_control": {
            "H0_bits": round(shuffled_h0, 6),
            "H_cond_bigram_bits": round(shuffled_h2, 6),
            "structure_score": round(shuffled_struct, 6),
        },
        "bootstrap": {
            "n_samples": 200,
            "subsample_frac": 0.30,
            "structure_score_mean": round(boot["mean"], 6),
            "structure_score_ci_low": round(boot["ci_low"], 6),
            "structure_score_ci_high": round(boot["ci_high"], 6),
            "sequential_score_mean": round(boot["seq_mean"], 6),
            "sequential_score_ci_low": round(boot["seq_ci_low"], 6),
            "sequential_score_ci_high": round(boot["seq_ci_high"], 6),
        },
        "mi_decay": {str(k): round(v, 6) for k, v in mi_profile.items()},
        "per_weekday": weekday_results,
        "hourly_profile_mean_counts": {
            str(int(h)): round(float(v), 1) for h, v in hourly_profile.items()
        },
        "elapsed_seconds": round(elapsed, 1),
    }

    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n[7] Results saved to {OUT_PATH}")

    print("\n" + "=" * 60)
    print("TRAFFIC FLOW ENTROPY SUMMARY -- A1M Southbound 2023")
    print("=" * 60)
    print(f"  Site:           {results['site_description']}")
    print(f"  Hours:            {n_chars:,}")
    print(f"  H0 (unigram):     {h0:.4f} bits  (max {h_max:.4f})")
    print(f"  H(X|bigram):      {h2:.4f} bits")
    print(f"  H(X|trigram):     {h3:.4f} bits")
    print(f"  Structure score:  {struct_score:.4f}  (1 = maximal)")
    print(f"  Sequential score: {seq_score:.4f}")
    print(f"  Bootstrap struct: {boot['mean']:.4f} [{boot['ci_low']:.4f}, {boot['ci_high']:.4f}]")
    print(f"  Shuffled struct:  {shuffled_struct:.4f}  (baseline)")
    print()
    print("  MI DECAY:")
    for lag, mi in mi_profile.items():
        note = ""
        if int(lag) == 24:
            note = "  <- daily period"
        elif int(lag) == 168:
            note = "  <- weekly period"
        print(f"    lag={lag:3d}h: {mi:.4f} bits{note}")
    print()
    if struct_score > 0.5:
        verdict = "HIGH"
    elif struct_score > 0.35:
        verdict = "MODERATE-HIGH"
    elif struct_score > 0.2:
        verdict = "MODERATE"
    else:
        verdict = "LOW"
    print(f"  VERDICT: {verdict} structure ({struct_score:.3f})")
    print(f"  Comparison: tidal=0.658, weather=0.201, network=0.492")
    print("=" * 60)
    print(f"  Elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
