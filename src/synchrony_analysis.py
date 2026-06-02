"""
synchrony_analysis.py - computes facial synchrony scores for all real and surrogate dyad pairs.

Signals:
  - Expressivity composite (sum of AU_r intensities)
  - Emotion composites (happiness, sadness, fear, anger, surprise, disgust)

Metrics per signal:
  - Zero-lag Pearson r
  - Peak cross-correlation r and lag (in seconds)
  - Windowed synchrony: mean and SD of rolling Pearson r

Outputs:
  - synchrony_raw.csv       : one row per pair (real + surrogate)
  - synchrony_aggregated.csv: real score vs mean surrogate per dyad x phase
  - synchrony_errors.csv     : pairs that failed to load or compute (if any)
"""
import os
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from config import RESAMPLED_DIR, ANALYSIS_DIR
from preproccesing import resample_to_30hz
 
SURROGATE_INDEX = os.path.join(ANALYSIS_DIR, "surrogate_index.csv")
 
# Sampling rate
SR = 30  # Hz

# Windowed synchrony settings
WINDOW_SEC  = 10
STEP_SEC    = 5 
WINDOW_FRAMES = WINDOW_SEC * SR
STEP_FRAMES   = STEP_SEC   * SR

# Cross-correlation settings
MAX_LAG_SEC = 5
MAX_LAG_FRAMES = MAX_LAG_SEC * SR

# Emotion composite definitions (AU_r columns)
# Based on Ekman & Friesen (1978) FACS prototypes, with minor deviations noted in thesis.
EMOTION_AUS = {
    "happiness": ["AU06_r", "AU12_r"],
    "sadness":   ["AU01_r", "AU04_r", "AU15_r"],
    "fear":      ["AU01_r", "AU02_r", "AU04_r", "AU05_r", "AU07_r", "AU20_r", "AU26_r"],
    "anger":     ["AU04_r", "AU05_r", "AU07_r", "AU23_r", "AU24_r"],
    "surprise":  ["AU01_r", "AU02_r", "AU05_r", "AU26_r"],
    "disgust":   ["AU09_r", "AU15_r", "AU16_r", "AU25_r", "AU26_r"],
}

def get_au_columns(df):
    """Return all AU_r intensity columns present in df."""
    return [c for c in df.columns if c.endswith("_r") and c.startswith("AU")]

def clean(arr):
    """Replace infs with NaN, then fill NaN with 0 (neutral face, no AU activation)."""
    arr = np.where(np.isinf(arr), np.nan, arr)
    arr = np.where(np.isnan(arr), 0.0, arr)
    return arr

def build_signals(df):
    """
    Build a dict of {signal_name: np.array} for all facial signals.
 
    Constructs an expressivity composite (sum of all AU_r intensities) and
    six emotion-specific composites (mean of relevant AUs per emotion).
    Only returns signals where the required columns are present in df.
 
    Args:
        df (pd.DataFrame): Cleaned OpenFace DataFrame with AU_r columns.
 
    Returns:
        dict: Mapping of signal name to cleaned numpy array.
    """
    signals = {}
    au_cols = get_au_columns(df)

    # Expressivity composite
    if au_cols:
        signals["expressivity"] = clean(df[au_cols].sum(axis=1).values.astype(float))

    # Emotion composites
    for emotion, aus in EMOTION_AUS.items():
        available = [a for a in aus if a in df.columns]
        if available:
            signals[f"emotion_{emotion}"] = clean(df[available].mean(axis=1).values.astype(float))

    return signals

def zero_lag_r(x, y):
    """Pearson r at zero lag. Returns NaN if variance is zero."""
    if np.std(x) < 1e-10 or np.std(y) < 1e-10:
        return np.nan
    r, _ = pearsonr(x, y)
    return r

def peak_crosscorr(x, y, max_lag=MAX_LAG_FRAMES, sr=SR):
    """
    Normalized cross-correlation over ±max_lag frames.
 
    Args:
        x, y (np.array): Signal arrays of equal length.
        max_lag (int): Maximum lag in frames to search over.
        sr (int): Sampling rate in Hz, used to convert lag to seconds.
 
    Returns:
        tuple[float, float]: (peak_r, peak_lag_seconds).
            peak_r is the maximum absolute correlation;
            peak_lag is positive if x leads y, negative if y leads x.
    """
    x = (x - np.mean(x)) / (np.std(x) + 1e-10)
    y = (y - np.mean(y)) / (np.std(y) + 1e-10)

    n = len(x)
    lags = np.arange(-max_lag, max_lag + 1)
    corrs = []

    for lag in lags:
        if lag < 0:
            xi, yi = x[:n + lag], y[-lag:]
        elif lag > 0:
            xi, yi = x[lag:], y[:n - lag]
        else:
            xi, yi = x, y
        if len(xi) < 2:
            corrs.append(np.nan)
        else:
            corrs.append(np.dot(xi, yi) / len(xi))

    corrs = np.array(corrs)
    peak_idx = np.nanargmax(np.abs(corrs))
    peak_r   = corrs[peak_idx]
    peak_lag = lags[peak_idx] / sr
    return peak_r, peak_lag

def windowed_synchrony(x, y, window=WINDOW_FRAMES, step=STEP_FRAMES):
    """
    Rolling Pearson r computed in successive stepped windows.
 
    Args:
        x, y (np.array): Signal arrays of equal length.
        window (int): Window length in frames.
        step (int): Step size in frames between windows.
 
    Returns:
        tuple[float, float]: (mean_r, sd_r) across all windows,
            or (NaN, NaN) if no valid windows exist.
    """
    n = min(len(x), len(y))
    rs = []
    start = 0
    while start + window <= n:
        xi = x[start:start + window]
        yi = y[start:start + window]
        r = zero_lag_r(xi, yi)
        if not np.isnan(r):
            rs.append(r)
        start += step

    if len(rs) == 0:
        return np.nan, np.nan
    return np.mean(rs), np.std(rs)

def align_series(x, y):
    """Trim both arrays to the same length (shorter one wins)."""
    n = min(len(x), len(y))
    return x[:n], y[:n]

def compute_pair_synchrony(nav_df, pil_df):
    """
    Compute all synchrony metrics for one navigator-pilot pair.
 
    Args:
        nav_df (pd.DataFrame): Cleaned OpenFace DataFrame for the navigator.
        pil_df (pd.DataFrame): Cleaned OpenFace DataFrame for the pilot.
 
    Returns:
        dict: Flat dict of metric values keyed as '{signal}__{metric}'.
            Returns NaN for all metrics if the block is too short.
    """
    nav_signals = build_signals(nav_df)
    pil_signals = build_signals(pil_df)

    common = set(nav_signals.keys()) & set(pil_signals.keys())
    row = {}

    for sig in sorted(common):
        x, y = align_series(nav_signals[sig], pil_signals[sig])

        if len(x) < WINDOW_FRAMES:
            row[f"{sig}__zero_r"]       = np.nan
            row[f"{sig}__peak_r"]       = np.nan
            row[f"{sig}__peak_lag_s"]   = np.nan
            row[f"{sig}__win_mean_r"]   = np.nan
            row[f"{sig}__win_sd_r"]     = np.nan
            continue

        row[f"{sig}__zero_r"]     = zero_lag_r(x, y)
        peak_r, peak_lag          = peak_crosscorr(x, y)
        row[f"{sig}__peak_r"]     = peak_r
        row[f"{sig}__peak_lag_s"] = peak_lag
        win_mean, win_sd          = windowed_synchrony(x, y)
        row[f"{sig}__win_mean_r"] = win_mean
        row[f"{sig}__win_sd_r"]   = win_sd

    return row

def load_resampled(path):
    """
    Load a participant file from RESAMPLED_DIR.
 
    Handles both already-cleaned files (with 'timestamp' column) and files
    that still need resampling (with 'time' column).
 
    Args:
        path (str): Full path to the participant CSV file.
 
    Returns:
        pd.DataFrame: DataFrame at 30 Hz with 'timestamp' as the time column.
 
    Raises:
        KeyError: If neither 'time' nor 'timestamp' is found in the file.
    """
    df = pd.read_csv(path, index_col=0)
    df.columns = df.columns.str.strip()

    if 'timestamp' in df.columns:
        return df

    if 'time' not in df.columns:
        raise KeyError(
            f"Neither 'time' nor 'timestamp' found in {path}. "
            f"Columns: {df.columns.tolist()}"
        )

    df = resample_to_30hz(df)
    df = df.rename(columns={'time': 'timestamp'})
    return df

def load_pair(nav_dyad, pil_dyad, task, phase, iteration):
    """
    Load the navigator from nav_dyad and the pilot from pil_dyad.
 
    For surrogate pairs, nav and pil come from different dyads.
 
    Args:
        nav_dyad (str): Dyad identifier for the navigator in 'navID_pilID' format.
        pil_dyad (str): Dyad identifier for the pilot in 'navID_pilID' format.
        task (str): Task type (e.g. 'reschu', 'instructional', 'discussion').
        phase (str): Phase label.
        iteration (str or int): Block iteration number.
 
    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: (navigator_df, pilot_df)
    """
    nav_id = nav_dyad.split("_")[0]
    pil_id = pil_dyad.split("_")[1]

    nav_file = f"pp{nav_id}_navigator_AU_{task}_{phase}_{iteration}.csv"
    pil_file = f"pp{pil_id}_pilot_AU_{task}_{phase}_{iteration}.csv"

    nav_path = os.path.join(RESAMPLED_DIR, nav_dyad, "raw", nav_id, nav_file)
    pil_path = os.path.join(RESAMPLED_DIR, pil_dyad, "raw", pil_id, pil_file)

    nav_df = load_resampled(nav_path)
    pil_df = load_resampled(pil_path)
    return nav_df, pil_df

def run_synchrony_pipeline(surrogate_index_path, out_dir):
    """
    Compute synchrony for all pairs in the surrogate index and save results.
 
    For each row in the surrogate index, loads the navigator and pilot files,
    computes all synchrony metrics, and collects results into two output files:
    synchrony_raw.csv (one row per pair) and synchrony_aggregated.csv (real
    scores with surrogate means and difference scores merged in).
 
    Args:
        surrogate_index_path (str): Path to surrogate_index.csv.
        out_dir (str): Directory to save output CSVs.
 
    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: (raw_df, agg_df)
    """
    index = pd.read_csv(surrogate_index_path)
    total = len(index)
    print(f"Computing synchrony for {total} pairs ({index['is_paired'].sum()} real, "
          f"{(~index['is_paired']).sum()} surrogate)...\n")

    raw_rows = []
    errors   = []

    for i, row in index.iterrows():
        nav_dyad  = row["nav_dyad"]
        pil_dyad  = row["pil_dyad"]
        task      = row["task"]
        phase     = row["phase"]
        iteration = row["iteration"]
        label     = f"{nav_dyad} → {pil_dyad} | {task} {phase} {iteration}"

        try:
            nav_df, pil_df = load_pair(nav_dyad, pil_dyad, task, phase, iteration)
            metrics = compute_pair_synchrony(nav_df, pil_df)
        except FileNotFoundError as e:
            errors.append({"pair": label, "error": str(e)})
            print(f"  [SKIP] {label} — file not found")
            continue
        except Exception as e:
            errors.append({"pair": label, "error": str(e)})
            print(f"  [ERROR] {label} — {e}")
            continue

        raw_rows.append({
            "nav_dyad":  nav_dyad,
            "pil_dyad":  pil_dyad,
            "nav_id":    row["nav_id"],
            "pil_id":    row["pil_id"],
            "task":      task,
            "phase":     phase,
            "iteration": iteration,
            "zoom":      row["zoom"],
            "is_paired": row["is_paired"],
            **metrics,
        })

        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{total} pairs processed...")

    # Save raw output
    raw_df = pd.DataFrame(raw_rows)
    raw_path = os.path.join(out_dir, "synchrony_raw.csv")
    raw_df.to_csv(raw_path, index=False)
    print(f"\nSaved synchrony_raw.csv → {raw_path}")
    print(f"  Shape: {raw_df.shape[0]} rows × {raw_df.shape[1]} columns")

    # Aggregated: real scores + surrogate means + difference scores
    metric_cols = [c for c in raw_df.columns if "__" in c]
    group_keys  = ["nav_dyad", "task", "phase", "iteration", "zoom"]

    real_df      = raw_df[raw_df["is_paired"]]
    surrogate_df = raw_df[~raw_df["is_paired"]]

    surr_mean = surrogate_df.groupby(group_keys)[metric_cols].mean().reset_index()
    surr_mean.columns = group_keys + [f"surr_mean__{c}" for c in metric_cols]

    agg_df = real_df[group_keys + ["nav_id", "pil_id"] + metric_cols].merge(
        surr_mean, on=group_keys, how="left"
    )

    for col in metric_cols:
        agg_df[f"diff__{col}"] = agg_df[col] - agg_df[f"surr_mean__{col}"]

    agg_path = os.path.join(out_dir, "synchrony_aggregated.csv")
    agg_df.to_csv(agg_path, index=False)
    print(f"Saved synchrony_aggregated.csv → {agg_path}")
    print(f"  Shape: {agg_df.shape[0]} rows × {agg_df.shape[1]} columns")

    # Error log
    if errors:
        err_df = pd.DataFrame(errors)
        err_path = os.path.join(out_dir, "synchrony_errors.csv")
        err_df.to_csv(err_path, index=False)
        print(f"\n  {len(errors)} pairs failed — see synchrony_errors.csv")

    print("\nDone.")
    return raw_df, agg_df

if __name__ == "__main__":
    os.makedirs(ANALYSIS_DIR, exist_ok=True)
    raw_df, agg_df = run_synchrony_pipeline(SURROGATE_INDEX, ANALYSIS_DIR)

    # Sanity check: mean zero-lag r per phase for real pairs
    print("\n── Sanity check: mean zero-lag Pearson r (expressivity) per phase ──")
    real = agg_df.copy()
    if "expressivity__zero_r" in real.columns:
        print(real.groupby("phase")["expressivity__zero_r"].describe().round(3))