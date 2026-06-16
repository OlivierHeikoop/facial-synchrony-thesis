"""
statistical_analysis.py — All four statistical tests for the thesis.

Tests:
    1. Phase comparison — does synchrony differ across instructional / discussion / RESCHU?
    2. Real vs surrogate — does real dyad synchrony exceed the chance baseline?
    3. Zoom effect — does visual access (zoom on/off) moderate synchrony?
    4. Synchrony–performance — does synchrony predict RESCHU task performance?

Signals analysed: expressivity + all six emotion composites.
Metric focus: zero-lag Pearson r (primary), peak cross-correlation r (secondary).

Outputs (all saved to ANALYSIS_DIR):
    - stats_phase_comparison.csv
    - stats_real_vs_surrogate.csv
    - stats_zoom_effect.csv
    - stats_performance_correlation.csv
    - stats_summary.txt

Run after synchrony_analysis.py.
"""

import os
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import wilcoxon, mannwhitneyu, spearmanr, shapiro
from statsmodels.stats.multitest import multipletests
import pingouin as pg
from config import ANALYSIS_DIR, PERFORMANCE_FILE, PERFORMANCE_COLS

warnings.filterwarnings("ignore", category=FutureWarning)

AGG_FILE = os.path.join(ANALYSIS_DIR, "synchrony_aggregated.csv")

FOCUS_SIGNALS = [
    "expressivity",
    "emotion_happiness",
    "emotion_sadness",
    "emotion_fear",
    "emotion_anger",
    "emotion_surprise",
    "emotion_disgust",
]
PRIMARY_METRICS = ["zero_r", "peak_r"]

SIGNAL_LABELS = {
    "expressivity":      "Expressivity",
    "emotion_happiness": "Happiness",
    "emotion_sadness":   "Sadness",
    "emotion_fear":      "Fear",
    "emotion_anger":     "Anger",
    "emotion_surprise":  "Surprise",
    "emotion_disgust":   "Disgust",
}

PHASE_MAP = {
    "video": "Instructional",
    "phase": "Discussion",
    "run":   "Collaborative",
}

def metric_cols(signals=FOCUS_SIGNALS, metrics=PRIMARY_METRICS):
    """Return all signal__metric column names for the configured signals and metrics."""
    return [f"{s}__{m}" for s in signals for m in metrics]


def normality_ok(x, alpha=0.05):
    """Shapiro-Wilk normality test. Returns True if normality is not rejected."""
    x = x.dropna()
    if len(x) < 3:
        return False
    _, p = shapiro(x)
    return p > alpha


def cohens_d(x, y=None):
    """Cohen's d effect size. One-sample vs 0 if y is None, two-sample otherwise."""
    x = np.array(x.dropna()) if hasattr(x, 'dropna') else np.array(x)
    if y is None:
        return np.mean(x) / (np.std(x, ddof=1) + 1e-10)
    y = np.array(y.dropna()) if hasattr(y, 'dropna') else np.array(y)
    pooled_sd = np.sqrt((np.std(x, ddof=1)**2 + np.std(y, ddof=1)**2) / 2)
    return (np.mean(x) - np.mean(y)) / (pooled_sd + 1e-10)


def rank_biserial(w, n):
    """Rank-biserial r as effect size for the Wilcoxon signed-rank test."""
    return 1 - (4 * w) / (n * (n + 1))


def fdr_correct(pvals, alpha=0.05):
    """Benjamini-Hochberg FDR correction. Returns (corrected p-values, reject mask)."""
    reject, pvals_corrected, _, _ = multipletests(pvals, alpha=alpha, method='fdr_bh')
    return pvals_corrected, reject


def load_performance_tsv(path):
    """
    Load and prepare the RESCHU performance TSV for merging with synchrony data.

    Keeps navigator rows only (team scores are identical for both roles),
    constructs dyadNumber from participant IDs, and returns one row per
    dyad per run with the three performance metrics.

    Args:
        path (str or Path): Path to task-reschu_performance.tsv.

    Returns:
        pd.DataFrame or None: Cleaned performance DataFrame with columns
            dyadNumber, iteration, team_total_score, team_payload_correct,
            team_damage. Returns None if the file is not found.
    """
    if not os.path.exists(path):
        print(f"  Performance file not found: {path}")
        return None

    perf_raw = pd.read_csv(path, sep='\t')
    print(f"  Loaded: {perf_raw.shape[0]} rows, "
          f"{perf_raw['participant_id'].nunique()} participants, "
          f"{perf_raw['run'].nunique()} runs")

    perf_nav = perf_raw[perf_raw['role'] == 'navigator'].copy()
    perf_nav['nav_num'] = perf_nav['participant_id'].str.replace('sub-', '').astype(int)
    perf_nav['pil_num'] = perf_nav['nav_num'] + 1
    perf_nav['dyadNumber'] = (
        perf_nav['nav_num'].apply(lambda x: f"{x:02d}") + "_" +
        perf_nav['pil_num'].apply(lambda x: f"{x:02d}")
    )
    perf_nav['iteration'] = perf_nav['run'].astype(str)

    perf_df = perf_nav[['dyadNumber', 'iteration',
                         'team_total_score', 'team_payload_correct',
                         'team_damage']].copy()

    print(f"  Performance scores: {len(perf_df)} rows "
          f"({perf_df['dyadNumber'].nunique()} dyads × "
          f"{perf_df['iteration'].nunique()} runs)")
    return perf_df


# Test 1: Phase comparison

def test_phase_comparison(agg):
    """
    Test whether facial synchrony differs across the three task phases.

    Uses repeated-measures ANOVA if data are normally distributed, otherwise Friedman's test. Dyad-level means are
    computed first, collapsing across runs within a phase.

    Args:
        agg (pd.DataFrame): synchrony_aggregated.csv loaded as a DataFrame.

    Returns:
        pd.DataFrame: One row per signal-metric combination.
    """
    print("\nTest 1: Phase Comparison")
    rows = []

    dyad_phase = (
        agg.groupby(["nav_dyad", "phase"])[metric_cols()]
        .mean()
        .reset_index()
    )
    dyad_phase["phase_label"] = dyad_phase["phase"].map(PHASE_MAP)
    phases = list(PHASE_MAP.keys())

    for col in metric_cols():
        wide = dyad_phase.pivot_table(index="nav_dyad", columns="phase", values=col)
        wide = wide[[p for p in phases if p in wide.columns]].dropna()

        if wide.shape[0] < 5 or wide.shape[1] < 2:
            continue

        groups       = [wide[p].values for p in wide.columns]
        phase_labels = [PHASE_MAP.get(p, p) for p in wide.columns]
        normal       = all(normality_ok(wide[p]) for p in wide.columns)

        if normal and wide.shape[1] == 3:
            long = wide.reset_index().melt(id_vars="nav_dyad", var_name="phase", value_name="score")
            try:
                rm = pg.rm_anova(data=long, dv="score", within="phase",
                                 subject="nav_dyad", correction=True)
                F    = rm["F"].iloc[0]
                p    = np.nan
                for _p_col in ["p_GG_corr", "p_unc", "p-GG-corr", "p-unc", "p_hf_corr", "p-hf-corr"]:
                    if _p_col in rm.columns and pd.notna(rm[_p_col].iloc[0]):
                        p = float(rm[_p_col].iloc[0])
                        break
                eta2 = np.nan
                for _eta_col in ["ng2", "np2", "eta-sq", "eta2", "eps"]:
                    if _eta_col in rm.columns:
                        eta2 = float(rm[_eta_col].iloc[0])
                        break
                test_name = "rm-ANOVA"
                ph = pg.pairwise_tests(data=long, dv="score", within="phase",
                                       subject="nav_dyad", padjust="fdr_bh")
                posthoc_str = "; ".join(
                    f"{PHASE_MAP.get(r.get('A', ''), r.get('A', ''))} vs "
                    f"{PHASE_MAP.get(r.get('B', ''), r.get('B', ''))}: "
                    f"t={r.get('T', r.get('t', float('nan'))):.2f}, "
                    f"p_corr={r.get('p_corr', r.get('p-corr', r.get('p_adjust', r.get('p-adjust', float('nan'))))):.3f}"
                    for _, r in ph.iterrows()
                )
            except Exception as e:
                F, p, eta2, test_name, posthoc_str = np.nan, np.nan, np.nan, f"rm-ANOVA failed: {e}", ""
        else:
            stat, p   = stats.friedmanchisquare(*groups)
            F         = stat
            eta2      = stat / (len(wide) * (len(groups) - 1))
            test_name = "Friedman"
            posthoc_str = ""

        rows.append({
            "signal_metric": col,
            "test":          test_name,
            "statistic":     round(F, 3)    if not np.isnan(F)    else np.nan,
            "p_value":       round(p, 4)    if not np.isnan(p)    else np.nan,
            "effect_size":   round(eta2, 3) if not np.isnan(eta2) else np.nan,
            "effect_label":  "η²p" if "ANOVA" in test_name else "W",
            "n_dyads":       len(wide),
            "phases":        " / ".join(phase_labels),
            "posthoc":       posthoc_str,
            **{f"mean_{PHASE_MAP.get(p, p)}": round(wide[p].mean(), 4) for p in wide.columns},
        })

        sig = "✓" if (not np.isnan(p) and p < 0.05) else ""
        print(f"  {col:40s}  {test_name}  F/χ²={F:.2f}  p={p:.3f}  η²={eta2:.3f} {sig}")

    return pd.DataFrame(rows)


# Test 2: Real vs surrogate

def test_real_vs_surrogate(agg):
    """
    Test whether real dyad synchrony exceeds the surrogate chance baseline.

    One-sample Wilcoxon signed-rank test on difference scores (real minus mean
    surrogate), tested non-directionally (alternative='two-sided'). FDR correction
    applied across signals within each phase separately.

    Args:
        agg (pd.DataFrame): synchrony_aggregated.csv loaded as a DataFrame.

    Returns:
        pd.DataFrame: One row per phase × signal-metric combination.
    """
    print("\nTest 2: Real vs Surrogate")
    rows = []
    diff_cols = [f"diff__{c}" for c in metric_cols() if f"diff__{c}" in agg.columns]

    for phase, label in PHASE_MAP.items():
        subset = agg[agg["phase"] == phase]
        if len(subset) < 5:
            continue

        pvals, col_order = [], []

        for col in diff_cols:
            vals = subset[col].dropna()
            if len(vals) < 5:
                continue
            try:
                w, p = wilcoxon(vals, alternative="two-sided")
            except Exception:
                w, p = np.nan, np.nan
            pvals.append(p)
            col_order.append((col, vals, w))

        if not pvals:
            continue
        pvals_corr, reject = fdr_correct(pvals)

        for (col, vals, w), p_raw, p_corr, rej in zip(col_order, pvals, pvals_corr, reject):
            d  = cohens_d(vals)
            rb = rank_biserial(w, len(vals)) if not np.isnan(w) else np.nan
            rows.append({
                "phase":         label,
                "signal_metric": col.replace("diff__", ""),
                "n":             len(vals),
                "mean_diff":     round(vals.mean(), 4),
                "median_diff":   round(vals.median(), 4),
                "W":             round(w, 2) if not np.isnan(w) else np.nan,
                "p_raw":         round(p_raw, 4),
                "p_fdr":         round(p_corr, 4),
                "significant":   rej,
                "cohens_d":      round(d, 3),
                "rank_biserial": round(rb, 3) if not np.isnan(rb) else np.nan,
            })
            sig = "✓" if rej else ""
            print(f"  [{label:15s}] {col.replace('diff__', ''):40s}  "
                  f"median_diff={vals.median():.3f}  p_fdr={p_corr:.3f} {sig}")

    return pd.DataFrame(rows)


# Test 3: Zoom effect

def test_zoom_effect(agg):
    """
    Test whether visual access to the partner's face moderates facial synchrony.

    Two-sided Mann-Whitney U test comparing zoom=on vs zoom=off dyads per
    signal, metric and phase. FDR correction applied within each phase.

    Args:
        agg (pd.DataFrame): synchrony_aggregated.csv loaded as a DataFrame.

    Returns:
        pd.DataFrame: One row per phase × signal-metric combination.
    """
    print("\nTest 3: Zoom Effect")
    rows = []

    for phase, label in PHASE_MAP.items():
        subset   = agg[agg["phase"] == phase]
        zoom_on  = subset[subset["zoom"] == True]
        zoom_off = subset[subset["zoom"] == False]

        if len(zoom_on) < 3 or len(zoom_off) < 3:
            print(f"  [{label}] skipped — too few dyads per condition")
            continue

        pvals, col_order = [], []

        for col in metric_cols():
            if col not in agg.columns:
                continue
            x = zoom_on[col].dropna()
            y = zoom_off[col].dropna()
            if len(x) < 3 or len(y) < 3:
                continue
            try:
                u, p = mannwhitneyu(x, y, alternative="two-sided")
            except Exception:
                u, p = np.nan, np.nan
            pvals.append(p)
            col_order.append((col, x, y, u))

        if not pvals:
            continue
        pvals_corr, reject = fdr_correct(pvals)

        for (col, x, y, u), p_raw, p_corr, rej in zip(col_order, pvals, pvals_corr, reject):
            d  = cohens_d(x, y)
            rb = 1 - (2 * u) / (len(x) * len(y)) if not np.isnan(u) else np.nan
            rows.append({
                "phase":         label,
                "signal_metric": col,
                "n_zoom_on":     len(x),
                "n_zoom_off":    len(y),
                "mean_zoom_on":  round(x.mean(), 4),
                "mean_zoom_off": round(y.mean(), 4),
                "U":             round(u, 2) if not np.isnan(u) else np.nan,
                "p_raw":         round(p_raw, 4),
                "p_fdr":         round(p_corr, 4),
                "significant":   rej,
                "cohens_d":      round(d, 3),
                "rank_biserial": round(rb, 3) if not np.isnan(rb) else np.nan,
            })
            sig = "✓" if rej else ""
            print(f"  [{label:15s}] {col:40s}  "
                  f"Zoom={x.mean():.3f} vs noZoom={y.mean():.3f}  p_fdr={p_corr:.3f} {sig}")

    return pd.DataFrame(rows)


# Test 4: Synchrony–performance correlation

def test_performance_correlation(agg, perf_df, perf_cols):
    """
    Test whether facial synchrony during collaborative runs predicts team performance.

    Spearman rank correlation between synchrony scores and performance metrics,
    restricted to RESCHU runs. Merged on dyadNumber × iteration. FDR correction
    applied jointly across all signal × metric × performance column combinations.

    Args:
        agg (pd.DataFrame): synchrony_aggregated.csv loaded as a DataFrame.
        perf_df (pd.DataFrame): Cleaned performance data from load_performance_tsv().
        perf_cols (list): Performance column names to correlate against.

    Returns:
        pd.DataFrame: One row per sync signal × performance metric combination.
    """
    print("\nTest 4: Synchrony–Performance Correlation")

    if perf_df is None:
        print("  Performance data not available — skipping.")
        return pd.DataFrame()

    reschu = agg[agg["phase"] == "run"].copy()
    reschu["iteration"]   = reschu["iteration"].astype(str)
    reschu["dyadNumber"]  = reschu["nav_dyad"].astype(str)
    perf_df["iteration"]  = perf_df["iteration"].astype(str)

    merged = reschu.merge(
        perf_df[["dyadNumber", "iteration"] + perf_cols],
        on=["dyadNumber", "iteration"], how="inner"
    )

    if len(merged) == 0:
        print("  No rows matched after merge. Check dyadNumber and iteration formats.")
        print("  Synchrony dyads (sample):",  sorted(reschu["dyadNumber"].unique())[:5])
        print("  Performance dyads (sample):", sorted(perf_df["dyadNumber"].unique())[:5])
        print("  Synchrony iterations:",       sorted(reschu["iteration"].unique())[:5])
        print("  Performance iterations:",     sorted(perf_df["iteration"].unique())[:5])
        return pd.DataFrame()

    print(f"  Merged {len(merged)} rows ({merged['dyadNumber'].nunique()} dyads × runs)")

    rows, pvals, col_order = [], [], []

    for sync_col in metric_cols():
        if sync_col not in merged.columns:
            continue
        for perf_col in perf_cols:
            mask = merged[sync_col].notna() & merged[perf_col].notna()
            if mask.sum() < 5:
                continue
            r, p = spearmanr(merged.loc[mask, sync_col], merged.loc[mask, perf_col])
            pvals.append(p)
            col_order.append((sync_col, perf_col, r, mask.sum()))

    if not pvals:
        print("  No valid pairs to correlate.")
        return pd.DataFrame()

    pvals_corr, reject = fdr_correct(pvals)

    for (sync_col, perf_col, r, n), p_raw, p_corr, rej in zip(col_order, pvals, pvals_corr, reject):
        sig_key, metric = sync_col.rsplit("__", 1)
        rows.append({
            "sync_signal": sync_col,
            "signal":      SIGNAL_LABELS.get(sig_key, sig_key),
            "sync_metric": metric,
            "perf_metric": perf_col,
            "n":           n,
            "spearman_r":  round(r, 3),
            "p_raw":       round(p_raw, 4),
            "p_fdr":       round(p_corr, 4),
            "significant": rej,
        })
        sig = "✓" if rej else ""
        print(f"  {sync_col:40s} × {perf_col:15s}  r={r:.3f}  p_fdr={p_corr:.3f} {sig}")

    return pd.DataFrame(rows)

def write_summary(results, out_path):
    """
    Write a summary of all test results to a text file.

    Args:
        results (dict): Mapping of test name to result DataFrame (or None).
        out_path (str): Path to save the summary .txt file.
    """
    lines = ["=" * 70, "STATISTICAL ANALYSIS SUMMARY", "=" * 70]

    for title, df in results.items():
        lines += [f"\n{'─' * 70}", f"  {title}", f"{'─' * 70}"]
        if df is None or len(df) == 0:
            lines.append("  No results.")
            continue

        sig_col = "significant" if "significant" in df.columns else None
        p_col   = "p_fdr"      if "p_fdr"       in df.columns else \
                  "p_value"    if "p_value"      in df.columns else None

        if sig_col and p_col:
            sig_rows = df[df[sig_col] == True]
            lines.append(f"  Total tests:       {len(df)}")
            lines.append(f"  Significant (FDR): {len(sig_rows)}")
            if len(sig_rows):
                lines.append("\n  Significant findings:")
                for _, r in sig_rows.iterrows():
                    desc  = r.get("signal_metric", r.get("sync_signal", ""))
                    phase = r.get("phase", "")
                    p     = r.get(p_col, "")
                    lines.append(f"    [{phase}] {desc}  p_fdr={p:.4f}")
        else:
            lines.append(df.to_string(index=False))

    lines.append("\n" + "=" * 70)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nSummary saved → {out_path}")

def test_surrogate_corrected_performance(agg, perf_df, perf_cols):
    """
    Repeat the synchrony–performance correlation using surrogate-corrected
    difference scores instead of raw synchrony.

    Isolates the component of synchrony attributable to genuine interpersonal
    coupling (real minus mean surrogate) and correlates it with performance.
    Results are compared against the raw correlations to assess robustness.
    FDR correction applied per performance metric.

    Args:
        agg (pd.DataFrame): synchrony_aggregated.csv loaded as a DataFrame.
        perf_df (pd.DataFrame): Cleaned performance data from load_performance_tsv().
        perf_cols (list): Performance column names to correlate against.

    Returns:
        pd.DataFrame: One row per signal × metric × performance combination,
            with both raw and surrogate-corrected r values.
    """
    print("\nTest 4b: Surrogate-Corrected Synchrony–Performance Correlation")

    if perf_df is None:
        print("  Performance data not available — skipping.")
        return pd.DataFrame()

    reschu = agg[agg["phase"] == "run"].copy()
    reschu["iteration"]  = reschu["iteration"].astype(str)
    reschu["dyadNumber"] = reschu["nav_dyad"].astype(str)
    perf_df["iteration"] = perf_df["iteration"].astype(str)

    merged = reschu.merge(
        perf_df[["dyadNumber", "iteration"] + perf_cols],
        on=["dyadNumber", "iteration"], how="inner"
    )

    if len(merged) == 0:
        print("  No rows matched after merge — skipping.")
        return pd.DataFrame()

    diff_cols_available = [c for c in merged.columns if c.startswith("diff__")]
    if not diff_cols_available:
        print("  No diff__ columns found in merged data.")
        print("  Check that synchrony_aggregated.csv was generated correctly.")
        return pd.DataFrame()

    rows = []
    for perf_col in perf_cols:
        if perf_col not in merged.columns:
            continue
        for sig in FOCUS_SIGNALS:
            for metric in PRIMARY_METRICS:
                diff_col = f"diff__{sig}__{metric}"
                raw_col  = f"{sig}__{metric}"
                if diff_col not in merged.columns:
                    continue
                mask = merged[diff_col].notna() & merged[perf_col].notna()
                if mask.sum() < 5:
                    continue
                r_surr, p_surr = spearmanr(merged.loc[mask, diff_col],
                                           merged.loc[mask, perf_col])
                if raw_col in merged.columns:
                    r_raw, p_raw = spearmanr(merged.loc[mask, raw_col],
                                             merged.loc[mask, perf_col])
                else:
                    r_raw, p_raw = np.nan, np.nan
                rows.append({
                    "perf_metric": perf_col,
                    "signal":      SIGNAL_LABELS.get(sig, sig),
                    "sync_metric": metric,
                    "n":           int(mask.sum()),
                    "r_raw":       round(r_raw,  3),
                    "r_corrected": round(r_surr, 3),
                    "p_raw_raw":   round(p_raw,  4),
                    "p_raw_corr":  round(p_surr, 4),
                })

    if not rows:
        print("  No valid pairs to correlate.")
        return pd.DataFrame()

    surr_corr_df = pd.DataFrame(rows)
    surr_dfs = []
    for pc in surr_corr_df["perf_metric"].unique():
        sub = surr_corr_df[surr_corr_df["perf_metric"] == pc].copy()
        _, p_fdr, _, _ = multipletests(sub["p_raw_corr"], method="fdr_bh")
        sub["p_fdr_corrected"] = p_fdr.round(4)
        sub["significant"]     = p_fdr < 0.05
        surr_dfs.append(sub)
    surr_corr_df = pd.concat(surr_dfs, ignore_index=True)

    sig_surr = surr_corr_df[surr_corr_df["significant"]]
    print(f"  Significant after FDR: {len(sig_surr)} / {len(surr_corr_df)}")
    if len(sig_surr):
        print(sig_surr[["perf_metric", "signal", "sync_metric",
                         "r_corrected", "p_fdr_corrected"]]
              .sort_values("p_fdr_corrected").to_string(index=False))

    return surr_corr_df

if __name__ == "__main__":
    print("Loading synchrony_aggregated.csv...")
    agg = pd.read_csv(AGG_FILE)
    print(f"  {len(agg)} rows, {len(agg.columns)} columns")
    print(f"  Phases: {agg['phase'].value_counts().to_dict()}")
    print(f"  Zoom:   {agg['zoom'].value_counts().to_dict()}")
    print(f"  Dyads:  {agg['nav_dyad'].nunique()}")

    os.makedirs(ANALYSIS_DIR, exist_ok=True)

    df_phase = test_phase_comparison(agg)
    df_phase.to_csv(os.path.join(ANALYSIS_DIR, "stats_phase_comparison.csv"), index=False)

    df_surr = test_real_vs_surrogate(agg)
    df_surr.to_csv(os.path.join(ANALYSIS_DIR, "stats_real_vs_surrogate.csv"), index=False)

    df_zoom = test_zoom_effect(agg)
    df_zoom.to_csv(os.path.join(ANALYSIS_DIR, "stats_zoom_effect.csv"), index=False)

    df_perf  = pd.DataFrame()
    perf_df  = load_performance_tsv(PERFORMANCE_FILE)
    if perf_df is not None:
        df_perf = test_performance_correlation(agg, perf_df, PERFORMANCE_COLS)
        df_perf.to_csv(os.path.join(ANALYSIS_DIR, "stats_performance_correlation.csv"), index=False)
    else:
        print("\nTest 4: Synchrony–Performance — skipping (file not found).")

    df_surr_perf = pd.DataFrame()
    if perf_df is not None:
        df_surr_perf = test_surrogate_corrected_performance(agg, perf_df, PERFORMANCE_COLS)
        if len(df_surr_perf):
            df_surr_perf.to_csv(
                os.path.join(ANALYSIS_DIR, "stats_surrogate_corrected_performance.csv"),
                index=False
            )

    write_summary({
        "Test 1: Phase Comparison":      df_phase,
        "Test 2: Real vs Surrogate":     df_surr,
        "Test 3: Zoom Effect":           df_zoom,
        "Test 4: Synchrony–Performance": df_perf if len(df_perf) else None,
    }, os.path.join(ANALYSIS_DIR, "stats_summary.txt"))

    print("\nAll done. Files saved to:", ANALYSIS_DIR)