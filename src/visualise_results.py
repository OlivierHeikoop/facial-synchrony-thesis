"""
visualise_results.py — All figures for the facial synchrony thesis.

Figures produced:
    1.  Expressivity synchrony across phases (bar + scatter)
    2.  Mean zero-lag synchrony heatmap (all signals × phase)
    3.  Per-dyad expressivity synchrony across phases (spaghetti plot)
    4.  Emotion composite synchrony by phase (grouped bar)
    5.  Real vs surrogate violin plots (all signals)
    6.  Above-chance difference scores by phase (bar grid)
    7.  Effect sizes: real vs surrogate (dot plot)
    8.  Synchrony over blocks × zoom condition (line grid, zero-lag and peak r)
    9.  Zoom effect size heatmap (Cohen's d)
    10. Zoom on vs off — discussion phase (dot plot)
    11. Synchrony–performance correlation heatmap (if performance file exists)
    12. Mean windowed synchrony by phase (bar grid)
    13. Within-block synchrony variability by phase (grouped bar)
    14. Windowed synchrony over blocks × zoom (line grid)
    15. Mean vs variability of windowed synchrony scatter (per phase)

Run after statistical_analysis.py. All figures are shown interactively via plt.show().
To save figures instead, replace plt.show() with plt.savefig() in each function.
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy.stats import spearmanr
from config import ANALYSIS_DIR, PERFORMANCE_COLS, PERFORMANCE_FILE, PRIMARY_METRIC

warnings.filterwarnings("ignore")

AGG_FILE   = os.path.join(ANALYSIS_DIR, "synchrony_aggregated.csv")
RAW_FILE   = os.path.join(ANALYSIS_DIR, "synchrony_raw.csv")
SURR_FILE  = os.path.join(ANALYSIS_DIR, "stats_real_vs_surrogate.csv")
PHASE_FILE = os.path.join(ANALYSIS_DIR, "stats_phase_comparison.csv")
ZOOM_FILE  = os.path.join(ANALYSIS_DIR, "stats_zoom_effect.csv")
PERF_FILE  = os.path.join(ANALYSIS_DIR, "stats_performance_correlation.csv")

PHASE_MAP    = {"video": "Instructional", "phase": "Discussion", "run": "Collaborative"}
PHASE_ORDER  = ["Instructional", "Discussion", "Collaborative"]
PHASE_COLORS = {"Instructional": "#5B8DB8", "Discussion": "#E07B54", "Collaborative": "#6BAE75"}

SIGNAL_LABELS = {
    "expressivity":      "Expressivity",
    "emotion_happiness": "Happiness",
    "emotion_sadness":   "Sadness",
    "emotion_fear":      "Fear",
    "emotion_anger":     "Anger",
    "emotion_surprise":  "Surprise",
    "emotion_disgust":   "Disgust",
}
SIGNAL_ORDER = list(SIGNAL_LABELS.keys())
SIGNAL_NAMES = [SIGNAL_LABELS[s] for s in SIGNAL_ORDER]

ZOOM_ON_COLOR  = "#6BAE75"
ZOOM_OFF_COLOR = "#5B8DB8"
REAL_COLOR     = "#6BAE75"
SURR_COLOR     = "#E07B54"

# Experiment block order: Video → Disc → Run 0–3 → Disc → Run 4–7
BLOCK_SEQUENCE = [
    ("video", 0, "Video"),
    ("phase", 0, "Disc."),
    ("run",   0, "Run 0"),
    ("run",   1, "Run 1"),
    ("run",   2, "Run 2"),
    ("run",   3, "Run 3"),
    ("phase", 1, "Disc."),
    ("run",   4, "Run 4"),
    ("run",   5, "Run 5"),
    ("run",   6, "Run 6"),
    ("run",   7, "Run 7"),
]

sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams.update({"figure.dpi": 120, "axes.spines.top": False, "axes.spines.right": False})

def sig_stars(p):
    """Convert a p-value to significance stars, or 'n.s.'."""
    if pd.isna(p): return ""
    if p < 0.001:  return "***"
    if p < 0.01:   return "**"
    if p < 0.05:   return "*"
    return "n.s."


def ci95(series):
    """95% confidence interval half-width (1.96 × SE)."""
    n = series.dropna().count()
    if n < 2: return 0
    return 1.96 * series.std() / np.sqrt(n)


def se(series):
    """Standard error of the mean."""
    n = series.dropna().count()
    if n < 2: return 0
    return series.std() / np.sqrt(n)


def get_blocks(agg):
    """Return BLOCK_SEQUENCE entries that are present in the data."""
    available = set(zip(agg["phase"], agg["iteration"].astype(int)))
    return [(ph, it, lbl) for ph, it, lbl in BLOCK_SEQUENCE if (ph, it) in available]


def phase_shade(ax, blocks):
    """Draw phase background shading behind plot elements."""
    bg = {"video": "#D6E8F5", "phase": "#FADDD0", "run": "#D4EDD9"}
    prev = None
    for xi, (ph, it, lbl) in enumerate(blocks):
        ax.axvspan(xi - 0.5, xi + 0.5, color=bg[ph], alpha=0.5, zorder=0)
        if prev and ph != prev:
            ax.axvline(xi - 0.5, color="gray", lw=0.8, ls=":", zorder=1)
        prev = ph

def fig_expressivity_by_phase(agg, phase_df):
    """Bar chart of expressivity zero-lag synchrony per phase with dyad-level scatter."""
    fig, ax = plt.subplots(figsize=(7, 5))
    dyad_means  = agg.groupby(["nav_dyad", "phase_label"])["expressivity__zero_r"].mean().reset_index()
    phase_means = dyad_means.groupby("phase_label")["expressivity__zero_r"].agg(["mean", "sem"]).reindex(PHASE_ORDER)
    ax.bar(PHASE_ORDER, phase_means["mean"], yerr=phase_means["sem"], capsize=5,
           color=[PHASE_COLORS[p] for p in PHASE_ORDER], edgecolor="white", lw=1.2,
           error_kw={"elinewidth": 1.5})
    for ph in PHASE_ORDER:
        vals = dyad_means[dyad_means["phase_label"] == ph]["expressivity__zero_r"]
        ax.scatter([ph] * len(vals), vals, color="black", alpha=0.3, s=18, zorder=3)
    row = phase_df[phase_df["signal_metric"] == "expressivity__zero_r"]
    if len(row):
        p = row["p_value"].values[0]
        ax.text(0.98, 0.97, f"p = {p:.3f}", transform=ax.transAxes,
                ha="right", va="top", fontsize=10, color="gray")
    ax.set_ylabel("Mean zero-lag Pearson r", fontsize=12)
    ax.set_title("Expressivity Synchrony Across Phases", fontsize=13, fontweight="bold")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.15)
    plt.tight_layout()
    plt.show()

def fig_synchrony_heatmap(agg):
    """Heatmap of mean zero-lag synchrony for all signals across phases."""
    heat_data = []
    for sig in SIGNAL_ORDER:
        col = f"{sig}__zero_r"
        if col not in agg.columns:
            continue
        heat_data.append({PHASE_MAP[pk]: agg[agg["phase"] == pk][col].mean() for pk in PHASE_MAP})
    heat_df = pd.DataFrame(heat_data, index=SIGNAL_NAMES)[PHASE_ORDER]
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(heat_df, annot=True, fmt=".3f", cmap="YlOrRd", linewidths=0.5, ax=ax,
                cbar_kws={"label": "Mean Pearson r"})
    ax.set_title("Mean Zero-Lag Synchrony by Signal and Phase", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.show()

def fig_perdyad_spaghetti(agg):
    """Per-dyad expressivity synchrony across phases with group mean overlay."""
    dyad_means = agg.groupby(["nav_dyad", "phase_label"])["expressivity__zero_r"].mean().reset_index()
    fig, ax    = plt.subplots(figsize=(8, 5))
    wide = dyad_means.pivot(index="nav_dyad", columns="phase_label",
                            values="expressivity__zero_r").reindex(columns=PHASE_ORDER).dropna()
    x_pos = list(range(len(PHASE_ORDER)))
    for _, row in wide.iterrows():
        ax.plot(x_pos, row.values, color="steelblue", alpha=0.25, lw=1)
    means = wide.mean()
    ax.plot(x_pos, means.values, color="navy", lw=2.5, marker="o",
            ms=7, label="Group mean", zorder=5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(PHASE_ORDER)
    ax.set_ylabel("Expressivity zero-lag Pearson r", fontsize=12)
    ax.set_title("Per-Dyad Expressivity Synchrony Across Phases", fontsize=13, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.show()

def fig_emotion_by_phase(agg):
    """Grouped bar chart of emotion composite synchrony by phase."""
    emotion_sigs  = [s for s in SIGNAL_ORDER if s != "expressivity"]
    emotion_names = [SIGNAL_LABELS[s] for s in emotion_sigs]
    x = np.arange(len(emotion_sigs))
    width = 0.25
    fig, ax = plt.subplots(figsize=(11, 5))
    for i, (ph_key, ph_label) in enumerate(PHASE_MAP.items()):
        means_v = [agg[agg["phase"] == ph_key][f"{s}__zero_r"].dropna().mean() for s in emotion_sigs]
        sems_v  = [agg[agg["phase"] == ph_key][f"{s}__zero_r"].dropna().sem()  for s in emotion_sigs]
        ax.bar(x + (i - 1) * width, means_v, width, yerr=sems_v, capsize=3,
               label=ph_label, color=PHASE_COLORS[ph_label], edgecolor="white",
               error_kw={"elinewidth": 1})
    ax.set_xticks(x)
    ax.set_xticklabels(emotion_names, rotation=15, ha="right")
    ax.set_ylabel("Mean zero-lag Pearson r", fontsize=12)
    ax.set_title("Emotion Composite Synchrony by Phase", fontsize=13, fontweight="bold")
    ax.legend(title="Phase")
    plt.tight_layout()
    plt.show()

def fig_real_vs_surrogate_violins(raw, surr_df):
    """Violin plots comparing real and surrogate synchrony distributions for all signals."""
    n_cols = 4
    n_rows = int(np.ceil(len(SIGNAL_ORDER) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, n_rows * 4), sharey=False)
    axes = axes.flatten()
    for ax_idx, sig in enumerate(SIGNAL_ORDER):
        ax  = axes[ax_idx]
        col = f"{sig}__zero_r"
        if col not in raw.columns:
            ax.set_visible(False)
            continue
        real_v = raw[raw["is_paired"] == True][col].dropna()
        surr_v = raw[raw["is_paired"] == False][col].dropna()
        plot_df = pd.DataFrame({
            "r":    pd.concat([real_v, surr_v], ignore_index=True),
            "type": ["Real"] * len(real_v) + ["Surrogate"] * len(surr_v),
        })
        sns.violinplot(data=plot_df, x="type", y="r",
                       palette={"Real": REAL_COLOR, "Surrogate": SURR_COLOR},
                       inner="box", ax=ax, linewidth=1.2, cut=0)
        sig_rows = surr_df[surr_df["signal_metric"] == col]
        if len(sig_rows):
            p_fdr = sig_rows["p_fdr"].min()
            stars = sig_stars(p_fdr)
            yhi  = plot_df["r"].quantile(0.995)
            yr   = yhi - plot_df["r"].quantile(0.005)
            ax.text(0.5, yhi + yr * 0.05, stars if stars != "n.s." else "n.s.",
                    ha="center", va="bottom",
                    fontsize=13 if stars != "n.s." else 9,
                    fontweight="bold",
                    color="black" if stars != "n.s." else "gray")
        ax.set_title(SIGNAL_LABELS[sig], fontsize=11, fontweight="bold")
        ax.set_xlabel("")
        ax.tick_params(labelsize=9)
        if ax_idx % n_cols != 0:
            ax.set_ylabel("")
    for ax_idx in range(len(SIGNAL_ORDER), len(axes)):
        axes[ax_idx].set_visible(False)
    fig.suptitle("Real vs Surrogate Synchrony (Zero-lag Pearson r)\n"
                 "(* p<.05, ** p<.01, *** p<.001 after FDR)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.show()

def fig_difference_scores(agg, surr_df):
    """Bar grid of real-minus-surrogate difference scores per signal and phase."""
    n_cols = 3
    n_rows = int(np.ceil(len(SIGNAL_ORDER) / n_cols))
    phase_colors_bar = {"Instructional": "#5B8DB8", "Discussion": "#E07B54", "Collaborative": "#6BAE75"}
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(13, n_rows * 3.5))
    axes = axes.flatten()
    for ax_idx, sig in enumerate(SIGNAL_ORDER):
        ax      = axes[ax_idx]
        col     = f"{sig}__zero_r"
        diff_col = f"diff__{col}"
        if diff_col not in agg.columns:
            ax.set_visible(False)
            continue
        medians, errs, stars_list, colors = [], [], [], []
        for ph_key, ph_label in PHASE_MAP.items():
            vals = agg[agg["phase"] == ph_key][diff_col].dropna()
            medians.append(vals.median())
            errs.append(ci95(vals))
            row = surr_df[(surr_df["phase"] == ph_label) & (surr_df["signal_metric"] == col)]
            p   = row["p_fdr"].values[0] if len(row) else 1.0
            stars_list.append(sig_stars(p))
            colors.append(phase_colors_bar[ph_label])
        valid = [(m, e) for m, e in zip(medians, errs) if not np.isnan(m)]
        if valid:
            tops  = [m + e for m, e in valid]
            bots  = [m - e for m, e in valid]
            yspan = max(tops) - min(bots) if max(tops) != min(bots) else 0.01
            ax.set_ylim(min(bots) - yspan * 0.2, max(tops) + yspan * 0.35)
        ax.bar(range(3), medians, color=colors, edgecolor="white", lw=1)
        ax.axhline(0, color="black", lw=0.8, ls="--", alpha=0.6)
        for i, (m, s) in enumerate(zip(medians, stars_list)):
            if s and s != "n.s." and not np.isnan(m):
                ypos = m + errs[i] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.03
                ax.text(i, ypos, s, ha="center", fontsize=12,
                        fontweight="bold", color="black", va="bottom")
        ax.set_title(SIGNAL_LABELS[sig], fontsize=11, fontweight="bold")
        ax.set_xticks(range(3))
        ax.set_xticklabels(["Inst", "Disc", "Coll"], fontsize=9)
        if ax_idx % n_cols == 0:
            ax.set_ylabel("Median real − surrogate r", fontsize=9)
    for ax_idx in range(len(SIGNAL_ORDER), len(axes)):
        axes[ax_idx].set_visible(False)
    handles = [mpatches.Patch(color=c, label=p) for p, c in phase_colors_bar.items()]
    fig.legend(handles=handles, loc="lower right", fontsize=10, title="Phase", framealpha=0.9)
    fig.suptitle("Above-Chance Synchrony: Real − Surrogate Difference Scores\n"
                 "(* p<.05, ** p<.01, *** p<.001 after FDR correction)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.show()

def fig_effect_sizes_real_vs_surrogate(surr_df):
    """Dot plot of Cohen's d effect sizes (real vs surrogate) per signal and phase."""
    fig, ax = plt.subplots(figsize=(9, 5))
    y_pos   = np.arange(len(SIGNAL_NAMES))
    offsets = {"Instructional": -0.25, "Discussion": 0, "Collaborative": 0.25}
    markers = {"Instructional": "o", "Discussion": "s", "Collaborative": "^"}
    for ph_key, ph_label in PHASE_MAP.items():
        ds, sig_mask = [], []
        for sig in SIGNAL_ORDER:
            col = f"{sig}__zero_r"
            row = surr_df[(surr_df["phase"] == ph_label) & (surr_df["signal_metric"] == col)]
            ds.append(row["cohens_d"].values[0] if len(row) else np.nan)
            sig_mask.append(row["significant"].values[0] if len(row) else False)
        y          = y_pos + offsets[ph_label]
        colors_pts = [PHASE_COLORS[ph_label] if s else "lightgray" for s in sig_mask]
        ax.scatter(ds, y, c=colors_pts, marker=markers[ph_label],
                   s=80, zorder=3, edgecolors="white", lw=0.5)
    ax.axvline(0, color="black", lw=0.8, ls="--")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(SIGNAL_NAMES)
    ax.set_xlabel("Cohen's d (real vs surrogate)", fontsize=12)
    ax.set_title("Effect Sizes: Real vs Surrogate\n(coloured = significant, grey = n.s.)",
                 fontsize=13, fontweight="bold")
    handles = [mpatches.Patch(color=PHASE_COLORS[p], label=p) for p in PHASE_ORDER]
    ax.legend(handles=handles, title="Phase", loc="lower right")
    plt.tight_layout()
    plt.show()

def fig_zoom_over_blocks(agg):
    """Line plots of synchrony over blocks split by zoom condition (zero-lag and peak r)."""
    blocks = get_blocks(agg)
    n_cols = 3
    n_rows = int(np.ceil(len(SIGNAL_ORDER) / n_cols))
    for metric_suffix, metric_label in [("__zero_r", "Zero-lag Pearson r"),
                                         ("__peak_r", "Peak cross-correlation r")]:
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, n_rows * 3.5), sharey=False)
        axes = axes.flatten()
        for ax_idx, sig in enumerate(SIGNAL_ORDER):
            ax  = axes[ax_idx]
            col = f"{sig}{metric_suffix}"
            if col not in agg.columns:
                ax.set_visible(False)
                continue
            x_labels              = [lbl for _, _, lbl in blocks]
            x                     = np.arange(len(blocks))
            m_on, ci_on, m_off, ci_off = [], [], [], []
            for ph, it, _ in blocks:
                sub = agg[(agg["phase"] == ph) & (agg["iteration"] == it)]
                on  = sub[sub["zoom"] == True][col].dropna()
                off = sub[sub["zoom"] == False][col].dropna()
                m_on.append(on.mean());   ci_on.append(ci95(on))
                m_off.append(off.mean()); ci_off.append(ci95(off))
            m_on  = np.array(m_on);  m_off  = np.array(m_off)
            ci_on = np.array(ci_on); ci_off = np.array(ci_off)
            phase_shade(ax, blocks)
            ax.plot(x, m_on,  color=ZOOM_ON_COLOR,  lw=2, marker="o", ms=5, label="Zoom On",  zorder=3)
            ax.fill_between(x, m_on  - ci_on,  m_on  + ci_on,  alpha=0.2, color=ZOOM_ON_COLOR,  zorder=2)
            ax.plot(x, m_off, color=ZOOM_OFF_COLOR, lw=2, marker="s", ms=5, label="Zoom Off", zorder=3, ls="--")
            ax.fill_between(x, m_off - ci_off, m_off + ci_off, alpha=0.2, color=ZOOM_OFF_COLOR, zorder=2)
            ax.set_title(SIGNAL_LABELS[sig], fontsize=11, fontweight="bold")
            ax.set_xticks(x)
            ax.set_xticklabels(x_labels, rotation=30, ha="right", fontsize=8)
            if ax_idx % n_cols == 0:
                ax.set_ylabel(metric_label, fontsize=9)
        for ax_idx in range(len(SIGNAL_ORDER), len(axes)):
            axes[ax_idx].set_visible(False)
        handles = [
            plt.Line2D([0], [0], color=ZOOM_ON_COLOR,  lw=2, marker="o", label="Zoom On"),
            plt.Line2D([0], [0], color=ZOOM_OFF_COLOR, lw=2, marker="s", ls="--", label="Zoom Off"),
        ]
        fig.legend(handles=handles, loc="lower right", fontsize=10,
                   title="Zoom condition", framealpha=0.9)
        fig.suptitle(f"Synchrony ({metric_label}) × Block × Zoom\nmean ±95% CI",
                     fontsize=12, fontweight="bold", y=1.01)
        plt.tight_layout()
        plt.show()

def fig_zoom_effect_heatmap(zoom_df):
    """Heatmap of Cohen's d for the zoom effect per signal and phase."""
    d_rows = []
    for sig in SIGNAL_ORDER:
        col = f"{sig}__zero_r"
        d_rows.append([
            zoom_df[(zoom_df["phase"] == ph) & (zoom_df["signal_metric"] == col)]["cohens_d"].values[0]
            if len(zoom_df[(zoom_df["phase"] == ph) & (zoom_df["signal_metric"] == col)]) else np.nan
            for ph in PHASE_ORDER
        ])
    d_df = pd.DataFrame(d_rows, index=SIGNAL_NAMES, columns=PHASE_ORDER)
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(d_df, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                linewidths=0.5, ax=ax, cbar_kws={"label": "Cohen's d (Zoom On − Off)"})
    ax.set_title("Zoom Effect Size by Signal and Phase\n(positive = Zoom On > Zoom Off)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.show()

def fig_zoom_discussion_dotplot(agg):
    """Dot plot comparing zoom on vs off synchrony during the discussion phase."""
    subset   = agg[agg["phase"] == "phase"]
    zoom_m   = [subset[subset["zoom"] == True][f"{s}__zero_r"].mean()  for s in SIGNAL_ORDER]
    nozoom_m = [subset[subset["zoom"] == False][f"{s}__zero_r"].mean() for s in SIGNAL_ORDER]
    y        = np.arange(len(SIGNAL_NAMES))
    fig, ax  = plt.subplots(figsize=(8, 5))
    ax.scatter(zoom_m,   y + 0.15, color=ZOOM_ON_COLOR,  s=80, label="Zoom On",  zorder=3)
    ax.scatter(nozoom_m, y - 0.15, color=ZOOM_OFF_COLOR, s=80, label="Zoom Off", zorder=3, marker="s")
    for i in range(len(SIGNAL_NAMES)):
        ax.plot([zoom_m[i], nozoom_m[i]], [y[i] + 0.15, y[i] - 0.15],
                color="gray", lw=0.8, alpha=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(SIGNAL_NAMES)
    ax.set_xlabel("Mean zero-lag Pearson r", fontsize=12)
    ax.set_title("Zoom On vs Off — Discussion Phase (all signals)",
                 fontsize=13, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.show()

def fig_performance_heatmap(perf_df):
    """
    Heatmaps of Spearman correlations between synchrony and performance metrics.

    Produces one heatmap per synchrony metric (zero-lag r and peak r),
    with signal on rows and performance metric on columns. Significant
    cells after FDR correction are marked with *.

    This is Figure 8 in the thesis.

    Args:
        perf_df (pd.DataFrame): stats_performance_correlation.csv loaded
            as a DataFrame. Must contain columns: sync_metric, signal,
            perf_metric, spearman_r, significant.
    """
    if perf_df is None or len(perf_df) == 0:
        print("fig_performance_heatmap: no data to plot.")
        return

    for sync_metric, metric_label in [("zero_r", "Zero-lag Pearson r"),
                                       ("peak_r", "Peak cross-correlation r")]:
        sub = perf_df[perf_df["sync_metric"] == sync_metric]
        if len(sub) == 0:
            continue

        pivot = sub.pivot_table(index="signal", columns="perf_metric",
                                values="spearman_r", aggfunc="mean")
        sig_p = sub.pivot_table(index="signal", columns="perf_metric",
                                values="significant", aggfunc="first")

        ordered = [SIGNAL_LABELS[s] for s in SIGNAL_ORDER
                   if SIGNAL_LABELS[s] in pivot.index]
        pivot = pivot.reindex(ordered)
        sig_p = sig_p.reindex(ordered)

        annot = pd.DataFrame("", index=pivot.index, columns=pivot.columns)
        for r_idx in pivot.index:
            for c_idx in pivot.columns:
                val  = pivot.loc[r_idx, c_idx]
                is_s = sig_p.loc[r_idx, c_idx] if r_idx in sig_p.index else False
                star = "*" if (not pd.isna(is_s) and is_s) else ""
                annot.loc[r_idx, c_idx] = f"{val:.3f}{star}" if not pd.isna(val) else ""

        fig, ax = plt.subplots(figsize=(max(7, len(pivot.columns) * 2.5), 5))
        sns.heatmap(pivot, annot=annot, fmt="", cmap="RdYlGn",
                    center=0, linewidths=0.5, ax=ax,
                    cbar_kws={"label": f"Spearman r ({metric_label})"})
        ax.set_title(f"Synchrony–Performance Correlations ({metric_label})\n"
                     f"(* significant after FDR correction)",
                     fontsize=12, fontweight="bold")
        ax.set_xlabel("Performance metric")
        ax.set_ylabel("")
        plt.tight_layout()
        plt.show()

def fig_windowed_synchrony_by_phase(agg):
    """Bar grid of mean windowed synchrony per signal and phase."""
    if "expressivity__win_mean_r" not in agg.columns:
        print("win_mean_r columns not found in aggregated data — check synchrony_analysis.py output.")
        return
    fig, axes = plt.subplots(1, 3, figsize=(13, 5), sharey=False)
    for ax, (ph_key, ph_label) in zip(axes, PHASE_MAP.items()):
        subset  = agg[agg["phase"] == ph_key]
        means_v = [subset[f"{s}__win_mean_r"].mean() for s in SIGNAL_ORDER]
        sems_v  = [subset[f"{s}__win_mean_r"].sem()  for s in SIGNAL_ORDER]
        x       = np.arange(len(SIGNAL_NAMES))
        ax.bar(x, means_v, yerr=sems_v, capsize=4, color=PHASE_COLORS[ph_label],
               edgecolor="white", error_kw={"elinewidth": 1.2})
        ax.axhline(0, color="black", lw=0.6, ls="--", alpha=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(SIGNAL_NAMES, rotation=40, ha="right", fontsize=9)
        ax.set_title(ph_label, fontsize=12, fontweight="bold")
        if ax == axes[0]:
            ax.set_ylabel("Mean windowed Pearson r (±SE)", fontsize=10)
    fig.suptitle("Mean Windowed Synchrony (10s windows, 5s step) by Phase",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.show()

def fig_windowed_variability(agg):
    """Grouped bar chart of within-block synchrony variability (SD) by phase."""
    if "expressivity__win_sd_r" not in agg.columns:
        print("win_sd_r columns not found.")
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    x     = np.arange(len(SIGNAL_NAMES))
    width = 0.25
    for i, (ph_key, ph_label) in enumerate(PHASE_MAP.items()):
        subset = agg[agg["phase"] == ph_key]
        sds    = [subset[f"{s}__win_sd_r"].mean() for s in SIGNAL_ORDER]
        sds_se = [subset[f"{s}__win_sd_r"].sem()  for s in SIGNAL_ORDER]
        ax.bar(x + (i - 1) * width, sds, width, yerr=sds_se, capsize=3,
               label=ph_label, color=PHASE_COLORS[ph_label], edgecolor="white",
               error_kw={"elinewidth": 1})
    ax.set_xticks(x)
    ax.set_xticklabels(SIGNAL_NAMES, rotation=15, ha="right")
    ax.set_ylabel("Mean SD of windowed r", fontsize=12)
    ax.set_title("Within-Block Synchrony Variability by Phase\n"
                 "(higher = more fluctuation over time)",
                 fontsize=13, fontweight="bold")
    ax.legend(title="Phase")
    plt.tight_layout()
    plt.show()

def fig_windowed_zoom_over_blocks(agg):
    """Line plots of windowed synchrony over blocks split by zoom condition."""
    if "expressivity__win_mean_r" not in agg.columns:
        print("win_mean_r columns not found.")
        return
    blocks = get_blocks(agg)
    n_cols = 3
    n_rows = int(np.ceil(len(SIGNAL_ORDER) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, n_rows * 3.5), sharey=False)
    axes = axes.flatten()
    for ax_idx, sig in enumerate(SIGNAL_ORDER):
        ax  = axes[ax_idx]
        col = f"{sig}__win_mean_r"
        if col not in agg.columns:
            ax.set_visible(False)
            continue
        x        = np.arange(len(blocks))
        x_labels = [lbl for _, _, lbl in blocks]
        m_on, se_on, m_off, se_off = [], [], [], []
        for ph, it, _ in blocks:
            sub = agg[(agg["phase"] == ph) & (agg["iteration"] == it)]
            on  = sub[sub["zoom"] == True][col].dropna()
            off = sub[sub["zoom"] == False][col].dropna()
            m_on.append(on.mean());   se_on.append(se(on))
            m_off.append(off.mean()); se_off.append(se(off))
        m_on  = np.array(m_on);  m_off  = np.array(m_off)
        se_on = np.array(se_on); se_off = np.array(se_off)
        phase_shade(ax, blocks)
        ax.plot(x, m_on,  color=ZOOM_ON_COLOR,  lw=2, marker="o", ms=5, zorder=3)
        ax.fill_between(x, m_on  - se_on,  m_on  + se_on,  alpha=0.25, color=ZOOM_ON_COLOR,  zorder=2)
        ax.plot(x, m_off, color=ZOOM_OFF_COLOR, lw=2, marker="o", ms=5, ls="--", zorder=3)
        ax.fill_between(x, m_off - se_off, m_off + se_off, alpha=0.25, color=ZOOM_OFF_COLOR, zorder=2)
        ax.set_title(SIGNAL_LABELS[sig], fontsize=11, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=30, ha="right", fontsize=8)
        if ax_idx % n_cols == 0:
            ax.set_ylabel("Mean windowed r (±1 SE)", fontsize=9)
    for ax_idx in range(len(SIGNAL_ORDER), len(axes)):
        axes[ax_idx].set_visible(False)
    handles = [
        plt.Line2D([0], [0], color=ZOOM_ON_COLOR,  lw=2, marker="o", label="Zoom On"),
        plt.Line2D([0], [0], color=ZOOM_OFF_COLOR, lw=2, marker="o", ls="--", label="Zoom Off"),
    ]
    fig.legend(handles=handles, loc="lower right", fontsize=10,
               title="Zoom condition", framealpha=0.9)
    fig.suptitle("Windowed Synchrony (10s windows) × Block × Zoom\nmean ±68% CI (1 SE)",
                 fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.show()

def fig_mean_vs_variability_scatter(agg):
    """Scatter of expressivity mean windowed r vs SD per block, per phase."""
    if "expressivity__win_mean_r" not in agg.columns:
        print("win_mean_r columns not found.")
        return
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, (ph_key, ph_label) in zip(axes, PHASE_MAP.items()):
        subset = agg[agg["phase"] == ph_key]
        col_m  = "expressivity__win_mean_r"
        col_s  = "expressivity__win_sd_r"
        x      = subset[col_m].dropna()
        y_s    = subset[col_s].dropna()
        common = x.index.intersection(y_s.index)
        ax.scatter(x[common], y_s[common], color=PHASE_COLORS[ph_label],
                   alpha=0.5, s=30, edgecolors="white", lw=0.3)
        if len(common) > 2:
            m, b = np.polyfit(x[common], y_s[common], 1)
            xl   = np.linspace(x[common].min(), x[common].max(), 100)
            ax.plot(xl, m * xl + b, color="black", lw=1.2, ls="--")
            r, p = spearmanr(x[common], y_s[common])
            ax.text(0.05, 0.95, f"r={r:.2f}, p={p:.3f}", transform=ax.transAxes,
                    fontsize=9, va="top",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))
        ax.set_xlabel("Mean windowed r", fontsize=10)
        ax.set_ylabel("SD of windowed r" if ax == axes[0] else "", fontsize=10)
        ax.set_title(ph_label, fontsize=12, fontweight="bold")
    fig.suptitle("Expressivity Synchrony: Mean vs Variability within Blocks\n"
                 "(higher mean + lower SD = stable coupling)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.show()

def fig_performance_scatter(merged, primary_metric="team_total_score"):
    """
    Scatter plots of expressivity synchrony vs performance metrics.

    Produces two panels per performance metric:
        - Left:  all observations coloured by zoom condition with overall regression line
        - Right: per-dyad points with common-slope regression lines per dyad

    Args:
        merged (pd.DataFrame): Synchrony and performance data merged on
            dyadNumber × iteration (from test_performance_correlation merge step).
        primary_metric (str): Primary performance column to plot first.
    """
    if merged is None or len(merged) == 0:
        print("fig_performance_scatter: no data to plot.")
        return

    sync_col = "expressivity__zero_r"

    for perf_col in [primary_metric, "team_payload_correct"]:
        if perf_col not in merged.columns:
            continue

        plot_data = merged[[sync_col, perf_col, "dyadNumber", "zoom"]].dropna()
        x_all     = plot_data[sync_col].values
        y_all     = plot_data[perf_col].values

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        ax = axes[0]
        for zoom_val, label, color, marker in [
            (True,  "Zoom On",  ZOOM_ON_COLOR,  "o"),
            (False, "Zoom Off", ZOOM_OFF_COLOR, "s"),
        ]:
            sub = plot_data[plot_data["zoom"] == zoom_val]
            ax.scatter(sub[sync_col], sub[perf_col],
                       color=color, alpha=0.65, s=45, label=label,
                       edgecolors="white", linewidth=0.3, marker=marker)
        if np.std(x_all) > 1e-10:
            m, b = np.polyfit(x_all, y_all, 1)
            xl    = np.linspace(x_all.min(), x_all.max(), 100)
            ax.plot(xl, m * xl + b, color="black", lw=1.5, ls="--", alpha=0.7)
        r, p = spearmanr(x_all, y_all)
        ax.text(0.05, 0.95, f"Spearman r = {r:.3f}\np = {p:.3f}",
                transform=ax.transAxes, va="top", fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        ax.set_xlabel("Expressivity zero-lag synchrony (r)", fontsize=11)
        ax.set_ylabel(perf_col.replace("_", " "), fontsize=11)
        ax.set_title("Expressivity Synchrony vs Performance",
                     fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)

        ax    = axes[1]
        dyads = plot_data["dyadNumber"].unique()
        cmap  = plt.cm.get_cmap("tab20", len(dyads))
        slope = np.polyfit(x_all, y_all, 1)[0] if np.std(x_all) > 1e-10 else 0

        for i, dyad in enumerate(dyads):
            grp = plot_data[plot_data["dyadNumber"] == dyad]
            ax.scatter(grp[sync_col], grp[perf_col],
                       color=cmap(i), alpha=0.6, s=35,
                       edgecolors="white", linewidth=0.3)
            if len(grp) > 1 and np.std(grp[sync_col]) > 1e-10:
                intercept = np.mean(grp[perf_col]) - slope * np.mean(grp[sync_col])
                xl        = np.array([grp[sync_col].min(), grp[sync_col].max()])
                ax.plot(xl, slope * xl + intercept,
                        color=cmap(i), alpha=0.45, lw=1.2)

        ax.set_xlabel("Expressivity zero-lag synchrony (r)", fontsize=11)
        ax.set_ylabel(perf_col.replace("_", " "), fontsize=11)
        ax.set_title("Per-Dyad Regression Lines (common slope)",
                     fontsize=12, fontweight="bold")
        ax.text(0.05, 0.95, f"N = {len(dyads)} dyads",
                transform=ax.transAxes, fontsize=9, va="top", color="gray")

        fig.suptitle(f"Expressivity Synchrony vs {perf_col.replace('_', ' ')} "
                     f"— Collaborative Runs",
                     fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.show()

def fig_trajectories(merged):
    """
    Synchrony and performance trajectories across collaborative runs.

    Shows expressivity synchrony (left) and team performance (right) across
    runs 0-7, illustrating the opposing learning-effect trajectories that
    serve as a potential confound for the negative synchrony-performance
    relationship.

    This is Figure 11 in the thesis.

    Args:
        merged (pd.DataFrame): Synchrony and performance data merged on
            dyadNumber × iteration.
    """
    if merged is None or len(merged) == 0:
        print("fig_trajectories: no data to plot.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    run_sync = (merged.groupby("iteration")["expressivity__zero_r"]
                .agg(["mean", "sem"]).reset_index()
                .sort_values("iteration"))
    x = np.arange(len(run_sync))
    ax.bar(x, run_sync["mean"], yerr=run_sync["sem"], capsize=4,
           color="#6BAE75", edgecolor="white", error_kw={"elinewidth": 1.2})
    ax.axhline(0, color="black", lw=0.6, ls="--", alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Run {it}" for it in run_sync["iteration"]],
                       rotation=30, ha="right")
    ax.set_ylabel("Mean expressivity zero-lag r", fontsize=11)
    ax.set_title("Expressivity Synchrony Across Runs",
                 fontsize=12, fontweight="bold")

    ax = axes[1]
    pm = None
    for perf_col, color in zip(
        ["team_total_score", "team_payload_correct"],
        ["#5B8DB8", "#E07B54"]
    ):
        if perf_col not in merged.columns:
            continue
        pm = (merged.groupby("iteration")[perf_col]
              .agg(["mean", "sem"]).reset_index()
              .sort_values("iteration"))
        x = np.arange(len(pm))
        ax.plot(x, pm["mean"], marker="o", lw=2, color=color,
                label=perf_col.replace("_", " "))
        ax.fill_between(x, pm["mean"] - pm["sem"], pm["mean"] + pm["sem"],
                        alpha=0.15, color=color)
    if pm is not None:
        ax.set_xticks(x)
        ax.set_xticklabels([f"Run {it}" for it in pm["iteration"]],
                           rotation=30, ha="right")
    ax.set_ylabel("Mean performance", fontsize=11)
    ax.set_title("Team Performance Across Runs",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)

    fig.suptitle("Synchrony and Performance Trajectories Across Collaborative Runs\n"
                 "(learning effect check)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.show()

def fig_raw_vs_corrected(surr_corr_df, perf_metric="team_total_score"):
    """
    Side-by-side bar chart comparing raw and surrogate-corrected
    synchrony–performance correlations per signal.

    Left panel shows raw Spearman r (orange = p<.05 uncorrected).
    Right panel shows surrogate-corrected r (blue = significant after FDR).

    This is Figure 10 in the thesis.

    Args:
        surr_corr_df (pd.DataFrame): Output of test_surrogate_corrected_performance().
        perf_metric (str): Performance column to display (default: team_total_score).
    """
    if surr_corr_df is None or len(surr_corr_df) == 0:
        print("fig_raw_vs_corrected: no data to plot.")
        return

    compare = surr_corr_df[
        (surr_corr_df["perf_metric"] == perf_metric) &
        (surr_corr_df["sync_metric"] == "zero_r")
    ].copy()

    if len(compare) == 0:
        print(f"fig_raw_vs_corrected: no data for {perf_metric} / zero_r.")
        return

    labels   = compare["signal"].tolist()
    r_raw    = compare["r_raw"].tolist()
    r_corr   = compare["r_corrected"].tolist()
    sig_mask = compare["significant"].tolist()
    y        = np.arange(len(labels))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

    ax = axes[0]
    raw_colors = ["#E07B54" if p < 0.05 else "lightgray"
                  for p in compare["p_raw_raw"]]
    ax.barh(y, r_raw, color=raw_colors, edgecolor="white", height=0.6)
    ax.axvline(0, color="black", lw=0.8, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Spearman r", fontsize=11)
    ax.set_title("Raw synchrony\nvs team total score",
                 fontsize=12, fontweight="bold")
    ax.text(0.98, 0.02, "orange = p<.05 (uncorrected)",
            transform=ax.transAxes, ha="right", fontsize=8, color="gray")

    ax = axes[1]
    corr_colors = ["#5B8DB8" if s else "lightgray" for s in sig_mask]
    ax.barh(y, r_corr, color=corr_colors, edgecolor="white", height=0.6)
    ax.axvline(0, color="black", lw=0.8, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Spearman r", fontsize=11)
    ax.set_title("Surrogate-corrected synchrony\nvs team total score",
                 fontsize=12, fontweight="bold")
    ax.text(0.98, 0.02, "blue = significant after FDR",
            transform=ax.transAxes, ha="right", fontsize=8, color="gray")

    fig.suptitle(
        "Raw vs Surrogate-Corrected Synchrony–Performance Correlations\n"
        "(Zero-lag r, team total score)",
        fontsize=13, fontweight="bold"
    )
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    agg      = pd.read_csv(AGG_FILE)
    raw      = pd.read_csv(RAW_FILE)
    surr_df  = pd.read_csv(SURR_FILE)
    phase_df = pd.read_csv(PHASE_FILE)
    zoom_df  = pd.read_csv(ZOOM_FILE)
    agg["phase_label"] = agg["phase"].map(PHASE_MAP)
    print(f"  {len(agg)} real pairs, {len(raw)} total pairs")
    print(f"  Phases: {agg['phase'].value_counts().to_dict()}")
    print(f"  Dyads:  {agg['nav_dyad'].nunique()}")

    # Phase comparison
    fig_expressivity_by_phase(agg, phase_df)
    fig_synchrony_heatmap(agg)
    fig_perdyad_spaghetti(agg)
    fig_emotion_by_phase(agg)

    # Real vs surrogate
    fig_real_vs_surrogate_violins(raw, surr_df)
    fig_difference_scores(agg, surr_df)
    fig_effect_sizes_real_vs_surrogate(surr_df)

    # Zoom effect
    fig_zoom_over_blocks(agg)
    fig_zoom_effect_heatmap(zoom_df)
    fig_zoom_discussion_dotplot(agg)

    # Performance heatmap
    if os.path.exists(PERF_FILE):
        perf_df = pd.read_csv(PERF_FILE)
        fig_performance_heatmap(perf_df)
    else:
        print(f"  Performance stats file not found at {PERF_FILE} — skipping heatmap.")

    # Performance scatter
    from statistical_analysis import load_performance_tsv
    perf_raw = load_performance_tsv(PERFORMANCE_FILE)
    if perf_raw is not None:
        reschu                = agg[agg["phase"] == "run"].copy()
        reschu["iteration"]   = reschu["iteration"].astype(str)
        reschu["dyadNumber"]  = reschu["nav_dyad"].astype(str)
        perf_raw["iteration"] = perf_raw["iteration"].astype(str)
        merged = reschu.merge(
            perf_raw[["dyadNumber", "iteration"] + PERFORMANCE_COLS],
            on=["dyadNumber", "iteration"], how="inner"
        )
        if len(merged) == 0:
            print("  Performance scatter: no rows matched after merge — skipping.")
        else:
            print(f"  Performance scatter: {len(merged)} rows merged.")
            fig_performance_scatter(merged, primary_metric=PRIMARY_METRIC)
            fig_trajectories(merged)
            surr_corr_df = pd.read_csv(
                os.path.join(ANALYSIS_DIR, "stats_surrogate_corrected_performance.csv")
            ) if os.path.exists(
                os.path.join(ANALYSIS_DIR, "stats_surrogate_corrected_performance.csv")
            ) else None
            fig_raw_vs_corrected(surr_corr_df, perf_metric=PRIMARY_METRIC)
    else:
        print("  Performance scatter: TSV not found — skipping.")

    # Windowed synchrony
    fig_windowed_synchrony_by_phase(agg)
    fig_windowed_variability(agg)
    fig_windowed_zoom_over_blocks(agg)
    fig_mean_vs_variability_scatter(agg)

    print("\nAll figures shown.")