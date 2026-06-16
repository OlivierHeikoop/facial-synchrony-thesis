import sys
sys.path.insert(0, '.')

import os
import pandas as pd
from config import ANALYSIS_DIR, PRIMARY_METRIC
from src.visualise_results import fig_raw_vs_corrected

surr_corr_path = os.path.join(ANALYSIS_DIR, "stats_surrogate_corrected_performance.csv")
surr_corr_df = pd.read_csv(surr_corr_path)

fig_raw_vs_corrected(surr_corr_df, perf_metric=PRIMARY_METRIC)