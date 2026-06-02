from pathlib import Path

DATA_DIR         = Path("data/raw")
RESAMPLED_DIR    = Path("data/resampled")
ANALYSIS_DIR     = Path("results")
PERFORMANCE_FILE = Path("data/task-reschu_performance.tsv")


PRIMARY_METRIC   = "team_total_score"
PERFORMANCE_COLS = ["team_total_score", "team_payload_correct", "team_damage"]

SYNC_SIGNALS = [
    "expressivity",
    "emotion_happiness",
    "emotion_sadness",
    "emotion_fear",
    "emotion_anger",
    "emotion_surprise",
    "emotion_disgust",
]
SYNC_METRICS = ["zero_r", "peak_r"]

try:
    from config_local import *
except ImportError:
    pass