"""
config.py — Central configuration for the facial synchrony pipeline.

Update the four path constants below to match your local folder structure
before running any pipeline script.
"""

from pathlib import Path

DATA_DIR         = Path(r"C:\Users\olivi\OneDrive\Radboud AI\year 3\Thesis\final_data\Data")
RESAMPLED_DIR    = Path(r"C:\Users\olivi\OneDrive\Radboud AI\year 3\Thesis\final_data_30hz")
ANALYSIS_DIR     = Path(r"C:\Users\olivi\OneDrive\Radboud AI\year 3\Thesis\data_analysis")
PERFORMANCE_FILE = Path(r"C:\Users\olivi\OneDrive\Radboud AI\year 3\Thesis\task-reschu_performance.tsv")

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