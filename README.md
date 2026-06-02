# Context-Dependent Facial Synchrony in Teams
Olivier Heikoop
Bachelor Thesis, Artificial Intelligence, Radboud University (July 2026)

This repository contains the analysis pipeline for my bachelor thesis examining whether facial synchrony in teams is context-dependent across task phases and whether it predicts team performance in a human-autonomy teaming (HAT) context.

## Research Summary

Facial expressions were recorded continuously for 29 navigator–pilot dyads completing the [RESCHU] simulated UAV task. OpenFace-derived Action Unit intensities were decomposed into an expressivity composite and six emotion-specific signals. Synchrony was computed across three task phases and compared against a surrogate dyad baseline to separate genuine interpersonal coupling from shared task-driven expression.

**Key findings:**
- Real dyads exceeded the surrogate baseline in 29/42 tests after FDR correction — genuine interpersonal coupling was present across all three phases, including the passive Instructional phase.
- No significant phase differences in synchrony, though a numerical trend (Discussion > Collaborative > Instructional) was observed.
- Zoom manipulation (visual access to partner's face) produced no significant effects, likely due to insufficient power.
- Facial synchrony during collaborative runs was **negatively** associated with team performance — contradicting Rabin et al. (2024) and persisting after surrogate correction. Whether this reflects a learning-effect confound or a genuine negative relationship remains an open question.

## Task Phases

| Phase | Description |
|---|---|
| Instructional | Shared 10-min instructional video — passive, no interaction |
| Discussion | 5-min free discussion before/between run sets |
| Collaborative | 8 RESCHU runs × 5 min — active UAV navigation task |

## Facial Signals

| Signal | Action Units |
|---|---|
| `expressivity` | Sum of all 17 AU intensities |
| `emotion_happiness` | AU6, AU12 |
| `emotion_sadness` | AU1, AU4, AU15 |
| `emotion_fear` | AU1, AU2, AU4, AU5, AU7, AU20, AU26 |
| `emotion_anger` | AU4, AU5, AU7, AU23, AU24 |
| `emotion_surprise` | AU1, AU2, AU5, AU26 |
| `emotion_disgust` | AU9, AU15, AU16, AU25, AU26 |

AU classifications follow Ekman & Friesen (1978), with minor deviations noted in the thesis.

## Synchrony Metrics

| Metric | Description |
|---|---|
| `zero_r` | Zero-lag Pearson r — simultaneous facial co-movement |
| `peak_r` | Peak cross-correlation within ±5 s lag |
| `peak_lag_s` | Lag (seconds) at peak correlation |
| `win_mean_r` | Mean rolling Pearson r (10 s windows, 5 s step) |
| `win_sd_r` | SD of rolling Pearson r — temporal variability of synchrony |

## Surrogate Dyad Method

For each navigator, all pilots from different dyads who completed the same block under the same zoom condition were identified as surrogate candidates. Synchrony was computed for every real and surrogate pairing (265 real + 3080 surrogate = 3345 total). Difference scores (real minus mean surrogate) isolate dyad-specific coupling from shared task structure.

## Pipeline

Run scripts in this order:

```
src/preproccesing.py        1. Index files, match dyads, apply inclusion thresholds, resample to 30 Hz
src/surrogate_analysis.py   2. Build real and surrogate pair index
src/synchrony_analysis.py   3. Compute synchrony metrics for all pairs
src/statisical_analysis.py  4. Statistical tests across all research questions
src/visualise_results.py    5. Figures and plots
```

**Inclusion thresholds:** dyads required ≥4 matched RESCHU blocks, ≥1 instructional block, ≥1 discussion block. 29 of 50 dyads were retained.

## Setup

```bash
pip install -r requirements.txt
```

Create `config_local.py` in the repo root with your local data paths (this file is gitignored):

```python
from pathlib import Path
DATA_DIR         = Path("path/to/raw/data")
RESAMPLED_DIR    = Path("path/to/resampled/output")
ANALYSIS_DIR     = Path("path/to/results")
PERFORMANCE_FILE = Path("path/to/task-reschu_performance.tsv")
```

Defaults in `config.py` point to `data/` and `results/` relative to the repo root.

## Data Format

Input: OpenFace 2.0 CSV output per participant, organized as:
```
<DATA_DIR>/<dyadID>/raw/<participantID>/pp<ID>_<role>_AU_<task>_<phase>_<iteration>.csv
```

Files recorded at rates other than 30 Hz are resampled via time-based linear interpolation. Binary AU presence columns (`*_c`) are rounded back to integers after resampling.

## Participants

50 dyads recruited; 29 retained (58 participants, 42F/16M, mean age 20.9). Dyads were assigned to zoom = on (n=14) or zoom = off (n=15) based on counterbalanced participant IDs. Participants completed the task seated back-to-back, communicating by speech only.

## Data Availability

The raw facial expression data used in this study cannot be shared publicly. Data are stored on a restricted institutional repository and contain recordings of human participants, subject to ethical and privacy constraints.

## References

- Rabin et al. (2024) — positive synchrony–performance relationship in collaborative bomb defusal
- Frohn (2025) — CRQA on the same dataset; no above-chance recurrence in real dyads
- Baltrušaitis et al. (2018) — OpenFace 2.0
- Ekman & Friesen (1978) — FACS action unit prototypes
- Nehme et al. (2009) — RESCHU task

