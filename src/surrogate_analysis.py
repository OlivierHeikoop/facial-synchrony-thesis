"""
surrogate_analysis.py — Surrogate dyad index construction.
 
Builds a complete table of all real and surrogate navigator-pilot pairings,
used as a chance-level baseline for synchrony analysis.
 
A surrogate pair replaces the real pilot with a pilot from a different dyad
who completed the same block under the same zoom condition. This preserves
the temporal structure of both time series while destroying the dyad-specific
pairing, so any remaining synchrony reflects shared task structure rather
than genuine interpersonal coupling.
 
Run after missing_data.py.
"""

import os
import pandas as pd
from config import DATA_DIR, ANALYSIS_DIR
from preproccesing import all_thresholds

MATCHED_FILE = os.path.join(DATA_DIR, "matched_files.csv")

def get_zoom(participant_id):
    """
    Derive the zoom condition for a participant from their ID number.
 
    Zoom condition was counterbalanced by ID: IDs 1-2, 5-6, 9-10, ... are zoom=True;
    IDs 3-4, 7-8, 11-12, ... are zoom=False.
 
    Args:
        participant_id (int): Numeric participant ID.
 
    Returns:
        bool: True if the participant was in the zoom=on condition, False otherwise.
    """
    return bool((participant_id - 1) % 4 < 2)
 
def get_dyad_zoom(dyad):
    """
    Derive the zoom condition for a dyad from the navigator's participant ID.
 
    Args:
        dyad (str): Dyad identifier in 'navID_pilID' format.
 
    Returns:
        bool: Zoom condition of the dyad's navigator.
    """
    nav_id = int(dyad.split('_')[0])
    return get_zoom(nav_id)

def build_surrogate_index(matched, valid_dyads):
    """
    Build the full surrogate pairing index for all valid dyads and blocks.
 
    For each navigator in valid_dyads, pairs them with every pilot (including
    their own) who completed the same task block under the same zoom condition.
    Real pairs are flagged with is_paired=True; all others are surrogate pairs.
 
    Args:
        matched (pd.DataFrame): Matched file index from matched_files.csv,
            filtered to valid dyads only.
        valid_dyads (list): Dyad identifiers that passed all inclusion thresholds.
 
    Returns:
        pd.DataFrame: Surrogate index with columns:
            nav_dyad, pil_dyad, task, phase, iteration, zoom, is_paired,
            nav_id, pil_id.
    """
    rows = []
    for dyad_a in valid_dyads:
        nav_id = dyad_a.split('_')[0]
        zoom_a = get_dyad_zoom(dyad_a)
        nav_blocks = matched[matched['dyadNumber'] == dyad_a]

        for _, block in nav_blocks.iterrows():
            task, phase, iteration = block['task'], block['phase'], block['iteration']

            candidates = matched[
                (matched['task'] == task) &
                (matched['phase'] == phase) &
                (matched['iteration'] == iteration)
            ]

            for _, candidate in candidates.iterrows():
                dyad_b = candidate['dyadNumber']
                pil_id_b = dyad_b.split('_')[1]
                zoom_b = get_dyad_zoom(dyad_b)

                if zoom_b != zoom_a:
                    continue

                rows.append({
                    'nav_dyad': dyad_a,
                    'pil_dyad': dyad_b,
                    'task': task,
                    'phase': phase,
                    'iteration': iteration,
                    'zoom': zoom_a,
                    'is_paired': (dyad_b == dyad_a),
                    'nav_id': nav_id,
                    'pil_id': pil_id_b,
                })

    surrogate_index = pd.DataFrame(rows)
    print(f"   Surrogate index built:")
    print(f"   Real pairs:       {surrogate_index['is_paired'].sum()}")
    print(f"   Surrogate pairs:  {(~surrogate_index['is_paired']).sum()}")
    print(f"   Zoom=True pairs:  {(surrogate_index['zoom'] == True).sum()}")
    print(f"   Zoom=False pairs: {(surrogate_index['zoom'] == False).sum()}")
    return surrogate_index

if __name__ == "__main__":
    valid_dyads = all_thresholds(MATCHED_FILE)
    matched = pd.read_csv(MATCHED_FILE)
    matched = matched[matched['dyadNumber'].isin(valid_dyads)].reset_index(drop=True)

    # Verify Zoom condition per dyad
    print("\nZoom condition per dyad:")
    for dyad in sorted(valid_dyads):
        print(f"  {dyad}: zoom={get_dyad_zoom(dyad)}")

    # Build and save surrogate index
    surrogate_index = build_surrogate_index(matched, valid_dyads)
    out_path = os.path.join(ANALYSIS_DIR, "surrogate_index.csv")
    surrogate_index.to_csv(out_path, index=False)
    print(f"\nSaved surrogate_index.csv to {out_path}")