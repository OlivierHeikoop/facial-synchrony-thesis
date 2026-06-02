"""
missing_data.py — Preprocessing pipeline for facial synchrony analysis.
 
Steps:
    1. Index all participant AU files and compute sampling rates.
    2. Match navigator and pilot files per dyad, task, phase and iteration.
    3. Apply inclusion thresholds to identify valid dyads.
    4. Resample all files to 30 Hz and save to RESAMPLED_DIR.
 
Run this script first before any other pipeline step.
"""

import os
import pandas as pd
import numpy as np
from config import DATA_DIR, RESAMPLED_DIR

def participant_files(data_folder):
    """
    Index all participant AU CSV files in data_folder and save a summary to participant_files.csv.
 
    Walks the dyad/raw/participant folder structure, parses filenames into components
    (ppNumber, role, task, phase, iteration) and computes the empirical sampling rate
    from each file's frame and time columns.
 
    Args:
        data_folder (str): Path to the root data directory.
 
    Saves:
        participant_files.csv to data_folder.
    """
    dyad_folders = [item for item in os.listdir(data_folder) if os.path.isdir(os.path.join(data_folder, item))]
    rows = []
    skipped = []

    for dyad in sorted(dyad_folders):
        raw_dyad_path = os.path.join(data_folder, dyad, "raw")
        try:
            for participant in os.listdir(raw_dyad_path):
                participant_path = os.path.join(raw_dyad_path, participant)
                for file in os.listdir(participant_path):
                    if not file.endswith(".csv"):
                        continue
                    full_path = os.path.join(participant_path, file)
                    try:
                        file_components = file.split(".csv")[0].split("_")
                        file_components.remove('AU')
                        data = pd.read_csv(full_path)
                        if 'frame' not in data.columns or 'time' not in data.columns:
                            skipped.append(f"Missing frame/time columns: {file}")
                            continue
                        if len(data) == 0 or data['time'].iloc[-1] == 0:
                            skipped.append(f"Empty or zero-duration file: {file}")
                            continue
                        sampling_rate = round(data['frame'].iloc[-1] / data['time'].iloc[-1], 2)
                        file_components.append(sampling_rate)
                        rows.append(file_components)
                    except Exception as e:
                        skipped.append(f"Error reading {file}: {e}")
        except:
            continue

    print(f" participant_files: {len(rows)} files indexed")
    print(f" Skipped: {len(skipped)} files")
    for s in skipped:
        print(f"   - {s}")

    output_path = os.path.join(data_folder, "participant_files.csv")
    df = pd.DataFrame(rows, columns=['ppNumber', 'role', 'task', 'phase', 'iteration', 'sampling_rate'])
    assert df.shape[1] == 6, f"Expected 6 columns, got {df.shape[1]}"
    print(f"   Sampling rates found: {sorted(df['sampling_rate'].unique())}")
    print(f"   Participants: {df['ppNumber'].nunique()}")
    print(f"   Tasks: {df['task'].unique()}")
    df.to_csv(output_path, index=False, sep=',')
    print(f"   Saved to {output_path}\n")

def matched_folder(data_folder):
    """
    Match navigator and pilot files within each dyad and save the result to matched_files.csv.
 
    A block is kept only if a valid recording exists for both roles (navigator and pilot)
    for the same task, phase and iteration. Unmatched files are reported but not included.
 
    Args:
        data_folder (str): Path to the root data directory.
 
    Saves:
        matched_files.csv to data_folder.
    """
    dyad_folders = [item for item in os.listdir(data_folder) if os.path.isdir(os.path.join(data_folder, item))]
    rows = []
    skipped_dyads = []

    for dyad in sorted(dyad_folders):
        raw_dyad_path = os.path.join(data_folder, dyad, "raw")
        pp_00 = []
        pp_01 = []

        try:
            participants = os.listdir(raw_dyad_path)
            if len(participants) != 2:
                skipped_dyads.append(f"{dyad}: expected 2 participant folders, found {len(participants)}")
                continue

            for idx, participant in enumerate(participants):
                participant_path = os.path.join(raw_dyad_path, participant)
                for file in os.listdir(participant_path):
                    if not file.endswith(".csv"):
                        continue
                    file_components = file.split(".csv")[0].split("_")
                    file_components.pop(0)
                    file_components.remove('AU')
                    if 'pilot' in file_components:
                        file_components.remove('pilot')
                    if 'navigator' in file_components:
                        file_components.remove('navigator')
                    file_components.insert(0, dyad)
                    if idx % 2 == 0:
                        pp_00.append(file_components)
                    if idx % 2 == 1:
                        pp_01.append(file_components)

            matched = [f for f in pp_00 if f in pp_01]
            only_00 = [f for f in pp_00 if f not in pp_01]
            only_01 = [f for f in pp_01 if f not in pp_00]

            if only_00:
                print(f"  {dyad}: {len(only_00)} file(s) only in participant 1 — skipped")
            if only_01:
                print(f"  {dyad}: {len(only_01)} file(s) only in participant 2 — skipped")

            rows.extend(matched)

        except Exception as e:
            skipped_dyads.append(f"{dyad}: {e}")
            continue

    print(f"\n matched_folder: {len(rows)} matched file pairs")
    if skipped_dyads:
        print(f"Skipped dyads:")
        for s in skipped_dyads:
            print(f"   - {s}")

    output_path = os.path.join(data_folder, "matched_files.csv")
    df = pd.DataFrame(rows, columns=['dyadNumber', 'task', 'phase', 'iteration'])
    assert df.shape[1] == 4, f"Expected 4 columns, got {df.shape[1]}"
    print(f"   Dyads matched: {df['dyadNumber'].nunique()}")
    print(f"   Tasks: {df['task'].unique()}")
    df.to_csv(output_path, index=False, sep=',')
    print(f"   Saved to {output_path}\n")

def threshold_participants(file_path, condition: str = 'reschu', minimum: int = 4):
    """
    Return dyads that have at least `minimum` matched blocks for a given task condition.
 
    Args:
        file_path (str): Path to matched_files.csv.
        condition (str): Task type to filter on (e.g. 'reschu', 'instructional', 'discussion').
        minimum (int): Minimum number of matched blocks required to include a dyad.
 
    Returns:
        list: Dyad identifiers that meet the threshold.
    """
    df = pd.read_csv(file_path)
    assert 'dyadNumber' in df.columns and 'task' in df.columns, \
        "matched_files.csv missing expected columns — did matched_folder() run correctly?"
    included_dyads = []
    for dyad_nr in df['dyadNumber'].unique():
        count = (df[df['dyadNumber'] == dyad_nr]['task'] == condition).sum()
        if count >= minimum:
            included_dyads.append(dyad_nr)
    print(f"   threshold_participants ({condition} >= {minimum}): {len(included_dyads)} dyads pass")
    return included_dyads

def all_thresholds(file_path):
    """
    Apply all inclusion thresholds and return dyads that pass all three criteria:
        - At least 4 matched RESCHU blocks
        - At least 1 matched instructional block
        - At least 1 matched discussion block
 
    Args:
        file_path (str): Path to matched_files.csv.
 
    Returns:
        list: Sorted dyad identifiers that pass all thresholds.
    """
    print("Applying inclusion thresholds...")
    reschu_pairs = threshold_participants(file_path, condition='reschu', minimum=4)
    instructional_pairs = threshold_participants(file_path, condition='instructional', minimum=1)
    discussion_pairs = threshold_participants(file_path, condition='discussion', minimum=1)

    common = set(reschu_pairs) & set(instructional_pairs) & set(discussion_pairs)

    df = pd.read_csv(file_path)
    dropped_reschu = set(df['dyadNumber'].unique()) - set(reschu_pairs)
    dropped_instructional = set(reschu_pairs) - set(instructional_pairs)
    dropped_discussion = set(instructional_pairs) - set(discussion_pairs)

    print(f"\n  all_thresholds: {len(common)} dyads pass all criteria")
    if dropped_reschu:
        print(f"   Dropped (reschu < 4): {sorted(dropped_reschu)}")
    if dropped_instructional:
        print(f"   Dropped (no instructional): {sorted(dropped_instructional)}")
    if dropped_discussion:
        print(f"   Dropped (no discussion): {sorted(dropped_discussion)}")

    return sorted(list(common))

def resample_to_30hz(df):
    """
    Resample a participant DataFrame to 30 Hz if it was recorded at a different rate.
 
    Files already at 30 Hz are returned unchanged. For other rates, a new evenly spaced
    time axis is constructed and the data is interpolated onto it. Binary AU presence
    columns (ending in '_c') are rounded back to integers after interpolation.
 
    Args:
        df (pd.DataFrame): Raw OpenFace output with 'frame' and 'time' columns.
 
    Returns:
        pd.DataFrame: DataFrame resampled to 30 Hz.
    """
    rate = round(df['frame'].iloc[-1] / df['time'].iloc[-1])
    if rate == 30:
        return df
    df = df.dropna(subset=['face_id']).sort_values('time').reset_index(drop=True)
    t_start = df['time'].iloc[0]
    t_end = df['time'].iloc[-1]
    new_time = np.arange(t_start, t_end, 1/30)
    df = df.set_index('time')
    df = df.reindex(df.index.union(new_time)).sort_index()
    df = df.interpolate(method='index')
    df = df.loc[new_time].reset_index().rename(columns={'index': 'time'})
    c_cols = [c for c in df.columns if c.endswith('_c')]
    df[c_cols] = df[c_cols].round().astype('Int64')
    return df

def load_and_clean(filepath):
    """
    Load a single participant CSV, resample to 30 Hz, and clean column names.
 
    Infinite values and NaNs are handled downstream in the synchrony computation step.
 
    Args:
        filepath (str): Path to the participant AU CSV file.
 
    Returns:
        pd.DataFrame: Cleaned DataFrame at 30 Hz with 'timestamp' as the time column.
    """
    df = pd.read_csv(filepath, index_col=0)
    df.columns = df.columns.str.strip()
    df = resample_to_30hz(df)
    df = df.rename(columns={'time': 'timestamp'})
    return df

def build_filepath(data_folder, dyad, task, phase, iteration):
    """
    Construct the expected file paths for the navigator and pilot of a given block.
 
    Args:
        data_folder (str): Path to the root data directory.
        dyad (str): Dyad identifier in 'navID_pilID' format.
        task (str): Task type (e.g. 'reschu', 'instructional', 'discussion').
        phase (str): Phase label.
        iteration (str or int): Block iteration number.
 
    Returns:
        tuple[str, str]: (navigator_path, pilot_path)
    """
    nav_id = dyad.split('_')[0]
    pil_id = dyad.split('_')[1]
    nav_file = f"pp{nav_id}_navigator_AU_{task}_{phase}_{iteration}.csv"
    pil_file = f"pp{pil_id}_pilot_AU_{task}_{phase}_{iteration}.csv"
    nav_path = os.path.join(data_folder, dyad, "raw", nav_id, nav_file)  # nav_id not pp{nav_id}
    pil_path = os.path.join(data_folder, dyad, "raw", pil_id, pil_file)  # pil_id not pp{pil_id}
    return nav_path, pil_path

if __name__ == "__main__":
    # Step 1: index and match files
    matched_folder(DATA_DIR)
    
    # Step 2: apply inclusion thresholds
    matched = pd.read_csv(os.path.join(DATA_DIR, "matched_files.csv"))
    valid_dyads = all_thresholds(os.path.join(DATA_DIR, "matched_files.csv"))
    matched = matched[matched['dyadNumber'].isin(valid_dyads)].reset_index(drop=True)
    
    # Step 3: load, verify and safe resamples files
    for _, row in matched.iterrows():
        nav_path, pil_path = build_filepath(
            DATA_DIR, row['dyadNumber'], row['task'], row['phase'], row['iteration']
        )
        nav = load_and_clean(nav_path)
        pil = load_and_clean(pil_path)
        
        nav_rate = round(len(nav) / nav['timestamp'].iloc[-1])
        pil_rate = round(len(pil) / pil['timestamp'].iloc[-1])
        
        if nav_rate != 30 or pil_rate != 30:
            print(f" Unexpected rate — dyad {row['dyadNumber']} | {row['task']} {row['phase']} {row['iteration']} — nav: {nav_rate} Hz, pil: {pil_rate} Hz")
        else:
            print(f" {row['dyadNumber']} | {row['task']} {row['phase']} {row['iteration']} — nav: {nav_rate} Hz, pil: {pil_rate} Hz")
            
        #step 4: save resampled files mirroring original folder structure
        for df, original_path in [(nav, nav_path), (pil, pil_path)]:
            relative = os.path.relpath(original_path, DATA_DIR)
            save_path = os.path.join(RESAMPLED_DIR, relative)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            df.to_csv(save_path)
    
    print("Done saving resampled files.")