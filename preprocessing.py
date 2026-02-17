import pandas as pd
import numpy as np
import ast


def parse_timeseries(cell):
    if isinstance(cell, str):
        cleaned = cell.replace('nan', 'None')
        parsed = ast.literal_eval(cleaned)
        arr = np.array(parsed, dtype=np.float64)
        if np.any(np.isnan(arr)):
            mask = np.isnan(arr)
            if np.any(~mask):
                arr[mask] = np.interp(
                    np.flatnonzero(mask), 
                    np.flatnonzero(~mask), 
                    arr[~mask]
                )
            else:
                arr[mask] = 0
        return arr
    return np.array(cell, dtype=np.float64)


KEYPOINTS_19 = [
    'nose', 'neck', 'mid_hip', 'left_shoulder', 'right_shoulder',
    'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist',
    'left_hip', 'right_hip', 'left_knee', 'right_knee',
    'left_ankle', 'right_ankle', 'right_first_finger_distal',
    'right_second_finger_distal', 'left_first_finger_distal',
    'left_second_finger_distal'
]


def extract_sequences(df, keypoints=KEYPOINTS_19):
    n_samples = len(df)
    n_kp = len(keypoints)
    seqs = np.zeros((n_samples, n_kp * 3, 240))
    
    for i in range(n_samples):
        row = df.iloc[i]
        for j, kp in enumerate(keypoints):
            for k, coord in enumerate(['x', 'y', 'z']):
                col = f'{kp}_{coord}'
                if col in df.columns:
                    seqs[i, j*3 + k, :] = parse_timeseries(row[col])
    return seqs


def normalize_by_participant(train_seq, test_seq, train_pid, test_pid):
    train_norm = train_seq.copy()
    test_norm = test_seq.copy()
    
    for p in np.unique(train_pid):
        train_mask = train_pid == p
        test_mask = test_pid == p
        
        for c in range(train_seq.shape[1]):
            mean = train_seq[train_mask, c, :].mean()
            std = train_seq[train_mask, c, :].std() + 1e-8
            train_norm[train_mask, c, :] = (train_seq[train_mask, c, :] - mean) / std
            if test_mask.sum() > 0:
                test_norm[test_mask, c, :] = (test_seq[test_mask, c, :] - mean) / std
    
    return train_norm, test_norm


def body_center_sequences(sequences, df, keypoints=KEYPOINTS_19):
    centered = sequences.copy()
    hip_idx = keypoints.index('mid_hip')
    
    for i in range(len(sequences)):
        hip_x = sequences[i, hip_idx * 3, :]
        hip_y = sequences[i, hip_idx * 3 + 1, :]
        hip_z = sequences[i, hip_idx * 3 + 2, :]
        
        for j in range(len(keypoints)):
            centered[i, j * 3, :] -= hip_x
            centered[i, j * 3 + 1, :] -= hip_y
            centered[i, j * 3 + 2, :] -= hip_z
    
    return centered
