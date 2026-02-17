import pandas as pd
import numpy as np
from preprocessing import parse_timeseries, KEYPOINTS_19


def extract_statistical_features(df, keypoints=KEYPOINTS_19):
    features = []
    
    for i in range(len(df)):
        row = df.iloc[i]
        feat = {'participant_id': row['participant_id']}
        
        for kp in keypoints:
            for coord in ['x', 'y', 'z']:
                col = f'{kp}_{coord}'
                if col not in df.columns:
                    continue
                    
                ts = parse_timeseries(row[col])
                prefix = f'{kp}_{coord}'
                
                feat[f'{prefix}_mean'] = ts.mean()
                feat[f'{prefix}_std'] = ts.std()
                feat[f'{prefix}_min'] = ts.min()
                feat[f'{prefix}_max'] = ts.max()
                feat[f'{prefix}_range'] = ts.max() - ts.min()
                feat[f'{prefix}_median'] = np.median(ts)
                
                velocity = np.diff(ts)
                feat[f'{prefix}_vel_mean'] = velocity.mean()
                feat[f'{prefix}_vel_std'] = velocity.std()
                feat[f'{prefix}_vel_max'] = np.abs(velocity).max()
                
                accel = np.diff(velocity)
                feat[f'{prefix}_acc_mean'] = accel.mean()
                feat[f'{prefix}_acc_std'] = accel.std()
                feat[f'{prefix}_acc_max'] = np.abs(accel).max()
                
                early = ts[:80]
                mid = ts[80:160]
                late = ts[160:]
                
                feat[f'{prefix}_early_mean'] = early.mean()
                feat[f'{prefix}_mid_mean'] = mid.mean()
                feat[f'{prefix}_late_mean'] = late.mean()
                feat[f'{prefix}_early_std'] = early.std()
                feat[f'{prefix}_mid_std'] = mid.std()
                feat[f'{prefix}_late_std'] = late.std()
        
        features.append(feat)
    
    return pd.DataFrame(features)


def find_release_frame(df_row):
    wrist_z = parse_timeseries(df_row['right_wrist_z'])
    search_window = wrist_z[-120:]
    release_idx = np.argmax(search_window) + (len(wrist_z) - 120)
    return min(release_idx, len(wrist_z) - 10)


def extract_physics_features(df):
    features = []
    
    for i in range(len(df)):
        row = df.iloc[i]
        feat = {'participant_id': row['participant_id']}
        
        release_frame = find_release_frame(row)
        feat['release_frame'] = release_frame
        
        shoulder = np.array([
            parse_timeseries(row['right_shoulder_x'])[release_frame],
            parse_timeseries(row['right_shoulder_y'])[release_frame],
            parse_timeseries(row['right_shoulder_z'])[release_frame]
        ])
        elbow = np.array([
            parse_timeseries(row['right_elbow_x'])[release_frame],
            parse_timeseries(row['right_elbow_y'])[release_frame],
            parse_timeseries(row['right_elbow_z'])[release_frame]
        ])
        wrist = np.array([
            parse_timeseries(row['right_wrist_x'])[release_frame],
            parse_timeseries(row['right_wrist_y'])[release_frame],
            parse_timeseries(row['right_wrist_z'])[release_frame]
        ])
        
        upper_arm = elbow - shoulder
        forearm = wrist - elbow
        cos_angle = np.dot(upper_arm, forearm) / (
            np.linalg.norm(upper_arm) * np.linalg.norm(forearm) + 1e-8
        )
        feat['elbow_angle_release'] = np.arccos(np.clip(cos_angle, -1, 1))
        
        hip_z = parse_timeseries(row['mid_hip_z'])[release_frame]
        feat['release_height'] = wrist[2] - hip_z
        
        if release_frame > 0:
            wrist_prev = np.array([
                parse_timeseries(row['right_wrist_x'])[release_frame-1],
                parse_timeseries(row['right_wrist_y'])[release_frame-1],
                parse_timeseries(row['right_wrist_z'])[release_frame-1]
            ])
            feat['wrist_velocity_mag'] = np.linalg.norm(wrist - wrist_prev)
        else:
            feat['wrist_velocity_mag'] = 0
        
        shoulder_x = parse_timeseries(row['right_shoulder_x'])[release_frame]
        hip_x = parse_timeseries(row['mid_hip_x'])[release_frame]
        feat['body_lean'] = shoulder_x - hip_x
        
        f1 = parse_timeseries(row['right_first_finger_distal_x'])[release_frame]
        f2 = parse_timeseries(row['right_second_finger_distal_x'])[release_frame]
        feat['finger_spread'] = abs(f1 - f2)
        
        features.append(feat)
    
    return pd.DataFrame(features)


def combine_all_features(df):
    stat_features = extract_statistical_features(df)
    phys_features = extract_physics_features(df)
    
    combined = stat_features.merge(
        phys_features, 
        on='participant_id', 
        suffixes=('', '_phys')
    )
    
    return combined
