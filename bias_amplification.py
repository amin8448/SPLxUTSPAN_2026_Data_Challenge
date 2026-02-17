import pandas as pd
import numpy as np

TARGET_RANGES = {
    'angle': (30, 60),
    'depth': (-12, 30),
    'left_right': (-16, 16)
}


def scaled_to_original(scaled, target):
    min_val, max_val = TARGET_RANGES[target]
    return scaled * (max_val - min_val) + min_val


def original_to_scaled(original, target):
    min_val, max_val = TARGET_RANGES[target]
    return (original - min_val) / (max_val - min_val)


def apply_participant_shifts(submission_df, test_df, lr_shifts, depth_shifts=None, angle_shifts=None):
    result = submission_df.copy()
    masks = {p: test_df['participant_id'].values == p for p in range(1, 6)}
    
    if lr_shifts:
        lr = scaled_to_original(result['scaled_left_right'].values, 'left_right')
        for p, shift in lr_shifts.items():
            lr[masks[p]] += shift
        result['scaled_left_right'] = original_to_scaled(lr, 'left_right').clip(0, 1)
    
    if depth_shifts:
        depth = scaled_to_original(result['scaled_depth'].values, 'depth')
        for p, shift in depth_shifts.items():
            depth[masks[p]] += shift
        result['scaled_depth'] = original_to_scaled(depth, 'depth').clip(0, 1)
    
    if angle_shifts:
        angle = scaled_to_original(result['scaled_angle'].values, 'angle')
        for p, shift in angle_shifts.items():
            angle[masks[p]] += shift
        result['scaled_angle'] = original_to_scaled(angle, 'angle').clip(0, 1)
    
    return result


def create_blended_submission(sub1_path, sub2_path, weight1, output_path):
    sub1 = pd.read_csv(sub1_path)
    sub2 = pd.read_csv(sub2_path)
    
    result = pd.DataFrame({'id': sub1['id']})
    result['scaled_angle'] = weight1 * sub1['scaled_angle'] + (1 - weight1) * sub2['scaled_angle']
    result['scaled_depth'] = weight1 * sub1['scaled_depth'] + (1 - weight1) * sub2['scaled_depth']
    result['scaled_left_right'] = weight1 * sub1['scaled_left_right'] + (1 - weight1) * sub2['scaled_left_right']
    
    result.to_csv(output_path, index=False)
    return result


def main():
    data_path = 'data'
    submission_path = 'submissions'
    
    test_df = pd.read_csv(f'{data_path}/test.csv')
    
    sub1 = pd.read_csv(f'{submission_path}/submission_e28_blend50_lr.csv')
    sub2 = pd.read_csv(f'{submission_path}/submission_e27_blend70_30.csv')
    
    blend_weight = 0.70
    blended = pd.DataFrame({'id': test_df['id']})
    blended['scaled_angle'] = blend_weight * sub1['scaled_angle'] + (1 - blend_weight) * sub2['scaled_angle']
    blended['scaled_depth'] = blend_weight * sub1['scaled_depth'] + (1 - blend_weight) * sub2['scaled_depth']
    blended['scaled_left_right'] = blend_weight * sub1['scaled_left_right'] + (1 - blend_weight) * sub2['scaled_left_right']
    
    lr_shifts = {1: -0.3, 2: -0.2, 3: +0.2, 4: +0.7, 5: -1.2}
    depth_shifts = {1: +0.5, 4: -0.7}
    
    final = apply_participant_shifts(blended, test_df, lr_shifts, depth_shifts)
    final.to_csv(f'{submission_path}/final_submission.csv', index=False)
    
    print("Final submission created with per-participant bias amplification")
    print(f"LR shifts: {lr_shifts}")
    print(f"Depth shifts: {depth_shifts}")


if __name__ == '__main__':
    main()
