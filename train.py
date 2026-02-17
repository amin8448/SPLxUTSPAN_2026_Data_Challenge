import pandas as pd
import numpy as np
import torch

from preprocessing import extract_sequences, normalize_by_participant
from features import extract_statistical_features, extract_physics_features
from transformer_model import train_transformer
from tree_model import train_tree_ensemble
from dilated_cnn import train_dilated_cnn


TARGET_RANGES = {
    'angle': (30, 60),
    'depth': (-12, 30),
    'left_right': (-16, 16)
}

TARGET_SCALE = {
    'angle': 30,
    'depth': 42,
    'left_right': 32
}


def original_to_scaled(original, target):
    min_val, max_val = TARGET_RANGES[target]
    return (original - min_val) / (max_val - min_val)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    data_path = 'data'
    submission_path = 'submissions'
    
    train_df = pd.read_csv(f'{data_path}/train.csv')
    test_df = pd.read_csv(f'{data_path}/test.csv')
    
    train_pid = train_df['participant_id'].values
    test_pid = test_df['participant_id'].values
    
    y_angle = train_df['angle'].values
    y_depth = train_df['depth'].values
    y_lr = train_df['left_right'].values
    
    print("Extracting sequences...")
    train_seq = extract_sequences(train_df)
    test_seq = extract_sequences(test_df)
    
    print("Normalizing by participant...")
    train_seq_norm, test_seq_norm = normalize_by_participant(
        train_seq, test_seq, train_pid, test_pid
    )
    
    print("Extracting features...")
    train_features = extract_statistical_features(train_df)
    test_features = extract_statistical_features(test_df)
    
    train_physics = extract_physics_features(train_df)
    test_physics = extract_physics_features(test_df)
    
    feat_cols = [c for c in train_features.columns if c != 'participant_id']
    X_train = train_features[feat_cols].fillna(0)
    X_test = test_features[feat_cols].fillna(0)
    
    print("\nTraining ANGLE models...")
    print("  Tree ensemble...")
    angle_tree_oof, angle_tree_test = train_tree_ensemble(X_train, X_test, y_angle)
    
    print("  Transformer...")
    angle_trans_oof, angle_trans_test = train_transformer(
        train_seq_norm, test_seq_norm, y_angle, train_pid, test_pid, device=device
    )
    
    angle_oof = 0.7 * angle_tree_oof + 0.3 * angle_trans_oof
    angle_test = 0.7 * angle_tree_test + 0.3 * angle_trans_test
    
    print("\nTraining DEPTH model...")
    depth_oof, depth_test = train_transformer(
        train_seq_norm, test_seq_norm, y_depth, train_pid, test_pid, device=device
    )
    
    print("\nTraining LEFT/RIGHT models...")
    print("  Transformer...")
    lr_trans_oof, lr_trans_test = train_transformer(
        train_seq_norm, test_seq_norm, y_lr, train_pid, test_pid, device=device
    )
    
    print("  Dilated CNN...")
    lr_dilated_oof, lr_dilated_test = train_dilated_cnn(
        train_seq_norm, test_seq_norm, y_lr, train_pid, test_pid, device=device
    )
    
    lr_oof = 0.5 * lr_trans_oof + 0.5 * lr_dilated_oof
    lr_test = 0.5 * lr_trans_test + 0.5 * lr_dilated_test
    
    print("\nOOF Results:")
    angle_mse = np.mean(((angle_oof - y_angle) / TARGET_SCALE['angle']) ** 2)
    depth_mse = np.mean(((depth_oof - y_depth) / TARGET_SCALE['depth']) ** 2)
    lr_mse = np.mean(((lr_oof - y_lr) / TARGET_SCALE['left_right']) ** 2)
    
    print(f"  Angle MSE: {angle_mse:.6f}")
    print(f"  Depth MSE: {depth_mse:.6f}")
    print(f"  LR MSE:    {lr_mse:.6f}")
    print(f"  Mean MSE:  {(angle_mse + depth_mse + lr_mse) / 3:.6f}")
    
    submission = pd.DataFrame({'id': test_df['id']})
    submission['scaled_angle'] = original_to_scaled(angle_test, 'angle').clip(0, 1)
    submission['scaled_depth'] = original_to_scaled(depth_test, 'depth').clip(0, 1)
    submission['scaled_left_right'] = original_to_scaled(lr_test, 'left_right').clip(0, 1)
    
    submission.to_csv(f'{submission_path}/submission_hybrid.csv', index=False)
    print(f"\nSubmission saved to {submission_path}/submission_hybrid.csv")
    
    oof_df = pd.DataFrame({
        'angle_tree_oof': angle_tree_oof,
        'angle_trans_oof': angle_trans_oof,
        'depth_oof': depth_oof,
        'lr_trans_oof': lr_trans_oof,
        'lr_dilated_oof': lr_dilated_oof
    })
    oof_df.to_csv(f'{data_path}/oof_predictions.csv', index=False)
    print(f"OOF predictions saved to {data_path}/oof_predictions.csv")


if __name__ == '__main__':
    main()
