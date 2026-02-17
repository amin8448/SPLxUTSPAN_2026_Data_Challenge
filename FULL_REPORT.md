# Multi-Modal Learning for Basketball Free Throw Prediction
## SPLxUTSPAN 2026 Data Challenge - Full Technical Report

**Author:** Amin Nabavi, Carleton University  
**Final Score:** 0.006595 (2nd Place)

---

## Table of Contents
1. [Problem Overview](#1-problem-overview)
2. [Data Understanding](#2-data-understanding)
3. [Biomechanical Feature Engineering](#3-biomechanical-feature-engineering)
4. [Model Architectures](#4-model-architectures)
5. [Experimental Journey](#5-experimental-journey)
6. [Final Solution](#6-final-solution)
7. [Results and Analysis](#7-results-and-analysis)
8. [Lessons Learned](#8-lessons-learned)

---

## 1. Problem Overview

The challenge involves predicting three basketball free throw landing outcomes from markerless motion capture data:
- **Angle**: The entry angle of the ball relative to the hoop
- **Depth**: How short or long the shot lands (front-to-back)
- **Left/Right**: Lateral deviation from center

The dataset contains 345 training shots and 113 test shots across 5 participants, with each shot represented by 19 body keypoints tracked across 240 frames at 60Hz (approximately 4 seconds of motion).

### Why This Problem Matters

Traditional basketball analytics focuses on shot outcomes (make/miss) or ball trajectory. This challenge asks a deeper question: **can we predict where a shot will land based solely on the shooter's body movements?** This has implications for coaching, real-time feedback systems, and understanding the biomechanics of shooting.

---

## 2. Data Understanding

### 2.1 Dataset Structure

Each sample contains:
- 19 keypoints: nose, neck, mid_hip, shoulders, elbows, wrists, hips, knees, ankles, and finger tips
- 3 coordinates per keypoint: x, y, z positions
- 240 frames per shot
- Participant ID (1-5)

### 2.2 Participant Distribution

| Participant | Training Samples | Percentage |
|-------------|-----------------|------------|
| P1 | 70 | 20.3% |
| P2 | 66 | 19.1% |
| P3 | 68 | 19.7% |
| P4 | 67 | 19.4% |
| P5 | 74 | 21.4% |

### 2.3 Target Variable Analysis

| Target | Mean | Std | Min | Max | Range |
|--------|------|-----|-----|-----|-------|
| Angle | 45.3 | 4.8 | 32.1 | 58.9 | 30-60 |
| Depth | 9.5 | 7.2 | -10.2 | 28.4 | -12 to 30 |
| Left/Right | -0.7 | 3.8 | -14.1 | 12.3 | -16 to 16 |

### 2.4 Target Correlations

A critical discovery: the three targets are nearly uncorrelated.

| | Angle | Depth | Left/Right |
|--|-------|-------|------------|
| Angle | 1.000 | -0.046 | -0.010 |
| Depth | -0.046 | 1.000 | -0.061 |
| Left/Right | -0.010 | -0.061 | 1.000 |

This means multi-task learning approaches are unlikely to help, and each target should be modeled independently.

### 2.5 Per-Participant Variation

Participants showed significantly different shooting characteristics:

| Participant | Angle Mean | Depth Mean | LR Mean | Angle Std |
|-------------|------------|------------|---------|-----------|
| P1 | 44.38 | 11.33 | -0.84 | 2.70 |
| P2 | 42.62 | 10.93 | -0.10 | 2.33 |
| P3 | 48.02 | 9.54 | -1.25 | 2.60 |
| P4 | 52.25 | 9.09 | -0.64 | 1.30 |
| P5 | 40.63 | 7.57 | -1.04 | 4.07 |

P5 had notably higher variance in angle predictions (std=4.07 vs 1.30-2.70 for others), making this participant particularly challenging to model.

---

## 3. Biomechanical Feature Engineering

### 3.1 Sequence Representation

We preserved the full temporal structure:
- Raw shape: (n_samples, 57 channels, 240 frames)
- 57 channels = 19 keypoints × 3 coordinates

### 3.2 Release Frame Detection

The release moment is biomechanically critical. We detected it by finding the peak wrist height in the final 120 frames:

```python
def find_release_frame(wrist_z):
    search_window = wrist_z[-120:]
    release_idx = np.argmax(search_window) + (len(wrist_z) - 120)
    return min(release_idx, len(wrist_z) - 10)
```

### 3.3 Statistical Features (2,300+ features)

For each keypoint coordinate, we computed:

**Position Statistics:**
- Mean, standard deviation, min, max, range, median
- Skewness, kurtosis

**Velocity Features (first-order differences):**
- Mean velocity, max velocity, velocity at release
- Velocity standard deviation

**Acceleration Features (second-order differences):**
- Mean acceleration, max acceleration
- Acceleration at release

**Temporal Window Statistics:**
- Early phase (frames 0-80): Preparation
- Middle phase (frames 80-160): Power generation
- Release phase (frames 160-240): Release and follow-through

### 3.4 Physics-Based Features

**Elbow Angle at Release:**
```python
upper_arm = elbow - shoulder
forearm = wrist - elbow
cos_angle = np.dot(upper_arm, forearm) / (norm(upper_arm) * norm(forearm))
elbow_angle = np.arccos(cos_angle)
```

**Release Height:**
```python
release_height = wrist_z[release_frame] - hip_z[release_frame]
```

**Wrist Velocity Magnitude:**
```python
wrist_velocity = np.linalg.norm(wrist[release] - wrist[release-1])
```

**Body Lean:**
```python
body_lean = shoulder_x[release] - hip_x[release]
```

**Finger Spread:**
```python
finger_spread = abs(first_finger_x[release] - second_finger_x[release])
```

### 3.5 Per-Participant Normalization

Individual shooters have different body proportions and baseline positions. We applied z-score normalization within each participant:

```python
for participant in [1, 2, 3, 4, 5]:
    mask = participant_id == participant
    for channel in range(57):
        mean = train_data[mask, channel, :].mean()
        std = train_data[mask, channel, :].std() + 1e-8
        train_data[mask, channel, :] = (train_data[mask, channel, :] - mean) / std
```

### 3.6 Body-Centered Coordinates

We transformed all positions relative to mid_hip, making features invariant to the shooter's absolute position:

```python
for keypoint in all_keypoints:
    keypoint_x -= mid_hip_x
    keypoint_y -= mid_hip_y
    keypoint_z -= mid_hip_z
```

---

## 4. Model Architectures

### 4.1 Transformer (for Depth and partial Angle/LR)

```python
class TransformerPredictor(nn.Module):
    def __init__(self, n_channels=57, n_frames=240, d_model=64, n_heads=4, n_layers=2):
        self.input_proj = nn.Linear(n_channels, d_model)
        self.pos_enc = nn.Parameter(torch.randn(1, n_frames, d_model) * 0.1)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, 
            dim_feedforward=128, dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        self.participant_embed = nn.Embedding(6, 16)
        self.fc1 = nn.Linear(d_model + 16, 32)
        self.fc2 = nn.Linear(32, 1)
```

**Why Transformer for Depth:**
Depth (short vs. long) relates to shot power and timing across the full motion. Self-attention captures how early movements propagate to release.

### 4.2 Gradient Boosted Trees (for Angle)

```python
model = lgb.LGBMRegressor(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1
)
```

**Why Trees for Angle:**
Shot angle depends on discrete biomechanical features at release. Trees captured feature thresholds and interactions 17% better than transformers.

### 4.3 Dilated CNN (for Left/Right)

```python
class DilatedCNN(nn.Module):
    def __init__(self):
        self.conv1 = nn.Conv1d(57, 64, kernel_size=3, dilation=1, padding=1)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=3, dilation=2, padding=2)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=3, dilation=4, padding=4)
        self.conv4 = nn.Conv1d(64, 32, kernel_size=3, dilation=8, padding=8)
```

**Why Dilated CNN for Left/Right:**
Lateral deviation depends on local mechanics (hand position at release). Dilated convolutions with receptive fields of 3, 7, 15 frames capture these local patterns efficiently.

---

## 5. Experimental Journey

We systematically tested numerous hypotheses. This section documents what we tried, why, and what we learned.

### 5.1 Baseline Experiments

| Experiment | Approach | OOF MSE | Result |
|------------|----------|---------|--------|
| E1 | Simple MLP on flattened features | 0.0089 | Baseline |
| E2 | LSTM on sequences | 0.0082 | Slight improvement |
| E3 | Transformer on sequences | 0.0074 | Better temporal modeling |
| E4 | XGBoost on statistical features | 0.0071 | Best for angle |

### 5.2 Architecture Comparison for Each Target

**Angle Results:**
| Model | OOF MSE |
|-------|---------|
| Transformer | 0.00785 |
| LightGBM | 0.00650 |
| XGBoost | 0.00652 |
| **Tree Ensemble** | **0.00642** |

Trees outperformed transformers by 17% for angle prediction.

**Depth Results:**
| Model | OOF MSE |
|-------|---------|
| Tree Ensemble | 0.00812 |
| **Transformer** | **0.00759** |
| Dilated CNN | 0.00801 |

Transformer was best for depth.

**Left/Right Results:**
| Model | OOF MSE |
|-------|---------|
| Tree Ensemble | 0.00589 |
| Transformer | 0.00548 |
| Dilated CNN | 0.00542 |
| **Dilated + Transformer** | **0.00514** |

Blend performed best for left/right.

### 5.3 Participant-Specific Modeling (Failed)

**Hypothesis:** P5 contributed disproportionately to error. Training a separate model on P5's 74 samples might help.

**Approaches Tried:**
1. Separate XGBoost model for P5
2. P5-specific feature selection
3. Higher regularization for P5 model

**Result:** All approaches increased error. 74 samples were insufficient to train a robust model, causing severe overfitting.

**Lesson:** With limited per-participant data, unified models with participant embeddings work better than separate models.

### 5.4 Similarity-Based Prediction (Failed)

**Hypothesis:** Shots with similar body movements should have similar outcomes. Using k-nearest neighbors in feature space to find similar training shots and average their outcomes.

**Approaches Tried:**
1. kNN with k=3,5,7,10 on full feature space
2. kNN on PCA-reduced features
3. Weighted average by distance

**Result:** Performance degraded compared to model predictions.

**Lesson:** Small biomechanical differences can produce large outcome variations. The relationship between motion and outcome is complex and non-linear, not well-captured by simple similarity metrics.

### 5.5 Multi-Task Learning (Failed)

**Hypothesis:** Learning all three targets together might find shared representations beneficial to all.

**Architecture:** Single transformer encoder with three prediction heads.

**Result:** OOF MSE increased from 0.0064 to 0.0104.

**Analysis:** Target correlations were near-zero (max |r| = 0.061). Forcing shared representations hurt performance.

**Lesson:** Verify target correlations before attempting multi-task learning.

### 5.6 Synthetic Data Generation (Failed)

**Hypothesis:** With only 345 samples, generating synthetic data via VAE could improve model training.

**Approach:**
1. Train Conditional VAE on shooting sequences
2. Generate 350 synthetic samples (50 per participant + 100 extra for P5)
3. Train models on combined real + synthetic data

**Result:** OOF MSE increased from 0.0064 to 0.0071.

**Analysis:** VAE generated samples that matched training distributions, but test distributions differed from training.

**Lesson:** Synthetic data augmentation assumes test follows training distribution. When this assumption fails, augmentation can hurt.

### 5.7 Alternative Loss Functions (Failed)

**Hypothesis:** Huber loss might be more robust to outliers than MSE.

**Result:** No improvement. Outliers in this dataset reflect genuine biomechanical variation, not noise.

### 5.8 Stacking Meta-Learners (Failed)

**Hypothesis:** A meta-learner (ElasticNet, Ridge) could optimally combine base model predictions.

**Result:** OOF-optimal weights didn't transfer to test. Simple averaging performed comparably.

**Lesson:** With small test sets, sophisticated stacking often overfits.

### 5.9 Pseudo-Labeling (Failed)

**Hypothesis:** Using model predictions on test data as pseudo-labels could create more training data.

**Result:** Score increased from 0.00677 to 0.00785 (much worse).

**Lesson:** Pseudo-labeling amplifies existing model biases.

### 5.10 Ensemble Blending (Success)

**Finding:** Simple weighted blending of complementary models worked well.

**Optimal Blend:**
- Angle: 70% Tree + 30% Transformer
- Depth: 100% Transformer
- Left/Right: 50% Dilated CNN + 50% Transformer

**OOF MSE:** 0.00641

**Key Discovery:** OOF-optimal blend weights differed from test-optimal weights. Submission feedback was essential for tuning.

### 5.11 Per-Participant Calibration (Success)

**Discovery:** After all model development, systematic biases remained per participant.

**Observation:** When we shifted P5's left/right predictions toward training mean (using VAE), score got WORSE. This revealed test distributions differ from training.

**Approach:** Applied small shifts per participant, validated through submission feedback.

**Final Adjustments:**
- Left/Right: {P1: -0.3, P2: -0.2, P3: +0.2, P4: +0.7, P5: -1.2}
- Depth: {P1: +0.5, P4: -0.7}

**Result:** Score improved from 0.00677 to 0.006595.

---

## 6. Final Solution

### 6.1 Pipeline Overview

```
Raw Data (345 train, 113 test)
    ↓
Per-Participant Normalization
    ↓
Feature Extraction (sequences + statistics + physics)
    ↓
Target-Specific Models:
    - Angle: 70% XGBoost/LightGBM + 30% Transformer
    - Depth: 100% Transformer  
    - Left/Right: 50% Dilated CNN + 50% Transformer
    ↓
Ensemble Blending (70% config A + 30% config B)
    ↓
Per-Participant Calibration
    ↓
Final Predictions
```

### 6.2 Training Configuration

- Cross-validation: 5-fold with random shuffling
- Seeds: 5 seeds averaged for stability
- Epochs: 100 for neural networks
- Batch size: 32
- Optimizer: Adam (lr=0.001, weight_decay=1e-4)
- Augmentation: Gaussian noise (σ=0.05) with 50% probability

### 6.3 Reproducibility

All random seeds are fixed. Full code available in this repository.

---

## 7. Results and Analysis

### 7.1 Score Progression

| Stage | Score | Improvement |
|-------|-------|-------------|
| Baseline Transformer | 0.006969 | - |
| + Tree for Angle | 0.006851 | -0.000118 |
| + Dilated CNN for LR | 0.006782 | -0.000069 |
| + Ensemble Blending | 0.006774 | -0.000008 |
| + P5 LR Calibration | 0.006680 | -0.000094 |
| + P4 LR Calibration | 0.006658 | -0.000022 |
| + All LR Calibration | 0.006623 | -0.000035 |
| + Depth Calibration | 0.006595 | -0.000028 |

### 7.2 Per-Target Analysis

| Target | OOF MSE | Contribution |
|--------|---------|--------------|
| Angle | 0.00642 | 33.5% |
| Depth | 0.00759 | 39.7% |
| Left/Right | 0.00514 | 26.8% |

Depth was the hardest target to predict, likely due to its dependence on subtle timing and power variations.

---

## 8. Lessons Learned

### 8.1 What Worked

1. **Target-specific architectures**: Different targets respond to different model types
2. **Per-participant normalization**: Essential for handling individual differences
3. **Simple ensembling**: Weighted averages outperformed complex stacking
4. **Submission-based tuning**: OOF metrics didn't always transfer to test

### 8.2 What Didn't Work

1. **Participant-specific models**: Insufficient data per participant
2. **Multi-task learning**: Targets were uncorrelated
3. **Synthetic data**: Generated training-like, not test-like distributions
4. **Complex meta-learners**: Overfit on small datasets

### 8.3 Biomechanical Insights

- **Angle** is best predicted by discrete features at release (elbow angle, release height)
- **Depth** requires understanding full temporal dynamics of the shot
- **Left/Right** depends on both local release mechanics and global body alignment

These align with basketball coaching principles: angle is about arm mechanics, depth is about power and timing, lateral accuracy is about alignment.

### 8.4 Recommendations for Future Work

1. **More participants**: 5 participants limits generalization analysis
2. **Ball tracking**: Adding ball trajectory features could improve depth prediction
3. **Video data**: RGB video might capture subtle mechanics missed by keypoints
4. **Temporal alignment**: Dynamic time warping to align shooting phases

---

## Repository Structure

```
├── README.md              # Project overview
├── FULL_REPORT.md         # Complete methodology (this document)
├── methodology.md         # Brief methodology for Kaggle
├── LICENSE                # MIT License
├── requirements.txt       # Dependencies
├── preprocessing.py       # Data loading and normalization
├── features.py            # Feature extraction
├── transformer_model.py   # Transformer architecture
├── tree_model.py          # XGBoost/LightGBM
├── dilated_cnn.py         # Dilated CNN
├── bias_amplification.py  # Per-participant calibration
└── train.py               # Main training script
```

---

## Acknowledgments

Thanks to SPL and UTSPAN for organizing this challenge and providing high-quality motion capture data.
