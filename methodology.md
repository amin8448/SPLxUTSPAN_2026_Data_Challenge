# Multi-Modal Learning for Basketball Free Throw Prediction: Biomechanical Feature Engineering and Target-Specific Modeling

**GitHub Repository:** https://github.com/amin8448/SPLxUTSPAN_2026_Data_Challenge

---

## 1. Problem Overview

This challenge tasks us with predicting basketball free throw landing outcomes (Angle, Depth, Left/Right) from markerless motion capture data of 19 body keypoints across 240 frames. The goal is to understand how a shooter's body movements during the shooting motion influence where the ball lands on the hoop.

## 2. Biomechanical Feature Engineering

### 2.1 Temporal Sequence Representation
Free throw shooting is a coordinated movement spanning approximately 240 frames. We preserved the full temporal structure by extracting sequences of shape (57, 240) representing 19 keypoints x 3 coordinates over time, allowing models to learn the dynamics of the shooting motion.

### 2.2 Release Phase Analysis
Basketball shooting research emphasizes the release moment as critical for shot outcome. We identified the release frame by detecting the peak wrist height in the final 120 frames, then extracted features specifically around this biomechanically significant event:
- Position, velocity, and acceleration at release
- Pre-release mechanics (30 frames before)
- Follow-through patterns (15 frames after)

### 2.3 Physics-Based Biomechanical Features
We computed features grounded in shooting mechanics research:
- **Elbow angle at release**: Computed from shoulder-elbow-wrist vectors, reflecting arm extension
- **Release height**: Wrist position relative to hip, indicating shot arc potential
- **Wrist velocity magnitude**: 3D velocity at release, related to shot power
- **Body lean**: Shoulder position relative to hip, capturing balance and alignment
- **Finger spread**: Distance between first and second finger at release, relating to ball control

### 2.4 Temporal Window Statistics
Shooting motion progresses through phases. We computed statistics (mean, std, range, velocity, acceleration) across:
- Early phase (frames 0-80): Preparation and initial movement
- Middle phase (frames 80-160): Upward motion and power generation
- Release phase (frames 160-240): Release and follow-through

### 2.5 Per-Participant Normalization
Individual shooters have different body proportions, shooting styles, and baseline positions. Per-participant z-score normalization allowed models to focus on relative motion patterns rather than absolute positions, effectively controlling for individual biomechanical differences.

## 3. Target-Specific Model Selection

A key finding was that each shot outcome responds to different aspects of the shooting motion:

**Angle Prediction - Tree-Based Models (70%) + Transformer (30%)**
Shot angle showed strong relationships with discrete biomechanical features like elbow extension, release height, and wrist position. Gradient boosted trees (XGBoost/LightGBM) outperformed deep learning by 17% for angle, suggesting that specific feature thresholds and interactions (e.g., elbow angle combined with release height) are more predictive than continuous temporal patterns.

**Depth Prediction - Transformer (100%)**
Depth (short vs. long) relates to shot power and timing, which emerge from the full temporal sequence of the shooting motion. The transformer's self-attention mechanism captures how early-phase movements propagate through to release, learning which temporal relationships predict depth outcomes.

**Left/Right Prediction - Dilated CNN (50%) + Transformer (50%)**
Lateral deviation depends on both local mechanics (hand position, wrist rotation at release) and global body alignment. Dilated convolutions with receptive fields of 3, 7, and 15 frames capture local temporal patterns, while the transformer provides global context. The blend leverages both perspectives.

## 4. Ensemble Strategy

Final model: 70% tree + 30% transformer for angle, pure transformer for depth, and 50/50 dilated CNN + transformer for left/right. Blend weights were validated through cross-validation with participant-aware splits.

## 5. Per-Participant Calibration

After developing our core models, we observed that prediction biases varied systematically by participant, reflecting individual shooting tendencies not fully captured by normalization alone. We applied small per-participant adjustments to account for these individual differences, improving generalization to held-out shots.

## 6. Results and Insights

Our approach achieved a final score of 0.006595. Key biomechanical insights:
- Shot angle is best predicted by discrete features at release (elbow angle, release height)
- Shot depth requires understanding the full temporal dynamics of the shooting motion
- Left/right deviation depends on both local release mechanics and global body alignment
- Individual shooter differences require explicit modeling through normalization and calibration

These findings align with basketball coaching principles: angle is about arm mechanics, depth is about power and timing, and lateral accuracy is about alignment and hand position.

---
**Word Count: ~720**
