# Basketball Free Throw Prediction

SPLxUTSPAN 2026 Data Challenge

**Author:** Amin Nabavi, Carleton University  
**Final Score:** 0.006595 (2nd Place)

---

## Overview

Predicting basketball free throw landing outcomes (Angle, Depth, Left/Right) from markerless motion capture data. This repository contains the complete solution including feature engineering, model training, and per-participant calibration.

For full methodology and experimental details, see [FULL_REPORT.md](FULL_REPORT.md).

## Key Findings

1. **Target-specific architectures matter:** Trees for angle, transformers for depth, hybrid CNN+transformer for left/right
2. **Per-participant normalization is essential:** Controls for individual biomechanical differences
3. **Simple ensembling beats complexity:** Weighted averages outperformed stacking meta-learners
4. **OOF metrics can mislead:** Submission feedback was necessary for final tuning

## Model Architecture

| Target | Model | Rationale |
|--------|-------|-----------|
| Angle | XGBoost (70%) + Transformer (30%) | Discrete features at release |
| Depth | Transformer (100%) | Full temporal dynamics |
| Left/Right | Dilated CNN (50%) + Transformer (50%) | Local + global features |

## Repository Structure

```
├── FULL_REPORT.md         # Complete methodology and experiments
├── preprocessing.py       # Data loading and normalization
├── features.py            # Feature extraction
├── transformer_model.py   # Transformer architecture
├── tree_model.py          # XGBoost/LightGBM
├── dilated_cnn.py         # Dilated CNN
├── bias_amplification.py  # Per-participant calibration
├── train.py               # Main training script
└── requirements.txt       # Dependencies
```

## Usage

```bash
pip install -r requirements.txt
python train.py
python bias_amplification.py
```

## Requirements

- Python 3.8+
- PyTorch 1.9+
- scikit-learn, pandas, numpy
- xgboost, lightgbm
