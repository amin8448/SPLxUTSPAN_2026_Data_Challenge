import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import KFold


def train_lightgbm(X_train, X_test, y, n_folds=5, n_seeds=5):
    oof = np.zeros(len(y))
    test_preds = []
    
    for seed in range(n_seeds):
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42 + seed)
        
        for train_idx, val_idx in kf.split(X_train):
            model = lgb.LGBMRegressor(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=0.1,
                random_state=42 + seed,
                n_jobs=-1,
                verbose=-1
            )
            model.fit(X_train.iloc[train_idx], y[train_idx])
            oof[val_idx] += model.predict(X_train.iloc[val_idx])
        
        model = lgb.LGBMRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=42 + seed,
            n_jobs=-1,
            verbose=-1
        )
        model.fit(X_train, y)
        test_preds.append(model.predict(X_test))
    
    return oof / n_seeds, np.mean(test_preds, axis=0)


def train_xgboost(X_train, X_test, y, n_folds=5, n_seeds=5):
    oof = np.zeros(len(y))
    test_preds = []
    
    for seed in range(n_seeds):
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42 + seed)
        
        for train_idx, val_idx in kf.split(X_train):
            model = xgb.XGBRegressor(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=0.1,
                random_state=42 + seed,
                n_jobs=-1
            )
            model.fit(X_train.iloc[train_idx], y[train_idx])
            oof[val_idx] += model.predict(X_train.iloc[val_idx])
        
        model = xgb.XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=42 + seed,
            n_jobs=-1
        )
        model.fit(X_train, y)
        test_preds.append(model.predict(X_test))
    
    return oof / n_seeds, np.mean(test_preds, axis=0)


def train_tree_ensemble(X_train, X_test, y, n_folds=5, n_seeds=3):
    lgb_oof, lgb_test = train_lightgbm(X_train, X_test, y, n_folds, n_seeds)
    xgb_oof, xgb_test = train_xgboost(X_train, X_test, y, n_folds, n_seeds)
    
    oof = 0.5 * lgb_oof + 0.5 * xgb_oof
    test_pred = 0.5 * lgb_test + 0.5 * xgb_test
    
    return oof, test_pred
