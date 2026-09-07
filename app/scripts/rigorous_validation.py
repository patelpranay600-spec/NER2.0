import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score, precision_recall_curve
import matplotlib.pyplot as plt

# --- Paths ---
PROJECT_ROOT = r"D:\rag_types\NER\ner-routing"
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "app", "data", "processed")
REAL_TARGET_DATA = os.path.join(PROCESSED_DIR, "master_panel_real_targets.parquet")

def train_and_evaluate(X_train, X_test, y_train, y_test, split_name):
    print(f"\n{'='*50}")
    print(f"EXPERIMENT: {split_name.upper()}")
    print(f"{'='*50}")
    
    pos_weight = (len(y_train) - sum(y_train)) / sum(y_train) if sum(y_train) > 0 else 1
    
    clf = xgb.XGBClassifier(
        n_estimators=100, 
        max_depth=5, 
        learning_rate=0.1, 
        scale_pos_weight=pos_weight,
        random_state=42, 
        n_jobs=-1
    )
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]
    
    # Core Metrics
    roc_auc = roc_auc_score(y_test, y_proba)
    pr_auc = average_precision_score(y_test, y_proba)
    
    print(classification_report(y_test, y_pred))
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"PR-AUC:  {pr_auc:.4f}")
    
    # Threshold Optimization Analysis
    print("\n--- Operational Threshold Analysis ---")
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)
    
    # Sample a few key thresholds for the operational matrix
    eval_thresholds = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]
    print(f"{'Threshold':<12} | {'Precision':<12} | {'Recall':<12} | {'F1-Score'}")
    print("-" * 55)
    
    for t in eval_thresholds:
        # Find the closest threshold in the precision_recall_curve arrays
        idx = np.abs(thresholds - t).argmin()
        p = precisions[idx]
        r = recalls[idx]
        f1 = 2 * (p * r) / (p + r) if (p + r) > 0 else 0
        print(f"{t:<12.2f} | {p:<12.4f} | {r:<12.4f} | {f1:.4f}")
        
    return clf

def run_validations():
    print(f"Loading real-target dataset from {REAL_TARGET_DATA}...")
    df = pd.read_parquet(REAL_TARGET_DATA)
    
    # Ensure date is parsed correctly for temporal splits
    if 'valid_time' in df.columns:
        df['valid_time'] = pd.to_datetime(df['valid_time'])
        
    features = [
        'length_km_calculated', 'slope_mean', 'slope_max', 'roughness_mean', 
        'twi_mean', 'twi_max', 'log_flow_accum_mean', 'log_flow_accum_max',
        'rainfall_24h', 'rainfall_3d', 'rainfall_7d', 'rainfall_anomaly',
        'ndvi', 'mndwi', 's1_vv', 's1_vh', 'vh_vv_ratio'
    ]
    
    # Fill any lingering NaNs
    df[features] = df[features].fillna(0.0)
    
    X = df[features]
    y = df['target_blockage']
    
    # ---------------------------------------------------------
    # EXPERIMENT 2: Grouped Spatial Split (Unseen Roads)
    # ---------------------------------------------------------
    gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups=df["road_id"]))
    
    train_and_evaluate(
        X.iloc[train_idx], X.iloc[test_idx], 
        y.iloc[train_idx], y.iloc[test_idx], 
        split_name="Grouped Road-ID Split (Generalizing to Unseen Roads)"
    )
    
    # ---------------------------------------------------------
    # EXPERIMENT 3: Temporal Forecasting Split (Unseen Future)
    # ---------------------------------------------------------
    if 'valid_time' in df.columns:
        # Define a cutoff: last 20% of the timeline acts as the test set
        cutoff_date = df['valid_time'].quantile(0.80)
        
        train_mask = df['valid_time'] < cutoff_date
        test_mask = df['valid_time'] >= cutoff_date
        
        train_and_evaluate(
            X[train_mask], X[test_mask], 
            y[train_mask], y[test_mask], 
            split_name=f"Temporal Forecasting (Test Data >= {cutoff_date.strftime('%Y-%m-%d')})"
        )
    else:
        print("\nSkipping Temporal Split: 'valid_time' column not found in dataset.")

if __name__ == "__main__":
    run_validations()