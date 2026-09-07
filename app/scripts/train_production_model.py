import os
import pandas as pd
import xgboost as xgb
import shap
import matplotlib.pyplot as plt

# --- Paths ---
PROJECT_ROOT = r"D:\rag_types\NER\ner-routing"
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "app", "data", "processed")
REAL_TARGET_DATA = os.path.join(PROCESSED_DIR, "master_panel_real_targets.parquet")
MODEL_FILE = os.path.join(PROCESSED_DIR, "xgboost_production_model.json")
SHAP_OUT = os.path.join(PROCESSED_DIR, "shap_production_summary.png")

def train_and_explain():
    print(f"Loading data from {REAL_TARGET_DATA}...")
    df = pd.read_parquet(REAL_TARGET_DATA)
    
    features = [
        'length_km_calculated', 'slope_mean', 'slope_max', 'roughness_mean', 
        'twi_mean', 'twi_max', 'log_flow_accum_mean', 'log_flow_accum_max',
        'rainfall_24h', 'rainfall_3d', 'rainfall_7d', 'rainfall_anomaly',
        'ndvi', 'mndwi', 's1_vv', 's1_vh', 'vh_vv_ratio', 'pred_label' # <-- ADDED HERE
    ]
    X = df[features].fillna(0.0)
    y = df['target_blockage']
    
    # Balance the 3.5% minority class
    pos_weight = (len(y) - sum(y)) / sum(y) if sum(y) > 0 else 1
    
    print("Training Final Production XGBoost Model on 100% of data...")
    clf = xgb.XGBClassifier(
        n_estimators=100, max_depth=5, learning_rate=0.1, 
        scale_pos_weight=pos_weight, random_state=42, n_jobs=-1
    )
    clf.fit(X, y)
    
    print(f"Saving model artifact to {MODEL_FILE}...")
    clf.save_model(MODEL_FILE)
    
    print("Generating SHAP Explanations...")
    explainer = shap.TreeExplainer(clf)
    
    # Use a 2000-row sample to prevent memory overload during SHAP calculation
    X_sample = X.sample(n=min(2000, len(X)), random_state=42)
    shap_values = explainer.shap_values(X_sample)
    
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.savefig(SHAP_OUT, bbox_inches='tight')
    plt.close()
    
    print(f"SHAP summary plot successfully saved to: {SHAP_OUT}")

if __name__ == "__main__":
    train_and_explain()