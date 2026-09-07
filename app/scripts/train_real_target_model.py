import os
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import shap
import matplotlib.pyplot as plt

# --- Paths ---
PROJECT_ROOT = r"D:\rag_types\NER\ner-routing"
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "app", "data", "processed")
REAL_TARGET_DATA = os.path.join(PROCESSED_DIR, "master_panel_real_targets.parquet")

def train_real_target_model():
    print(f"Loading real-target dataset from {REAL_TARGET_DATA}...")
    df = pd.read_parquet(REAL_TARGET_DATA)
    
    # Feature Selection
    features = [
        'length_km_calculated', 'slope_mean', 'slope_max', 'roughness_mean', 
        'twi_mean', 'twi_max', 'log_flow_accum_mean', 'log_flow_accum_max',
        'rainfall_24h', 'rainfall_3d', 'rainfall_7d', 'rainfall_anomaly',
        'ndvi', 'mndwi', 's1_vv', 's1_vh', 'vh_vv_ratio'
    ]
    
    available_features = [col for col in features if col in df.columns]
    
    X = df[available_features]
    y = df['target_blockage']
    
    print(f"\nDataset Shape: {X.shape[0]} rows (Road-Days) x {X.shape[1]} features")
    print(f"Total Historical Blockage Events in Dataset: {int(y.sum())}")
    
    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Calculate class imbalance ratio dynamically
    pos_weight = (len(y_train) - sum(y_train)) / sum(y_train) if sum(y_train) > 0 else 1
    
    # Train Model
    print("\nTraining Multi-Modal XGBoost Model on REAL Historical Targets...")
    clf = xgb.XGBClassifier(
        n_estimators=100, 
        max_depth=5, 
        learning_rate=0.1, 
        eval_metric='logloss',
        scale_pos_weight=pos_weight,  # Balances precision and recall for rare disaster events
        random_state=42, 
        n_jobs=-1
    )
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]
    
    print("\n--- Real Target Model Performance ---")
    print(classification_report(y_test, y_pred))
    print(f"ROC-AUC Score: {roc_auc_score(y_test, y_proba):.4f}")
    
    # SHAP Explainability
    print("\nGenerating SHAP explanations...")
    explainer = shap.TreeExplainer(clf)
    
    X_test_sample = X_test.sample(n=min(1000, len(X_test)), random_state=42)
    shap_values = explainer.shap_values(X_test_sample)
    
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test_sample, show=False)
    shap_out = os.path.join(PROCESSED_DIR, "shap_real_target_summary.png")
    plt.savefig(shap_out, bbox_inches='tight')
    plt.close()
    
    print(f"SHAP summary plot successfully saved to: {shap_out}")
    
    # Save the model artifact for the NetworkX routing engine
    model_out = os.path.join(PROCESSED_DIR, "xgboost_real_target_model.json")
    clf.save_model(model_out)
    print(f"Trained model saved to: {model_out}")

if __name__ == "__main__":
    train_real_target_model()