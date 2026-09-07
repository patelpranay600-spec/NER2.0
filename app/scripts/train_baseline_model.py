import os
import geopandas as gpd
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

# --- Paths ---
PROJECT_ROOT = r"D:\rag_types\NER\ner-routing"
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "app", "data", "processed")
FEATURES_PATH = os.path.join(PROCESSED_DIR, "master_roads_final_features.gpkg")

def build_baseline_model():
    print(f"Loading final feature matrix from {FEATURES_PATH}...")
    roads = gpd.read_file(FEATURES_PATH)
    
    # 1. Select the extracted features (Phase 1 & Phase 2)
    feature_cols = [
        'length_km_calculated', 
        'slope_mean', 'slope_max', 
        'roughness_mean', 
        'twi_mean', 'twi_max', 
        'log_flow_accum_mean', 'log_flow_accum_max'
    ]
    
    # Check which features actually exist in the GeoPackage
    available_features = [col for col in feature_cols if col in roads.columns]
    
    df = roads[available_features].copy()
    
    # 2. Handle NaNs (Resolves the log1p RuntimeWarning)
    print("Cleaning missing values and NaNs...")
    df = df.fillna(0) # Fill NaNs with 0 for the baseline
    
    # 3. Create a Synthetic Target Variable (Until Phase 5 labels are integrated)
    # We will simulate 'blockage_risk' based on high slope and flow accumulation to test the pipeline
    print("Generating synthetic target for pipeline testing (Phase 5 placeholder)...")
    if 'slope_max' in df.columns and 'log_flow_accum_max' in df.columns:
        risk_condition = (df['slope_max'] > 30) | (df['log_flow_accum_max'] > 5)
        df['target_blockage'] = np.where(risk_condition, 1, 0)
    else:
        # Fallback random target if features are missing
        df['target_blockage'] = np.random.randint(0, 2, size=len(df))
        
    X = df[available_features]
    y = df['target_blockage']
    
    # 4. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 5. XGBoost Baseline Model
    print("\nTraining XGBoost Baseline Model...")
    clf = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        eval_metric='logloss',
        random_state=42
    )
    
    clf.fit(X_train, y_train)
    
    # 6. Evaluation
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]
    
    print("\n--- Baseline Model Performance ---")
    print(classification_report(y_test, y_pred))
    print(f"ROC-AUC Score: {roc_auc_score(y_test, y_proba):.4f}")
    
    # 7. Feature Importance
    print("\n--- Feature Importances ---")
    importances = pd.Series(clf.feature_importances_, index=available_features).sort_values(ascending=False)
    for feat, imp in importances.items():
        print(f"{feat}: {imp:.4f}")

if __name__ == "__main__":
    build_baseline_model()