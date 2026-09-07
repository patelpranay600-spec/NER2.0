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
STATIC_FEATURES = os.path.join(PROCESSED_DIR, "master_roads_final_features.gpkg")
WEATHER_FEATURES = os.path.join(PROCESSED_DIR, "weather", "temporal_roads_era5.parquet")

def build_temporal_baseline():
    print("Loading static terrain and hydrology features...")
    static_df = gpd.read_file(STATIC_FEATURES)
    # Drop geometry to make it a standard pandas DataFrame for ML
    static_df = pd.DataFrame(static_df.drop(columns='geometry'))
    
    print("Loading temporal weather features...")
    weather_df = pd.read_parquet(WEATHER_FEATURES)
    
    print("Merging static and temporal datasets...")
    # This creates a panel dataset: every road gets its static features copied across all dates
    df = pd.merge(weather_df, static_df, on='road_id', how='left')
    
    # Fill any NaNs from the static extraction (like the log1p warnings earlier)
    df = df.fillna(0)
    
    # Select our features from Phases 1, 2, and 3
    features = [
        'length_km_calculated', 'slope_mean', 'slope_max', 'roughness_mean', 
        'twi_mean', 'twi_max', 'log_flow_accum_mean', 'log_flow_accum_max',
        'rainfall_24h', 'rainfall_3d', 'rainfall_7d', 'rainfall_anomaly'
    ]
    
    available_features = [col for col in features if col in df.columns]
    
    print("Generating synthetic target (high slope + heavy recent rain)...")
    # Simulate a trigger: Vulnerable terrain + acute weather event
    risk_condition = (df['slope_max'] > 25) & (df['rainfall_3d'] > df['rainfall_3d'].quantile(0.80))
    df['target_blockage'] = np.where(risk_condition, 1, 0)
    
    X = df[available_features]
    y = df['target_blockage']
    
    print(f"\nDataset Shape: {X.shape[0]} rows (Road-Days) x {X.shape[1]} features")
    print(f"Positive Class (Blockages): {y.sum()} instances")
    
    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("\nTraining XGBoost Temporal Baseline...")
    clf = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        eval_metric='logloss',
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]
    
    print("\n--- Temporal Baseline Performance ---")
    print(classification_report(y_test, y_pred))
    print(f"ROC-AUC Score: {roc_auc_score(y_test, y_proba):.4f}")
    
    print("\n--- Feature Importances ---")
    importances = pd.Series(clf.feature_importances_, index=available_features).sort_values(ascending=False)
    for feat, imp in importances.items():
        print(f"{feat}: {imp:.4f}")

if __name__ == "__main__":
    build_temporal_baseline()