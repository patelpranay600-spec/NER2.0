import os
import glob
import pandas as pd
import geopandas as gpd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import shap
import matplotlib.pyplot as plt

# --- Paths ---
PROJECT_ROOT = r"D:\rag_types\NER\ner-routing"
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "app", "data", "processed")

# Ensure this points to the exact folder containing your CSV chunks
SENTINEL_DIR = os.path.join(PROJECT_ROOT, "sentinel data") 
STATIC_FEATURES = os.path.join(PROCESSED_DIR, "master_roads_final_features.gpkg")
WEATHER_FEATURES = os.path.join(PROCESSED_DIR, "weather", "temporal_roads_era5.parquet")
COMBINED_SENTINEL_OUT = os.path.join(PROCESSED_DIR, "weather", "sentinel_features_combined.csv")

def build_final_model():
    # 1. Combine Sentinel Chunks
    chunk_files = glob.glob(os.path.join(SENTINEL_DIR, "*.csv"))
    if not chunk_files:
        print(f"Error: No CSV files found in {SENTINEL_DIR}. Please check the path.")
        return
        
    print(f"Combining {len(chunk_files)} Sentinel CSV chunks...")
    df_sentinel = pd.concat([pd.read_csv(f) for f in chunk_files], ignore_index=True)
    
    # Clean up Earth Engine system columns
    cols_to_drop = [c for c in ['system:index', '.geo'] if c in df_sentinel.columns]
    df_sentinel = df_sentinel.drop(columns=cols_to_drop)
    
    # FAIL-SAFE: Ensure all expected Sentinel columns exist even if GEE masked them out
    expected_sentinel_cols = ['ndvi', 'mndwi', 's1_vv', 's1_vh', 'vh_vv_ratio']
    for col in expected_sentinel_cols:
        if col not in df_sentinel.columns:
            print(f"Notice: '{col}' missing from GEE export. Auto-filling with 0.0.")
            df_sentinel[col] = 0.0
            
    # Save the combined Sentinel dataset into the processed folder
    print(f"Saving combined Sentinel dataset to {COMBINED_SENTINEL_OUT}...")
    df_sentinel.to_csv(COMBINED_SENTINEL_OUT, index=False)
    
    # 2. Load and Merge Existing Data
    print("Loading static terrain/hydrology and temporal weather features...")
    static_df = pd.DataFrame(gpd.read_file(STATIC_FEATURES).drop(columns='geometry'))
    weather_df = pd.read_parquet(WEATHER_FEATURES)
    
    print("Merging Phase 1-4 into the master multi-modal panel...")
    # Merge Weather + Static
    df = pd.merge(weather_df, static_df, on='road_id', how='left')
    
    # Ensure road_id formats match before merging Sentinel
    df['road_id'] = df['road_id'].astype(str)
    df_sentinel['road_id'] = df_sentinel['road_id'].astype(str)
    
    # Merge + Sentinel
    df = pd.merge(df, df_sentinel, on='road_id', how='left')
    
    # Fill missing values from potential extraction artifacts
    df = df.fillna(0.0)
    
    # 3. Feature Selection
    features = [
        'length_km_calculated', 'slope_mean', 'slope_max', 'roughness_mean', 
        'twi_mean', 'twi_max', 'log_flow_accum_mean', 'log_flow_accum_max',
        'rainfall_24h', 'rainfall_3d', 'rainfall_7d', 'rainfall_anomaly',
        'ndvi', 'mndwi', 's1_vv', 's1_vh', 'vh_vv_ratio'
    ]
    
    available_features = [col for col in features if col in df.columns]
    
    print("Generating noisy synthetic target to simulate real-world uncertainty...")
    # Create a base risk score where higher slopes and more rain increase probability
    base_risk = (df['slope_max'] / df['slope_max'].max() * 0.4) + \
                (df['rainfall_3d'] / df['rainfall_3d'].max() * 0.6)
    
    # Inject random noise to simulate unmeasured variables (soil type, road age)
    np.random.seed(42)
    noise = np.random.normal(0, 0.15, len(df)) 
    final_risk = base_risk + noise
    
    # Flag the top 10% most vulnerable road-days as actual blockages
    threshold = np.percentile(final_risk, 90)
    df['target_blockage'] = np.where(final_risk > threshold, 1, 0)
    
    X = df[available_features]
    y = df['target_blockage']
    
    print(f"\nDataset Shape: {X.shape[0]} rows (Road-Days) x {X.shape[1]} features")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 4. Train Model
    print("\nTraining Final Multi-Modal XGBoost Model...")
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
    
    print("\n--- Final Model Performance ---")
    print(classification_report(y_test, y_pred))
    print(f"ROC-AUC Score: {roc_auc_score(y_test, y_proba):.4f}")
    
    # 5. SHAP Explainability
    print("\nGenerating SHAP explanations for dynamic routing logic...")
    explainer = shap.TreeExplainer(clf)
    X_test_sample = X_test.sample(n=min(1000, len(X_test)), random_state=42)
    shap_values = explainer.shap_values(X_test_sample)
    
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test_sample, show=False)
    shap_out = os.path.join(PROCESSED_DIR, "shap_summary.png")
    plt.savefig(shap_out, bbox_inches='tight')
    plt.close()
    
    print(f"SHAP summary plot successfully saved to: {shap_out}")

if __name__ == "__main__":
    build_final_model()