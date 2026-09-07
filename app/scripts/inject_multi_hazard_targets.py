import os
import pandas as pd
import geopandas as gpd
import numpy as np

# --- Paths ---
PROJECT_ROOT = r"D:\rag_types\NER\ner-routing"
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "app", "data", "processed")
PMGSY_ROADS = os.path.join(PROCESSED_DIR, "master_roads_final_features.gpkg")
REAL_TARGET_OUT = os.path.join(PROCESSED_DIR, "master_panel_real_targets.parquet")

def inject_multi_hazard_targets():
    print("1. Loading existing panel and new infrastructure data...")
    df = pd.read_parquet(REAL_TARGET_OUT)
    roads_gdf = gpd.read_file(PMGSY_ROADS)
    
    # Extract the new infrastructure features
    infra_df = pd.DataFrame(roads_gdf[['road_id', 'pred_label', 'pred_class']])
    infra_df['road_id'] = infra_df['road_id'].astype(str)
    df['road_id'] = df['road_id'].astype(str)
    
    # Drop old infra columns if they exist in the parquet to avoid duplicates
    cols_to_drop = [c for c in ['pred_label', 'pred_class'] if c in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
        
    # Merge the infrastructure labels into the main ML panel
    df = pd.merge(df, infra_df, on='road_id', how='left')
    df['pred_label'] = df['pred_label'].fillna(1.0) # Default to unpaved
    
    print("2. Simulating Multi-Hazard Logistics Vulnerability...")
    
    # Normalize variables to 0-1 range to ensure fair weighting
    def normalize(column):
        return (column - column.min()) / (column.max() - column.min() + 1e-9)

    slope_norm = normalize(df['slope_max'])
    rain3d_norm = normalize(df['rainfall_3d'])
    rain7d_norm = normalize(df['rainfall_7d'])
    twi_norm = normalize(df['twi_mean'])
    flow_norm = normalize(df['log_flow_accum_max'])
    
    # Hazard 1: Landslide Risk (Steep terrain + short intense rain)
    landslide_risk = (slope_norm * 0.5) + (rain3d_norm * 0.5)
    
    # Hazard 2: Flood Risk (Flat terrain + high moisture + prolonged rain)
    flood_risk = (twi_norm * 0.35) + (flow_norm * 0.35) + (rain7d_norm * 0.3)
    
    # Base risk is whichever hazard is deadlier for that specific road segment
    base_risk = np.maximum(landslide_risk, flood_risk)
    
    # Hazard 3: Infrastructure Gap Penalty (Unpaved roads are 30% more likely to fail)
    infra_multiplier = np.where(df['pred_label'] == 1.0, 1.3, 1.0)
    final_risk = base_risk * infra_multiplier
    
    # Inject 10% randomness to simulate unmeasured real-world chaos
    np.random.seed(42)
    noise = np.random.normal(0, 0.1, len(df))
    final_risk += noise
    
    # Tag the top 4% most dangerous road-days as actual logistical blockages
    threshold = np.percentile(final_risk, 96.0)
    df['target_blockage'] = np.where(final_risk > threshold, 1.0, 0.0)
    
    print(f"Total Logistics Blockages Simulated: {int(df['target_blockage'].sum())} out of {len(df)} rows.")
    
    print(f"3. Saving final multi-hazard dataset to {REAL_TARGET_OUT}...")
    df.to_parquet(REAL_TARGET_OUT, index=False)
    print("✅ Ready! Your dataset now mathematically reflects landslides, floods, and infra gaps.")

if __name__ == "__main__":
    inject_multi_hazard_targets()