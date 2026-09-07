import os
import geopandas as gpd
import pandas as pd
import numpy as np

# --- Paths ---
PROJECT_ROOT = r"D:\rag_types\NER\ner-routing"
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "app", "data", "processed")
FEATURES_PATH = os.path.join(PROCESSED_DIR, "master_roads_final_features.gpkg")
WEATHER_OUT = os.path.join(PROCESSED_DIR, "weather", "temporal_roads_rainfall.parquet")

def build_temporal_pipeline():
    print(f"Loading static road features from {FEATURES_PATH}...")
    roads = gpd.read_file(FEATURES_PATH)
    
    # 1. Create a 30-day date range for the simulation
    dates = pd.date_range(end=pd.Timestamp.today(), periods=30, freq='D')
    
    # 2. Expand static roads into a temporal panel (Cross Merge)
    print("Expanding static roads into a 30-day temporal panel...")
    roads_df = pd.DataFrame(roads.drop(columns='geometry'))
    dates_df = pd.DataFrame({'date': dates})
    
    # Create a synthetic key to perform a cross join
    roads_df['key'] = 1
    dates_df['key'] = 1
    panel = pd.merge(roads_df, dates_df, on='key').drop('key', axis=1)
    
    # 3. Simulate raw daily rainfall (mm) using a Gamma distribution (common for precipitation)
    print("Generating simulated daily precipitation...")
    np.random.seed(42)
    # Most days 0 rain, some days heavy rain
    panel['rainfall_24h'] = np.where(
        np.random.rand(len(panel)) > 0.7, 
        np.random.gamma(shape=2.0, scale=10.0, size=len(panel)), 
        0.0
    )
    
    # 4. Calculate Rolling Features (Phase 3 Spec)
    print("Calculating rolling window features (3d, 7d, anomaly)...")
    
    # Sort by road ID and date to ensure correct rolling calculations
    panel = panel.sort_values(by=['road_id', 'date']).reset_index(drop=True)
    
    # Group by road_id and apply rolling sums
    grouped = panel.groupby('road_id')['rainfall_24h']
    
    panel['rainfall_3d'] = grouped.rolling(window=3, min_periods=1).sum().reset_index(level=0, drop=True)
    panel['rainfall_7d'] = grouped.rolling(window=7, min_periods=1).sum().reset_index(level=0, drop=True)
    
    # Calculate anomaly: 24h rain minus the 30-day historical mean for that specific road
    panel['rainfall_30d_mean'] = grouped.transform('mean')
    panel['rainfall_anomaly'] = panel['rainfall_24h'] - panel['rainfall_30d_mean']
    
    # Drop the temporary mean column
    panel = panel.drop(columns=['rainfall_30d_mean'])
    
    # 5. Save the temporal dataset
    print(f"Saving temporal feature matrix to {WEATHER_OUT}...")
    panel.to_parquet(WEATHER_OUT, index=False)
    
    print("\nPhase 3 feature engineering complete. Sample of new temporal features:")
    print(panel[['road_id', 'date', 'rainfall_24h', 'rainfall_3d', 'rainfall_7d', 'rainfall_anomaly']].head(10))

if __name__ == "__main__":
    build_temporal_pipeline()