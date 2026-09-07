import os
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import numpy as np

# --- Paths ---
PROJECT_ROOT = r"D:\rag_types\NER\ner-routing"
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "app", "data", "processed")
STATIC_FEATURES = os.path.join(PROCESSED_DIR, "master_roads_final_features.gpkg")
WEATHER_FEATURES = os.path.join(PROCESSED_DIR, "weather", "temporal_roads_era5.parquet")
SENTINEL_COMBINED = os.path.join(PROCESSED_DIR, "weather", "sentinel_features_combined.csv")
REAL_TARGET_OUT = os.path.join(PROCESSED_DIR, "master_panel_real_targets.parquet")
GLC_LOCAL_CSV = os.path.join(PROCESSED_DIR, "NASA_GLC.csv") 

def fetch_and_merge_real_targets():
    print(f"1. Loading local NASA Global Landslide Catalog from {GLC_LOCAL_CSV}...")
    
    if not os.path.exists(GLC_LOCAL_CSV):
        print(f"Error: Could not find {GLC_LOCAL_CSV}. Please download it and place it in the processed folder.")
        return
        
    glc_df = pd.read_csv(GLC_LOCAL_CSV)
    print(f"Loaded {len(glc_df)} global historical landslide records.")
    
    # Filter for India and drop rows missing coordinates
    glc_india = glc_df[(glc_df['country_name'] == 'India')].dropna(subset=['latitude', 'longitude']).copy()
    print(f"Filtered down to {len(glc_india)} historical events in India.")
    
    # Convert to GeoDataFrame
    geometry = [Point(xy) for xy in zip(glc_india['longitude'], glc_india['latitude'])]
    glc_gdf = gpd.GeoDataFrame(glc_india, geometry=geometry, crs="EPSG:4326")
    
    # Reproject to metric CRS to apply a 2km impact buffer around each landslide point
    glc_gdf = glc_gdf.to_crs("EPSG:32646")
    glc_gdf['geometry'] = glc_gdf.buffer(2000)
    
    print("\n2. Loading PMGSY Master Roads...")
    roads_gdf = gpd.read_file(STATIC_FEATURES)
    
    # FIX 1: Ensure exact CRS match before spatial join
    glc_gdf = glc_gdf.to_crs(roads_gdf.crs)
    
    # Spatial Join: Tag roads that fall within 2km of a historical NASA landslide
    print("Performing spatial intersection between roads and historical landslides...")
    roads_with_target = gpd.sjoin(roads_gdf, glc_gdf[['event_id', 'event_date', 'geometry']], how="left", predicate="intersects")
    
    # Create the binary target: 1 if intersected a real event, 0 otherwise
    roads_with_target['target_blockage'] = np.where(roads_with_target['event_id'].notna(), 1, 0)
    
    # Group back to unique roads
    target_mapping = roads_with_target.groupby('road_id')['target_blockage'].max().reset_index()
    
    print(f"Real Target Summary: Found {target_mapping['target_blockage'].sum()} vulnerable road segments out of {len(target_mapping)} total.")
    
    print("\n3. Assembling Final Multi-Modal Panel with Real Targets...")
    static_df = pd.DataFrame(roads_gdf.drop(columns='geometry'))
    weather_df = pd.read_parquet(WEATHER_FEATURES)
    sentinel_df = pd.read_csv(SENTINEL_COMBINED)
    
    weather_df['road_id'] = weather_df['road_id'].astype(str)
    static_df['road_id'] = static_df['road_id'].astype(str)
    sentinel_df['road_id'] = sentinel_df['road_id'].astype(str)
    target_mapping['road_id'] = target_mapping['road_id'].astype(str)
    
    df = pd.merge(weather_df, static_df, on='road_id', how='left')
    df = pd.merge(df, sentinel_df, on='road_id', how='left')
    df = pd.merge(df, target_mapping, on='road_id', how='left')
    
    # Fill numerical NaNs with 0.0
    df = df.fillna(0.0)
    
    # FIX 2: Sanitize string columns to prevent PyArrow Mixed-Type errors
    if 'road_name' in df.columns:
        df['road_name'] = df['road_name'].astype(str)
    
    print(f"Saving final real-target panel to {REAL_TARGET_OUT}...")
    df.to_parquet(REAL_TARGET_OUT, index=False)
    print("Done! You can now run your XGBoost script using this parquet file.")

if __name__ == "__main__":
    fetch_and_merge_real_targets()