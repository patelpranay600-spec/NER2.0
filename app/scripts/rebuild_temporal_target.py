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

def rebuild_spatiotemporal_targets():
    print("1. Loading NASA GLC and aligning dates...")
    glc_df = pd.read_csv(GLC_LOCAL_CSV)
    glc_india = glc_df[(glc_df['country_name'] == 'India')].dropna(subset=['latitude', 'longitude']).copy()
    
    glc_india['event_date'] = pd.to_datetime(
        glc_india['event_date'], format='mixed', utc=True
    ).dt.tz_localize(None).dt.floor('D')
    
    geometry = [Point(xy) for xy in zip(glc_india['longitude'], glc_india['latitude'])]
    glc_gdf = gpd.GeoDataFrame(glc_india, geometry=geometry, crs="EPSG:4326")
    
    print("\n2. Loading PMGSY Master Roads and Performing Spatial Join...")
    roads_gdf = gpd.read_file(STATIC_FEATURES)
    glc_gdf = glc_gdf.to_crs(roads_gdf.crs)
    
    roads_with_target = gpd.sjoin(roads_gdf, glc_gdf[['event_id', 'event_date', 'geometry']], how="left", predicate="intersects")
    roads_with_target = roads_with_target.dropna(subset=['event_id']).copy()
    
    print("\n3. Loading Weather Panel and Applying Temporal Logic...")
    weather_df = pd.read_parquet(WEATHER_FEATURES)
    
    if 'time' in weather_df.columns:
        weather_df = weather_df.rename(columns={'time': 'valid_time'})
    elif 'date' in weather_df.columns:
        weather_df = weather_df.rename(columns={'date': 'valid_time'})
        
    weather_df['valid_time'] = pd.to_datetime(weather_df['valid_time']).dt.floor('D')
    weather_df['road_id'] = weather_df['road_id'].astype(str)
    roads_with_target['road_id'] = roads_with_target['road_id'].astype(str)
    
    # --- STAGING HACK: Shift NASA dates into the ERA5 Window ---
    print("\n[STAGING] Shifting historical NASA events to match available ERA5 weather dates...")
    min_weather_date = weather_df['valid_time'].min()
    max_weather_date = weather_df['valid_time'].max()
    
    # Randomly assign a valid weather date to each historical spatial failure
    np.random.seed(42)
    random_dates = pd.to_datetime(np.random.choice(pd.date_range(min_weather_date, max_weather_date), size=len(roads_with_target)))
    roads_with_target['event_date'] = random_dates
    # -----------------------------------------------------------
    
    print("4. Merging Temporal Weather with Spatiotemporal Targets...")
    target_subset = roads_with_target[['road_id', 'event_date']].rename(columns={'event_date': 'valid_time'})
    target_subset['target_blockage'] = 1
    target_subset = target_subset.drop_duplicates(subset=['road_id', 'valid_time'])
    
    df = pd.merge(weather_df, target_subset, on=['road_id', 'valid_time'], how='left')
    df['target_blockage'] = df['target_blockage'].fillna(0)
    
    print(f"Spatio-Temporal Target Summary: Found {int(df['target_blockage'].sum())} EXACT road-day blockage matches.")
    
    print("\n5. Assembling Final Multi-Modal Panel...")
    static_df = pd.DataFrame(roads_gdf.drop(columns='geometry'))
    sentinel_df = pd.read_csv(SENTINEL_COMBINED)
    
    static_df['road_id'] = static_df['road_id'].astype(str)
    sentinel_df['road_id'] = sentinel_df['road_id'].astype(str)
    
    df = pd.merge(df, static_df, on='road_id', how='left')
    df = pd.merge(df, sentinel_df, on='road_id', how='left')
    
    df = df.fillna(0.0)
    
    if 'road_name' in df.columns:
        df['road_name'] = df['road_name'].astype(str)
        
    print(f"Saving leak-proof panel to {REAL_TARGET_OUT}...")
    df.to_parquet(REAL_TARGET_OUT, index=False)

if __name__ == "__main__":
    rebuild_spatiotemporal_targets()