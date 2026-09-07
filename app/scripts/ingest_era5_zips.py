import os
import glob
import zipfile
import geopandas as gpd
import pandas as pd
import xarray as xr
import numpy as np

# --- Paths ---
PROJECT_ROOT = r"D:\rag_types\NER\ner-routing"
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "app", "data", "processed")
FEATURES_PATH = os.path.join(PROCESSED_DIR, "master_roads_final_features.gpkg")
WEATHER_OUT = os.path.join(PROCESSED_DIR, "weather", "temporal_roads_era5.parquet")
EXTRACT_DIR = os.path.join(PROJECT_ROOT, "app", "data", "raw", "weather")

def preprocess_era5(ds):
    """Safely collapse and completely remove the 'expver' dimension."""
    # If expver is a dimension, collapse it by taking the max (merges ERA5 and ERA5T)
    if 'expver' in ds.dims:
        ds = ds.max('expver', skipna=True)
    
    # Drop the coordinate/variable entirely to prevent stitching conflicts
    if 'expver' in ds.coords or 'expver' in ds.data_vars:
        ds = ds.drop_vars('expver', errors='ignore')
        
    return ds

def process_zipped_era5():
    nc_files = glob.glob(os.path.join(EXTRACT_DIR, "**", "*.nc"), recursive=True)
    print(f"Found {len(nc_files)} NetCDF files to process.")

    print("Loading master roads and calculating metric centroids...")
    roads = gpd.read_file(FEATURES_PATH)
    
    # Calculate centroids in projected metric CRS first (EPSG:32646), then project points to WGS84
    metric_centroids = roads.geometry.centroid
    centroids_4326 = gpd.GeoSeries(metric_centroids, crs=roads.crs).to_crs("EPSG:4326")
    
    lons = xr.DataArray(centroids_4326.x.values, dims="road")
    lats = xr.DataArray(centroids_4326.y.values, dims="road")
    road_ids = roads['road_id'].values

    print("Stitching NetCDF files along the time dimension...")
    # Removed compat='override' to fix the ValueError
    ds = xr.open_mfdataset(
        nc_files, 
        combine='by_coords', 
        parallel=False,
        preprocess=preprocess_era5
    )
    
    lon_name = 'longitude' if 'longitude' in ds.dims else 'lon'
    lat_name = 'latitude' if 'latitude' in ds.dims else 'lat'
    time_name = 'time' if 'time' in ds.dims else 'valid_time'
    precip_var = 'tp' if 'tp' in ds.data_vars else list(ds.data_vars)[0] 
    
    print(f"Extracting '{precip_var}' time series for {len(road_ids)} roads using nearest-neighbor...")
    precip_ts = ds[precip_var].sel(
        {lon_name: lons, lat_name: lats}, 
        method="nearest"
    )
    
    print("Converting to tabular panel format...")
    df = precip_ts.to_dataframe().reset_index()
    
    unique_times = len(df[time_name].unique())
    df['road_id'] = np.tile(road_ids, unique_times)
    
    df['rainfall_24h'] = df[precip_var] * 1000.0 
    
    panel = df[['road_id', time_name, 'rainfall_24h']].rename(columns={time_name: 'date'})
    panel['date'] = pd.to_datetime(panel['date']).dt.normalize()
    
    print("Aggregating hourly data to daily sums...")
    panel = panel.groupby(['road_id', 'date'])['rainfall_24h'].sum().reset_index()
    
    print("Calculating rolling window features (3d, 7d, anomaly)...")
    panel = panel.sort_values(by=['road_id', 'date']).reset_index(drop=True)
    grouped = panel.groupby('road_id')['rainfall_24h']
    
    panel['rainfall_3d'] = grouped.rolling(window=3, min_periods=1).sum().reset_index(level=0, drop=True)
    panel['rainfall_7d'] = grouped.rolling(window=7, min_periods=1).sum().reset_index(level=0, drop=True)
    
    panel['hist_mean'] = grouped.transform('mean')
    panel['rainfall_anomaly'] = panel['rainfall_24h'] - panel['hist_mean']
    panel = panel.drop(columns=['hist_mean'])
    
    print(f"Saving real ERA5 temporal matrix to {WEATHER_OUT}...")
    panel.to_parquet(WEATHER_OUT, index=False)
    print("Phase 3 weather integration complete!")

if __name__ == "__main__":
    process_zipped_era5()