import os
import geopandas as gpd
import pandas as pd
from rasterstats import zonal_stats

# --- Paths ---
PROJECT_ROOT = r"D:\rag_types\NER\ner-routing"
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "app", "data", "processed")
ROADS_PATH = os.path.join(PROCESSED_DIR, "master_pmgsy_roads.gpkg")

# Raster paths (EPSG:4326)
SLOPE_RASTER = os.path.join(PROCESSED_DIR, "terrain", "slope_deg.tif")
ASPECT_RASTER = os.path.join(PROCESSED_DIR, "terrain", "aspect_deg.tif")
ROUGHNESS_RASTER = os.path.join(PROCESSED_DIR, "terrain", "roughness_3x3.tif")

# Buffer distance in meters (Project spec recommends 100-250m)
BUFFER_DIST_METERS = 150

def extract_features():
    print(f"Loading master roads from {ROADS_PATH}...")
    roads = gpd.read_file(ROADS_PATH)
    
    print(f"Creating {BUFFER_DIST_METERS}m buffers in EPSG:32646...")
    # Buffer geometries in the metric CRS
    buffers = roads.geometry.buffer(BUFFER_DIST_METERS)
    
    # Create a temporary GeoDataFrame for the buffers and project to raster CRS (EPSG:4326)
    buffers_gdf = gpd.GeoDataFrame(geometry=buffers, crs=roads.crs)
    print("Projecting buffers to EPSG:4326 to match terrain rasters...")
    buffers_gdf_4326 = buffers_gdf.to_crs("EPSG:4326")
    
    # --- Extract Slope ---
    if os.path.exists(SLOPE_RASTER):
        print("Extracting Slope statistics (windowed reads)...")
        slope_stats = zonal_stats(
            buffers_gdf_4326, 
            SLOPE_RASTER, 
            stats="mean max", 
            nodata=-9999
        )
        roads['slope_mean'] = [s['mean'] for s in slope_stats]
        roads['slope_max'] = [s['max'] for s in slope_stats]
    else:
        print(f"Warning: {SLOPE_RASTER} not found.")

    # --- Extract Roughness ---
    if os.path.exists(ROUGHNESS_RASTER):
        print("Extracting Roughness statistics (windowed reads)...")
        rough_stats = zonal_stats(
            buffers_gdf_4326, 
            ROUGHNESS_RASTER, 
            stats="mean", 
            nodata=-9999
        )
        roads['roughness_mean'] = [s['mean'] for s in rough_stats]
    else:
        print(f"Warning: {ROUGHNESS_RASTER} not found.")

    # Save the enriched dataset
    out_path = os.path.join(PROCESSED_DIR, "master_roads_with_features.gpkg")
    print(f"Saving enriched road layer to {out_path}...")
    roads.to_file(out_path, driver="GPKG")
    print("Feature extraction complete.")

if __name__ == "__main__":
    extract_features()