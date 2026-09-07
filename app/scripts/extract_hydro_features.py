import os
import geopandas as gpd
import pandas as pd
import numpy as np
from rasterstats import zonal_stats

# --- Paths ---
PROJECT_ROOT = r"D:\rag_types\NER\ner-routing"
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "app", "data", "processed")
HYDRO_DIR = os.path.join(PROCESSED_DIR, "hydrology")

# Input/Output Roads
ROADS_PATH = os.path.join(PROCESSED_DIR, "master_roads_with_features.gpkg")
OUTPUT_PATH = os.path.join(PROCESSED_DIR, "master_roads_final_features.gpkg")

# Raster paths (EPSG:4326)
TWI_RASTER = os.path.join(HYDRO_DIR, "twi.tif")
ACCUM_RASTER = os.path.join(HYDRO_DIR, "flow_accumulation_d8_clean.tif")

BUFFER_DIST_METERS = 150

def extract_hydrology():
    print(f"Loading enriched roads from {ROADS_PATH}...")
    roads = gpd.read_file(ROADS_PATH)
    
    print(f"Creating {BUFFER_DIST_METERS}m buffers in EPSG:32646...")
    # Buffer geometries in the metric CRS
    buffers = roads.geometry.buffer(BUFFER_DIST_METERS)
    
    # Project to EPSG:4326 to match raster CRS
    buffers_gdf = gpd.GeoDataFrame(geometry=buffers, crs=roads.crs)
    print("Projecting buffers to EPSG:4326...")
    buffers_gdf_4326 = buffers_gdf.to_crs("EPSG:4326")
    
    # --- Extract TWI ---
    if os.path.exists(TWI_RASTER):
        print("Extracting TWI statistics (windowed reads)...")
        twi_stats = zonal_stats(
            buffers_gdf_4326, 
            TWI_RASTER, 
            stats="mean max", 
            nodata=-9999.0
        )
        roads['twi_mean'] = [s['mean'] for s in twi_stats]
        roads['twi_max'] = [s['max'] for s in twi_stats]
    else:
        print(f"Error: {TWI_RASTER} not found.")

    # --- Extract Flow Accumulation ---
    if os.path.exists(ACCUM_RASTER):
        print("Extracting Flow Accumulation statistics (windowed reads)...")
        accum_stats = zonal_stats(
            buffers_gdf_4326, 
            ACCUM_RASTER, 
            stats="mean max", 
            nodata=-9999.0
        )
        
        # Apply the required log1p transformation: ln(1 + flow_accumulation)
        print("Applying log1p transformation to flow accumulation...")
        roads['log_flow_accum_mean'] = [np.log1p(s['mean']) if s['mean'] is not None else None for s in accum_stats]
        roads['log_flow_accum_max'] = [np.log1p(s['max']) if s['max'] is not None else None for s in accum_stats]
    else:
        print(f"Error: {ACCUM_RASTER} not found.")

    print(f"Saving final feature layer to {OUTPUT_PATH}...")
    roads.to_file(OUTPUT_PATH, driver="GPKG")
    print("Hydrology feature extraction complete.")

if __name__ == "__main__":
    extract_hydrology()