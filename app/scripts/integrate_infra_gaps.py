import os
import geopandas as gpd
import pandas as pd

# --- Paths ---
PROJECT_ROOT = r"D:\rag_types\NER\ner-routing"
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "app", "data", "processed")
PMGSY_ROADS = os.path.join(PROCESSED_DIR, "master_roads_final_features.gpkg")
HDX_OSM_ROADS = r"D:\rag_types\NER\ner-routing\heigit_ind_roadsurface_lines.gpkg" # KEEP YOUR EXACT PATH

def integrate_infrastructure_gaps():
    print(f"1. Loading PMGSY Master Roads from {PMGSY_ROADS}...")
    pmgsy_gdf = gpd.read_file(PMGSY_ROADS)
    original_crs = pmgsy_gdf.crs
    
    # --- THE FIX: Drop the empty columns from the previous failed run ---
    infra_cols = ['pred_class', 'pred_label', 'width', 'smoothness', 'bridge']
    existing_cols = [c for c in infra_cols if c in pmgsy_gdf.columns]
    if existing_cols:
        pmgsy_gdf = pmgsy_gdf.drop(columns=existing_cols)
        print(f"   Cleaned out old infrastructure columns: {existing_cols}")
    
    # Ensure PMGSY is in WGS84 before getting the bbox
    if pmgsy_gdf.crs != "EPSG:4326":
        pmgsy_gdf = pmgsy_gdf.to_crs("EPSG:4326")
        
    ner_bbox = tuple(pmgsy_gdf.total_bounds)
    print(f"   NER Bounding Box (WGS84): {ner_bbox}")
    
    print(f"\n2. Reading HDX OSM dataset...")
    # Global datasets are almost always EPSG:4326 natively
    osm_gdf = gpd.read_file(HDX_OSM_ROADS, bbox=ner_bbox)
    print(f"   Loaded {len(osm_gdf)} OSM road segments within the NER bounding box.")
    
    if len(osm_gdf) == 0:
        print("❌ ERROR: Zero OSM roads found in this region.")
        return

    keep_cols = ['pred_class', 'pred_label', 'width', 'smoothness', 'bridge', 'geometry']
    available_cols = [col for col in keep_cols if col in osm_gdf.columns]
    osm_gdf = osm_gdf[available_cols]
    
    print("\n3. Projecting to metric CRS (EPSG:32646) for a wider 500m search...")
    pmgsy_gdf = pmgsy_gdf.to_crs(epsg=32646)
    osm_gdf = osm_gdf.to_crs(epsg=32646)
    
    print("4. Performing Nearest-Neighbor Spatial Join (500m radius)...")
    merged_gdf = gpd.sjoin_nearest(pmgsy_gdf, osm_gdf, how="left", max_distance=500, distance_col="osm_dist")
    
    matches = merged_gdf['pred_label'].notna().sum()
    print(f"   ✅ Successfully matched {matches} out of {len(pmgsy_gdf)} roads within 500m.")
    
    print("\n5. Cleaning and imputing missing infrastructure data...")
    merged_gdf['pred_label'] = merged_gdf['pred_label'].fillna(1.0) # 1 = Unpaved
    merged_gdf['pred_class'] = merged_gdf['pred_class'].fillna('unpaved')
    
    if 'index_right' in merged_gdf.columns:
        merged_gdf = merged_gdf.drop(columns=['index_right', 'osm_dist'])
        
    merged_gdf = merged_gdf.to_crs(original_crs)
    
    # Deduplicate in case a PMGSY road matched multiple OSM segments
    merged_gdf = merged_gdf.drop_duplicates(subset=['road_id'])
    
    print(f"\n6. Overwriting master road features...")
    merged_gdf.to_file(PMGSY_ROADS, driver="GPKG")
    print("Integration complete. Now your pipeline has Paved vs Unpaved infrastructure data!")

if __name__ == "__main__":
    integrate_infrastructure_gaps()