import os
import geopandas as gpd

# --- Paths ---
PROJECT_ROOT = r"D:\rag_types\NER\ner-routing"
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "app", "data", "processed")
PMGSY_ROADS = os.path.join(PROCESSED_DIR, "master_roads_final_features.gpkg")

def check_merged_data():
    print(f"Loading merged Master Roads from {PMGSY_ROADS}...\n")
    gdf = gpd.read_file(PMGSY_ROADS)
    print(gdf.columns)
    
    print(f"Total Road Segments: {len(gdf)}")
    print("-" * 50)
    print("Infrastructure Columns Present:")
    
    target_cols = ['pred_class', 'pred_label', 'width', 'smoothness', 'bridge']
    found_cols = [col for col in target_cols if col in gdf.columns]
    
    if not found_cols:
        print("❌ ERROR: No infrastructure columns found. The merge failed.")
        return
        
    for col in found_cols:
        missing = gdf[col].isna().sum()
        print(f" - {col:<12} | Missing: {missing:<5} | Type: {gdf[col].dtype}")
        
    print("-" * 50)
    
    if 'pred_class' in gdf.columns:
        print("\nDistribution of Road Surface Types:")
        print(gdf['pred_class'].value_counts(dropna=False))
        
    if 'pred_label' in gdf.columns:
        print("\nDistribution of Binary Infrastructure Labels (0 = Paved, 1 = Unpaved):")
        print(gdf['pred_label'].value_counts(dropna=False))

if __name__ == "__main__":
    check_merged_data()