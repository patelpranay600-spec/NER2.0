import os
import geopandas as gpd
import pandas as pd
import networkx as nx
import xgboost as xgb
import random
import warnings

# Suppress warnings for cleaner terminal output
warnings.filterwarnings('ignore')

# --- Paths ---
PROJECT_ROOT = r"D:\rag_types\NER\ner-routing"
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "app", "data", "processed")
ROADS_GPKG = os.path.join(PROCESSED_DIR, "master_roads_final_features.gpkg")
REAL_TARGET_DATA = os.path.join(PROCESSED_DIR, "master_panel_real_targets.parquet")
MODEL_FILE = os.path.join(PROCESSED_DIR, "xgboost_production_model.json")

def run_dynamic_routing():
    print("==================================================")
    print("🚛 NER LOGISTICS: DYNAMIC ROUTING ENGINE")
    print("==================================================\n")
    
    print("1. Loading Infrastructure Data and Weather Models...")
    roads = gpd.read_file(ROADS_GPKG)
    df = pd.read_parquet(REAL_TARGET_DATA)
    
    # Get the most recent day in the dataset to act as "today's forecast"
    latest_date = df['valid_time'].max()
    df_current = df[df['valid_time'] == latest_date].copy()
    
    print("2. Initializing AI Engine (Multi-Hazard Risk Analysis)...")
    clf = xgb.XGBClassifier()
    clf.load_model(MODEL_FILE)
    
    # EXACT feature list matching your trained model
    features = [
        'length_km_calculated', 'slope_mean', 'slope_max', 'roughness_mean', 
        'twi_mean', 'twi_max', 'log_flow_accum_mean', 'log_flow_accum_max',
        'rainfall_24h', 'rainfall_3d', 'rainfall_7d', 'rainfall_anomaly',
        'ndvi', 'mndwi', 's1_vv', 's1_vh', 'vh_vv_ratio', 'pred_label'
    ]
    
    # Predict Vulnerability Score (0.0 to 1.0)
    df_current['LVI_Score'] = clf.predict_proba(df_current[features].fillna(0.0))[:, 1]
    
    # Merge Risk Scores onto Road Geometries
    roads['road_id'] = roads['road_id'].astype(str)
    df_current['road_id'] = df_current['road_id'].astype(str)
    roads_risk = pd.merge(roads, df_current[['road_id', 'LVI_Score']], on='road_id', how='left')
    roads_risk['LVI_Score'] = roads_risk['LVI_Score'].fillna(0.0)
    
    print("3. Building Topological Routing Graph (NetworkX)...")
    G = nx.Graph()
    alpha = 30  # High penalty multiplier for risky roads
    
    for idx, row in roads_risk.iterrows():
        geom = row.geometry
        
        # Handle both LineStrings and MultiLineStrings
        if geom.geom_type == 'LineString':
            lines = [geom]
        elif geom.geom_type == 'MultiLineString':
            lines = list(geom.geoms)
        else:
            continue
            
        dist = row['length_km_calculated']
        risk = row['LVI_Score']
        
        # Dynamic Cost: Distance * (1 + risk penalty)
        dynamic_cost = dist * (1 + (alpha * (risk ** 3)))
        
        for line in lines:
            coords = list(line.coords)
            start_node, end_node = coords[0], coords[-1]
            G.add_edge(start_node, end_node, road_id=row['road_id'], 
                       normal_weight=dist, dynamic_weight=dynamic_cost, risk=risk)
                       
    # Isolate the largest connected network to prevent "No path" routing errors
    largest_cc = max(nx.connected_components(G), key=len)
    G_sub = G.subgraph(largest_cc)
    nodes = list(G_sub.nodes())
    
    print(f"   Graph built successfully with {len(nodes)} interconnected logistics hubs.\n")
    
    # Select random Origin and Destination far apart for demonstration
    random.seed(42)
    source_node = random.choice(nodes[:100])
    target_node = random.choice(nodes[-100:])
    
    print("==================================================")
    print(f"ROUTE REQUEST: Hub {hash(source_node) % 1000} ➔ Hub {hash(target_node) % 1000}")
    print(f"Forecast Date: {latest_date.date()}")
    print("==================================================")
    
    try:
        # Route 1: Standard Shortest Path (Ignoring weather/risk)
        normal_path = nx.shortest_path(G_sub, source=source_node, target=target_node, weight='normal_weight')
        normal_dist = nx.path_weight(G_sub, normal_path, 'normal_weight')
        normal_risk = max([G_sub[u][v]['risk'] for u, v in zip(normal_path[:-1], normal_path[1:])])
        
        # Route 2: AI Risk-Aware Path (Avoiding washouts/landslides)
        safe_path = nx.shortest_path(G_sub, source=source_node, target=target_node, weight='dynamic_weight')
        safe_dist = nx.path_weight(G_sub, safe_path, 'normal_weight')
        safe_max_risk = max([G_sub[u][v]['risk'] for u, v in zip(safe_path[:-1], safe_path[1:])])
        
        # Output Results
        print("\n📍 STANDARD ROUTE (GPS Default):")
        print(f"   Total Distance : {normal_dist:.2f} km")
        print(f"   Estimated Time : {(normal_dist/30)*60:.0f} minutes (avg 30 km/h)")
        
        if normal_risk > 0.4:
            print(f"   🚨 ALERT: Critical vulnerability detected on route. Max segment failure probability: {normal_risk*100:.1f}%")
            
            print("\n🛡️ AI ALTERNATIVE ROUTE (Risk-Aware):")
            print(f"   Total Distance : {safe_dist:.2f} km")
            print(f"   Estimated Time : {(safe_dist/30)*60:.0f} minutes")
            print(f"   Max Risk on Detour: {safe_max_risk*100:.1f}%")
            
            delay = ((safe_dist - normal_dist) / 30) * 60
            print(f"\n✅ RECOMMENDATION: Take AI alternative. Adds {delay:.0f} minutes to journey but safely bypasses high-risk infrastructure.")
        else:
            print(f"   ✅ Route is currently CLEAR of weather and terrain disruptions. Safe to proceed.")
            
    except nx.NetworkXNoPath:
        print("ERROR: No physical connection exists between these two hubs in the current network.")
        
    print("\n==================================================")

if __name__ == "__main__":
    run_dynamic_routing()