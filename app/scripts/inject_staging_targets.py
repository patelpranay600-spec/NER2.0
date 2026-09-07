import os
import pandas as pd
import numpy as np

# --- Paths ---
PROJECT_ROOT = r"D:\rag_types\NER\ner-routing"
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "app", "data", "processed")
REAL_TARGET_OUT = os.path.join(PROCESSED_DIR, "master_panel_real_targets.parquet")

def inject_staging_targets():
    print(f"Loading {REAL_TARGET_OUT}...")
    df = pd.read_parquet(REAL_TARGET_OUT)
    
    print("Injecting plausible staging targets (3.5% event rate) to test pipeline plumbing...")
    # Create a base risk score where higher slopes and more rain increase probability
    base_risk = (df['slope_max'] / df['slope_max'].max() * 0.4) + \
                (df['rainfall_3d'] / df['rainfall_3d'].max() * 0.6)
    
    # Inject random noise to simulate unmeasured variables
    np.random.seed(42)
    noise = np.random.normal(0, 0.15, len(df)) 
    final_risk = base_risk + noise
    
    # Flag the top 3.5% most vulnerable road-days as blockages
    threshold = np.percentile(final_risk, 96.5)
    df['target_blockage'] = np.where(final_risk > threshold, 1.0, 0.0)
    
    print(f"Total Blockages Injected: {int(df['target_blockage'].sum())} out of {len(df)} rows.")
    
    print("Saving updated panel...")
    df.to_parquet(REAL_TARGET_OUT, index=False)
    print("Done! You are ready to run validation and routing.")

if __name__ == "__main__":
    inject_staging_targets()