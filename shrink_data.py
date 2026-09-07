import os
import pandas as pd

# Define paths
PROCESSED_DIR = r"app\data\processed"
full_parquet = os.path.join(PROCESSED_DIR, "master_panel_real_targets.parquet")
slim_parquet = os.path.join(PROCESSED_DIR, "current_day_snapshot.parquet")

print("Loading massive panel (this might take a few seconds)...")
df = pd.read_parquet(full_parquet)

# Filter out the historical data, keep only the live forecast
latest_date = df['valid_time'].max()
df_latest = df[df['valid_time'] == latest_date].copy()

# Save the tiny file
df_latest.to_parquet(slim_parquet, index=False)
print(f"Success! Shrunk from {len(df)} rows to {len(df_latest)} rows.")