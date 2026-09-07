import pandas as pd

# Update with your actual download path
gsi_path = r"D:\rag_types\NER\ner-routing\gsi_landslide_inventory.parquet"

df = pd.read_parquet(gsi_path)

# Drop rows where BOTH Initiation and Report are missing/empty
date_check_df = df.dropna(subset=['INITIATION', 'REPORT'], how='all')

print(f"Found {len(date_check_df)} rows with Initiation or Report data.")
print("\n--- Sample Entries ---")
print(date_check_df[['SLIDE_NO', 'INITIATION', 'REPORT']].head(10))