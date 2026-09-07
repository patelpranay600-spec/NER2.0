import pandas as pd
df = pd.read_csv(r"D:\rag_types\NER\ner-routing\app\data\processed\weather\sentinel_features_combined.csv")
print(df[['ndvi', 'mndwi', 's1_vv', 's1_vh', 'vh_vv_ratio']].describe())