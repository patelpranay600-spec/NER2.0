from pathlib import Path

DEM_DIR = Path(
    r"D:\rag_types\NER\ner-routing\app\data\processed\dem_normalized_v2"
)

OUTPUT_LIST = Path("dem_files_v2.txt")

files = sorted(
    DEM_DIR.glob("*.tif")
)

print("=" * 70)
print("CREATING DEM FILE LIST")
print("=" * 70)

print(f"Found {len(files)} DEM files")

if len(files) != 78:
    raise RuntimeError(
        f"Expected 78 DEM files, found {len(files)}"
    )

with open(OUTPUT_LIST, "w", encoding="utf-8") as f:
    for path in files:
        f.write(str(path.resolve()) + "\n")

print(f"Created: {OUTPUT_LIST.resolve()}")
print("First file:", files[0])
print("Last file :", files[-1])