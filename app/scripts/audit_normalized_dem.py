from pathlib import Path
import rasterio

DEM_DIR = Path(
    r"D:\rag_types\NER\ner-routing\app\data\processed\dem_normalized"
)

files = sorted(DEM_DIR.glob("*.tif"))

print(f"Found {len(files)} normalized DEMs\n")

dtypes = set()
crs_set = set()
resolutions = set()

left = float("inf")
bottom = float("inf")
right = float("-inf")
top = float("-inf")

for path in files:

    with rasterio.open(path) as src:

        dtypes.add(src.dtypes[0])
        crs_set.add(str(src.crs))
        resolutions.add(src.res)

        b = src.bounds

        left = min(left, b.left)
        bottom = min(bottom, b.bottom)
        right = max(right, b.right)
        top = max(top, b.top)


print("====================================")
print("NORMALIZED DEM AUDIT")
print("====================================")

print("Number of files :", len(files))
print("Data types      :", dtypes)
print("CRS             :", crs_set)
print("Resolutions     :", resolutions)

print("\nUnion bounds:")
print("Left   :", left)
print("Bottom :", bottom)
print("Right  :", right)
print("Top    :", top)

print("\nExpected:")
print("Data type → float32")
print("CRS       → EPSG:4326")
print("Coverage  → approximately 85–98 E, 19–29 N")