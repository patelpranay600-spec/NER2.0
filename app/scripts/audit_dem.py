from pathlib import Path
import rasterio
import numpy as np


DEM_DIR = Path(
    r"D:\rag_types\NER\ner-routing\app\data\processed\dem_normalized_v2"
)

files = sorted(
    DEM_DIR.glob("*.tif")
)

print("=" * 70)
print("NORMALIZED DEM AUDIT")
print("=" * 70)

print(
    "Number of files:",
    len(files)
)

dtypes = set()
nodatas = set()
crs_set = set()
resolutions = set()

left = float("inf")
bottom = float("inf")
right = float("-inf")
top = float("-inf")


for path in files:

    with rasterio.open(path) as src:

        dtypes.add(
            src.dtypes[0]
        )

        nodatas.add(
            src.nodata
        )

        crs_set.add(
            str(src.crs)
        )

        resolutions.add(
            src.res
        )

        b = src.bounds

        left = min(
            left,
            b.left
        )

        bottom = min(
            bottom,
            b.bottom
        )

        right = max(
            right,
            b.right
        )

        top = max(
            top,
            b.top
        )


print("\nDatatype:")
print(dtypes)

print("\nNoData:")
print(nodatas)

print("\nCRS:")
print(crs_set)

print("\nResolution:")
print(resolutions)

print("\nUnion bounds:")
print(
    left,
    bottom,
    right,
    top
)