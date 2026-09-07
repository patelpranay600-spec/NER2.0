from pathlib import Path
import rasterio
import numpy as np

DEM = Path(
    r"D:\rag_types\NER\ner-routing\app\data\processed\dem_complete_v2.vrt"
)

print("=" * 70)
print("FINAL DEM QUALITY CHECK")
print("=" * 70)

with rasterio.open(DEM) as src:

    print("CRS      :", src.crs)
    print("Size     :", src.width, "x", src.height)
    print("dtype    :", src.dtypes[0])
    print("nodata   :", src.nodata)
    print("bounds   :", src.bounds)
    print("resolution:", src.res)

    # Read in windows instead of loading the entire 1.68B-pixel raster
    total = 0
    nodata_count = 0
    minus1024_count = 0
    negative_count = 0

    min_val = np.inf
    max_val = -np.inf

    for _, window in src.block_windows(1):

        arr = src.read(1, window=window)

        total += arr.size

        if src.nodata is not None:
            nodata_count += np.count_nonzero(arr == src.nodata)

        minus1024_count += np.count_nonzero(arr == -1024)

        valid = arr[
            np.isfinite(arr) &
            (arr != -9999)
        ]

        if valid.size:
            min_val = min(min_val, float(valid.min()))
            max_val = max(max_val, float(valid.max()))

        negative_count += np.count_nonzero(
            (arr < 0) & (arr != -9999)
        )

print()
print("=" * 70)
print("RESULTS")
print("=" * 70)

print(f"Total pixels       : {total:,}")
print(f"NoData pixels      : {nodata_count:,}")
print(
    f"-1024 pixels       : {minus1024_count:,} "
    f"({100 * minus1024_count / total:.4f}%)"
)
print(
    f"Negative pixels    : {negative_count:,} "
    f"({100 * negative_count / total:.4f}%)"
)

print(f"Valid minimum      : {min_val}")
print(f"Valid maximum      : {max_val}")

print("=" * 70)