from pathlib import Path
import rasterio
import numpy as np


BASE = Path(
    r"D:\rag_types\NER\ner-routing\app\data\processed\terrain"
)

FILES = {
    "SLOPE": BASE / "slope_deg.tif",
    "ASPECT": BASE / "aspect_deg.tif",
}


def inspect(name, path):

    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)

    with rasterio.open(path) as src:

        print("dtype      :", src.dtypes[0])
        print("nodata     :", src.nodata)
        print("size       :", src.width, "x", src.height)
        print("CRS        :", src.crs)
        print("resolution :", src.res)
        print("bounds     :", src.bounds)

        min_val = np.inf
        max_val = -np.inf

        total = 0
        valid_count = 0
        nodata_count = 0

        for _, window in src.block_windows(1):

            arr = src.read(1, window=window)

            total += arr.size

            if src.nodata is not None:
                nodata_count += np.count_nonzero(
                    arr == src.nodata
                )

            valid = arr[
                np.isfinite(arr) &
                (arr != src.nodata)
            ]

            if valid.size:

                min_val = min(
                    min_val,
                    float(valid.min())
                )

                max_val = max(
                    max_val,
                    float(valid.max())
                )

                valid_count += valid.size

        print()
        print("Total pixels :", f"{total:,}")
        print("Valid pixels :", f"{valid_count:,}")
        print("NoData       :", f"{nodata_count:,}")
        print("Minimum      :", min_val)
        print("Maximum      :", max_val)


for name, path in FILES.items():

    inspect(name, path)


print("\n" + "=" * 70)
print("INSPECTION COMPLETE")
print("=" * 70)