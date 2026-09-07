import rasterio
import numpy as np
from pathlib import Path

DEM_ROOT = Path(
    r"D:\rag_types\NER\ner-routing\satalite model"
)

files = sorted(DEM_ROOT.rglob("*.tif"))

print("=" * 80)
print("SEARCHING FOR -1024")
print("=" * 80)

for path in files:

    with rasterio.open(path) as src:

        data = src.read(1)

        count = np.sum(data == -1024)

        if count > 0:

            percentage = count / data.size * 100

            print(
                f"{path.name:50s}"
                f" count={count:10d}"
                f"  percentage={percentage:7.3f}%"
                f"  nodata={src.nodata}"
            )