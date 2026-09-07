import rasterio
import numpy as np

VRT = r"D:\rag_types\NER\ner-routing\app\data\processed\dem_complete.vrt"

with rasterio.open(VRT) as src:

    data = src.read(1)

    print("====================================")
    print("DEM VALUE DIAGNOSTIC")
    print("====================================")

    print("dtype:", data.dtype)
    print("min:", np.nanmin(data))
    print("max:", np.nanmax(data))
    print("mean:", np.nanmean(data))

    count_1024 = np.sum(data == -1024)

    print("\n-1024 count:", count_1024)

    total = data.size

    print(
        "-1024 percentage:",
        count_1024 / total * 100
    )

    print("\nPercentiles:")

    for p in [0, 1, 5, 25, 50, 75, 95, 99, 100]:
        print(
            f"{p:>3}% :",
            np.percentile(data, p)
        )