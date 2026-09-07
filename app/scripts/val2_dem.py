import rasterio

VRT = (
    r"D:\rag_types\NER\ner-routing"
    r"\app\data\processed\dem_complete_v2.vrt"
)

with rasterio.open(VRT) as src:

    print("=" * 70)
    print("FINAL DEM VRT")
    print("=" * 70)

    print("Size:")
    print(src.width, "x", src.height)

    print("\nCRS:")
    print(src.crs)

    print("\nResolution:")
    print(src.res)

    print("\nDatatype:")
    print(src.dtypes[0])

    print("\nNoData:")
    print(src.nodata)

    print("\nBounds:")
    print(src.bounds)