import rasterio

VRT = r"D:\rag_types\NER\ner-routing\app\data\processed\dem_complete.vrt"

test_points = [
    (88.5, 22.5),
    (92.5, 24.5),
    (95.0, 27.5),
    (97.5, 28.5),
]


with rasterio.open(VRT) as src:

    print("====================================")
    print("DEM SAMPLING TEST")
    print("====================================")

    print("VRT bounds:")
    print(src.bounds)

    for lon, lat in test_points:

        inside = (
            src.bounds.left <= lon <= src.bounds.right
            and
            src.bounds.bottom <= lat <= src.bounds.top
        )

        if inside:

            value = next(src.sample([(lon, lat)]))[0]

            print(
                f"\nPoint: ({lon}, {lat})"
            )

            print("Inside :", inside)
            print("DEM    :", value)

        else:

            print(
                f"\nPoint: ({lon}, {lat})"
            )

            print("Inside : False")
            print("DEM    : NaN")