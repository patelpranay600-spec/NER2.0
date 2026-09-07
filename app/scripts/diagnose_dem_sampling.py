from pathlib import Path

import geopandas as gpd
import rasterio
import numpy as np


ROADS_PATH = Path(
    "data/processed/roads.parquet"
)

VRT_PATH = Path(
    "data/processed/dem.vrt"
)


def main():

    print("\n==============================")
    print("DEM SAMPLING DIAGNOSTIC")
    print("==============================\n")

    # ---------------------------------------------------------
    # LOAD ROADS
    # ---------------------------------------------------------

    roads = gpd.read_parquet(
        ROADS_PATH
    )

    print("ROADS")
    print("------------------------------")

    print(
        f"Count : {len(roads):,}"
    )

    print(
        f"CRS   : {roads.crs}"
    )

    print(
        f"Bounds: {roads.total_bounds}"
    )

    # ---------------------------------------------------------
    # OPEN VRT
    # ---------------------------------------------------------

    print("\nVRT")
    print("------------------------------")

    with rasterio.open(VRT_PATH) as src:

        print(
            f"CRS        : {src.crs}"
        )

        print(
            f"Width      : {src.width:,}"
        )

        print(
            f"Height     : {src.height:,}"
        )

        print(
            f"Resolution : {src.res}"
        )

        print(
            f"Bounds     : {src.bounds}"
        )

        print(
            f"NoData     : {src.nodata}"
        )

        # -----------------------------------------------------
        # Compare bounds
        # -----------------------------------------------------

        print("\nFIRST 10 ROAD CENTROIDS")
        print("------------------------------")

        roads_wgs84 = roads.to_crs(
            "EPSG:4326"
        )

        for i, geometry in enumerate(
            roads_wgs84.geometry.head(10)
        ):

            centroid = geometry.centroid

            inside = (
                src.bounds.left
                <= centroid.x
                <= src.bounds.right
                and
                src.bounds.bottom
                <= centroid.y
                <= src.bounds.top
            )

            print(
                f"Road {i:02d}: "
                f"lon={centroid.x:.6f}, "
                f"lat={centroid.y:.6f}, "
                f"inside_vrt={inside}"
            )

        # -----------------------------------------------------
        # Sample first road
        # -----------------------------------------------------

        print("\nFIRST ROAD SAMPLE")
        print("------------------------------")

        geometry = roads_wgs84.geometry.iloc[0]

        points = [
            geometry.interpolate(
                fraction,
                normalized=True
            )
            for fraction in np.linspace(
                0,
                1,
                10
            )
        ]

        coordinates = [
            (point.x, point.y)
            for point in points
        ]

        samples = list(
            src.sample(
                coordinates
            )
        )

        for coordinate, value in zip(
            coordinates,
            samples
        ):

            print(
                f"lon={coordinate[0]:.6f}, "
                f"lat={coordinate[1]:.6f}, "
                f"value={value[0]}"
            )

        # -----------------------------------------------------
        # Read actual VRT block
        # -----------------------------------------------------

        print("\nVRT VALUE STATISTICS")
        print("------------------------------")

        data = src.read(
            1,
            out_shape=(500, 500)
        )

        data = data.astype(
            np.float64
        )

        print(
            f"Min    : {np.nanmin(data)}"
        )

        print(
            f"Max    : {np.nanmax(data)}"
        )

        print(
            f"Mean   : {np.nanmean(data)}"
        )

        print(
            f"Unique sample values: "
            f"{len(np.unique(data))}"
        )

        print("\n==============================")
        print("END")
        print("==============================")


if __name__ == "__main__":
    main()