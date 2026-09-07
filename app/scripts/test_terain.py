from pathlib import Path

import geopandas as gpd

from app.preprocessing.terrain import TerrainEnricher


DEM_ROOT = Path(
    r"D:\rag_types\NER\ner-routing\satalite model"
)

VRT_PATH = Path(
    "data/processed/dem.vrt"
)

ROADS_PATH = Path(
    "data/processed/roads.parquet"
)


def main():

    print("Loading PMGSY roads...")

    roads = gpd.read_parquet(
        ROADS_PATH
    )

    print(
        f"Total roads: {len(roads):,}"
    )

    print(
        f"Road CRS: {roads.crs}"
    )

    # ---------------------------------------------------------
    # TEST ONLY
    # ---------------------------------------------------------

    test_roads = roads.head(10).copy()

    print(
        "\nTesting first 10 roads..."
    )

    terrain = TerrainEnricher(
        dem_root=DEM_ROOT,
        vrt_path=VRT_PATH,
        sample_spacing_m=100,
    )

    result = terrain.enrich(
        test_roads
    )

    print(
        "\n========== RESULTS ==========\n"
    )

    columns = [
        "elevation_mean",
        "elevation_min",
        "elevation_max",
        "slope_mean",
        "slope_max",
        "curvature_mean",
        "curvature_max",
    ]

    print(
        result[columns].to_string(
            index=False
        )
    )

    output = Path(
        "data/processed/"
        "roads_terrain_test.parquet"
    )

    result.to_parquet(
        output
    )

    print(
        f"\nSaved test output:\n"
        f"{output.resolve()}"
    )


if __name__ == "__main__":
    main()