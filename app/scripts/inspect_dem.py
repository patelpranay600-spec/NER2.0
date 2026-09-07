from pathlib import Path

from app.preprocessing.terrain import TerrainEnricher


DEM_ROOT = Path(
    r"D:\rag_types\NER\ner-routing"
)

VRT_PATH = Path(
    "data/processed/dem.vrt"
)


def main():

    terrain = TerrainEnricher(
        dem_root=DEM_ROOT,
        vrt_path=VRT_PATH,
    )

    files = terrain.discover_dem_files()

    print("\nFirst 10 DEM files:\n")

    for file in files[:10]:

        print(file)

    print(
        f"\nTotal DEM files: "
        f"{len(files):,}"
    )

    print(
        "\nInspecting first DEM..."
    )

    info = terrain.inspect_dem(
        files[0]
    )

    for key, value in info.items():

        print(
            f"{key}: {value}"
        )


if __name__ == "__main__":
    main()