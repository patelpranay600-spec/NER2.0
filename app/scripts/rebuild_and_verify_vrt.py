from pathlib import Path
import rasterio
from osgeo import gdal


DEM_ROOT = Path(
    r"D:\rag_types\NER\ner-routing\satalite model"
)

VRT_PATH = Path(
    r"D:\rag_types\NER\ner-routing\app\data\processed\dem.vrt"
)


print("=" * 70)
print("REBUILDING DEM VRT")
print("=" * 70)


# ---------------------------------------------------------
# DISCOVER DEMs
# ---------------------------------------------------------

dem_files = sorted(
    set(
        list(DEM_ROOT.rglob("*.tif")) +
        list(DEM_ROOT.rglob("*.tiff"))
    )
)

print(f"\nDEM files discovered: {len(dem_files)}")

if len(dem_files) != 78:
    print("WARNING: Expected 78 DEMs!")


# ---------------------------------------------------------
# REMOVE OLD VRT
# ---------------------------------------------------------

if VRT_PATH.exists():

    print("\nRemoving old VRT...")
    VRT_PATH.unlink()


# ---------------------------------------------------------
# BUILD VRT
# ---------------------------------------------------------

print("\nBuilding new VRT...")

options = gdal.BuildVRTOptions(
    resampleAlg="nearest"
)

vrt = gdal.BuildVRT(
    str(VRT_PATH),
    [str(p) for p in dem_files],
    options=options
)

if vrt is None:
    raise RuntimeError("GDAL BuildVRT failed")

vrt.FlushCache()
vrt = None


# ---------------------------------------------------------
# INSPECT
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("NEW VRT INSPECTION")
print("=" * 70)

with rasterio.open(VRT_PATH) as src:

    print(f"\nCRS        : {src.crs}")
    print(f"Width      : {src.width}")
    print(f"Height     : {src.height}")
    print(f"Resolution : {src.res}")
    print(f"Bounds     : {src.bounds}")
    print(f"NoData     : {src.nodata}")

    # -----------------------------------------------------
    # SAMPLE IMPORTANT ROAD LOCATIONS
    # -----------------------------------------------------

    test_points = [
        # Western
        (88.5, 22.5),

        # Central
        (92.5, 24.5),

        # Eastern
        (95.0, 27.5),

        # Far eastern
        (97.5, 28.5),
    ]

    print("\n" + "=" * 70)
    print("TEST POINT SAMPLING")
    print("=" * 70)

    for lon, lat in test_points:

        inside = (
            src.bounds.left <= lon <= src.bounds.right
            and
            src.bounds.bottom <= lat <= src.bounds.top
        )

        if inside:

            value = next(
                src.sample([(lon, lat)])
            )[0]

            print(
                f"lon={lon:.2f}, lat={lat:.2f} "
                f"inside=True  value={value}"
            )

        else:

            print(
                f"lon={lon:.2f}, lat={lat:.2f} "
                f"inside=False"
            )


print("\n" + "=" * 70)
print("DONE")
print("=" * 70)