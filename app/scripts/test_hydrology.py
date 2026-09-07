from pathlib import Path
from osgeo import gdal
from whitebox.whitebox_tools import WhiteboxTools


BASE = Path(
    r"D:\rag_types\NER\ner-routing\app\data\processed"
)

DEM = BASE / "dem_complete_v2.vrt"
HYDRO = BASE / "hydrology"

HYDRO.mkdir(parents=True, exist_ok=True)

TEST_DEM = HYDRO / "test_dem.tif"
FILLED = HYDRO / "test_dem_filled.tif"


# ------------------------------------------------------------
# Extract a small test region
# ------------------------------------------------------------

print("=" * 70)
print("EXTRACTING TEST DEM")
print("=" * 70)

gdal.Translate(
    str(TEST_DEM),
    str(DEM),
    projWin=[
        92.25,   # west
        25.25,   # north
        92.75,   # east
        24.75    # south
    ],
    format="GTiff",
    creationOptions=[
        "COMPRESS=LZW",
        "TILED=YES"
    ]
)

print("Test DEM:")
print(TEST_DEM)


# ------------------------------------------------------------
# WhiteboxTools
# ------------------------------------------------------------

wbt = WhiteboxTools()

wbt.set_working_dir(str(HYDRO))

print("\nWhiteboxTools:")
print(wbt.version())

print("\nWorking directory:")
print(HYDRO)


# ------------------------------------------------------------
# Fill depressions
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("FILLING DEPRESSIONS")
print("=" * 70)

result = wbt.fill_depressions(
    dem=str(TEST_DEM),
    output=str(FILLED),
    fix_flats=True
)

print("\nWhitebox result:")
print(result)

print("\nOutput:")
print(FILLED)

print("\n" + "=" * 70)
print("TEST COMPLETE")
print("=" * 70)