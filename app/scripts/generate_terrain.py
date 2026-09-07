from pathlib import Path
from osgeo import gdal
import subprocess


BASE = Path(r"D:\rag_types\NER\ner-routing\app")

DEM = BASE / "data" / "processed" / "dem_complete_v2.vrt"
OUT = BASE / "data" / "processed" / "terrain"

OUT.mkdir(parents=True, exist_ok=True)


def run(command):
    print("\n" + "=" * 70)
    print("RUNNING:")
    print(" ".join(map(str, command)))
    print("=" * 70)

    subprocess.run(command, check=True)


# ============================================================
# SLOPE
# ============================================================

slope = OUT / "slope_deg.tif"

run([
    "gdaldem",
    "slope",
    str(DEM),
    str(slope),
    "-s",
    "111120",
    "-compute_edges",
    "-of",
    "GTiff",
    "-co",
    "COMPRESS=LZW",
    "-co",
    "TILED=YES",
])

print("\nSlope generated:")
print(slope)


# ============================================================
# ASPECT
# ============================================================

aspect = OUT / "aspect_deg.tif"

run([
    "gdaldem",
    "aspect",
    str(DEM),
    str(aspect),
    "-compute_edges",
    "-of",
    "GTiff",
    "-co",
    "COMPRESS=LZW",
    "-co",
    "TILED=YES",
])

print("\nAspect generated:")
print(aspect)


print("\n" + "=" * 70)
print("TERRAIN GENERATION COMPLETE")
print("=" * 70)