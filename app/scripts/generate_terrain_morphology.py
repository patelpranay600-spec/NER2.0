from pathlib import Path
import subprocess
import shutil
import sys


BASE = Path(
    r"D:\rag_types\NER\ner-routing\app\data\processed"
)

DEM = BASE / "dem_complete_v2.vrt"
OUT = BASE / "terrain"

OUT.mkdir(parents=True, exist_ok=True)


# ============================================================
# FIND GDALDEM
# ============================================================

GDALDEM = shutil.which("gdaldem")

if GDALDEM is None:
    raise RuntimeError(
        "\n"
        "ERROR: gdaldem.exe was not found in PATH.\n\n"
        "Run these commands:\n"
        "  conda activate geo_clean\n"
        "  conda install -c conda-forge gdal\n\n"
        "Then restart PowerShell and try again.\n"
    )

print("=" * 70)
print("GDAL ENVIRONMENT")
print("=" * 70)

print("Python:")
print(sys.executable)

print("\ngdaldem:")
print(GDALDEM)


# ============================================================
# COMMAND RUNNER
# ============================================================

def run(cmd):

    print("\n" + "=" * 70)
    print("RUNNING:")
    print(" ".join(map(str, cmd)))
    print("=" * 70)

    subprocess.run(cmd, check=True)


# ============================================================
# TRI
# ============================================================

tri = OUT / "tri.tif"

run([
    GDALDEM,
    "TRI",
    str(DEM),
    str(tri),
    "-compute_edges",
    "-of",
    "GTiff",
    "-co",
    "COMPRESS=LZW",
    "-co",
    "TILED=YES",
    "-co",
    "BIGTIFF=YES",
])


# ============================================================
# TPI
# ============================================================

tpi = OUT / "tpi.tif"

run([
    GDALDEM,
    "TPI",
    str(DEM),
    str(tpi),
    "-compute_edges",
    "-of",
    "GTiff",
    "-co",
    "COMPRESS=LZW",
    "-co",
    "TILED=YES",
    "-co",
    "BIGTIFF=YES",
])


# ============================================================
# CURVATURE
# ============================================================

curvature = OUT / "curvature.tif"

run([
    GDALDEM,
    "curvature",
    str(DEM),
    str(curvature),
    "-compute_edges",
    "-of",
    "GTiff",
    "-co",
    "COMPRESS=LZW",
    "-co",
    "TILED=YES",
    "-co",
    "BIGTIFF=YES",
])


print("\n" + "=" * 70)
print("TERRAIN MORPHOLOGY COMPLETE")
print("=" * 70)

print("\nGenerated:")

for f in [tri, tpi, curvature]:
    print(f"  {f}")