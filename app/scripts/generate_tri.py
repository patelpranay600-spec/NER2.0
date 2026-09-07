from pathlib import Path
import subprocess
import shutil
import sys


BASE = Path(
    r"D:\rag_types\NER\ner-routing\app\data\processed"
)

DEM = BASE / "dem_complete_v2.vrt"
OUT = BASE / "terrain"
TRI = OUT / "tri.tif"

OUT.mkdir(parents=True, exist_ok=True)


# ============================================================
# CHECK GDAL
# ============================================================

GDALDEM = shutil.which("gdaldem")

print("=" * 70)
print("GDAL CHECK")
print("=" * 70)

print("Python:")
print(sys.executable)

print("\ngdaldem:")
print(GDALDEM)

if GDALDEM is None:
    raise RuntimeError(
        "gdaldem.exe was not found. "
        "Activate geo_clean and check GDAL installation."
    )


# ============================================================
# CHECK INPUT
# ============================================================

if not DEM.exists():
    raise FileNotFoundError(
        f"DEM VRT not found:\n{DEM}"
    )

print("\nInput DEM:")
print(DEM)

print("\nOutput:")
print(TRI)


# ============================================================
# GENERATE TRI
# ============================================================

cmd = [
    GDALDEM,
    "TRI",
    str(DEM),
    str(TRI),

    "-compute_edges",

    "-of",
    "GTiff",

    "-co",
    "COMPRESS=LZW",
    "-co",
    "TILED=YES",
    "-co",
    "BIGTIFF=YES",
]

print("\n" + "=" * 70)
print("GENERATING TRI")
print("=" * 70)

print(" ".join(map(str, cmd)))

subprocess.run(cmd, check=True)


# ============================================================
# RESULT
# ============================================================

print("\n" + "=" * 70)
print("TRI GENERATION COMPLETE")
print("=" * 70)

print(TRI)
