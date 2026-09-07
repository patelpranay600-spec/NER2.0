from pathlib import Path
import subprocess
import shutil
import sys

BASE = Path(
    r"D:\rag_types\NER\ner-routing\app\data\processed"
)

DEM = BASE / "dem_complete_v2.vrt"
OUT_DIR = BASE / "hydrology"

FILLED_DEM = OUT_DIR / "dem_filled.tif"

OUT_DIR.mkdir(parents=True, exist_ok=True)

GDALDEM = shutil.which("gdaldem")

if GDALDEM is None:
    raise RuntimeError("gdaldem.exe not found.")

print("=" * 70)
print("HYDROLOGY — DEM FILL")
print("=" * 70)

print("Python :", sys.executable)
print("gdaldem:", GDALDEM)
print("Input  :", DEM)
print("Output :", FILLED_DEM)

cmd = [
    GDALDEM,
    "fill",
    str(DEM),
    str(FILLED_DEM),

    "-of",
    "GTiff",

    "-co",
    "COMPRESS=LZW",
    "-co",
    "TILED=YES",
    "-co",
    "BIGTIFF=YES",
]

print("\nRunning:")
print(" ".join(map(str, cmd)))

subprocess.run(cmd, check=True)

print("\n" + "=" * 70)
print("DEM FILL COMPLETE")
print("=" * 70)
print(FILLED_DEM)
