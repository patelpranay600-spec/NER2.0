from pathlib import Path
from whitebox.whitebox_tools import WhiteboxTools


BASE = Path(
    r"D:\rag_types\NER\ner-routing\app\data\processed"
)

HYDRO = BASE / "hydrology"

FILLED = HYDRO / "test_dem_filled.tif"
FLOW_DIR = HYDRO / "test_flow_direction.tif"

wbt = WhiteboxTools()
wbt.set_working_dir(str(HYDRO))

print("=" * 70)
print("D8 FLOW DIRECTION")
print("=" * 70)

print("Input:")
print(FILLED)

print("\nOutput:")
print(FLOW_DIR)

result = wbt.d8_pointer(
    dem=str(FILLED),
    output=str(FLOW_DIR)
)

print("\nWhitebox result:")
print(result)

print("\n" + "=" * 70)
print("D8 FLOW DIRECTION COMPLETE")
print("=" * 70)