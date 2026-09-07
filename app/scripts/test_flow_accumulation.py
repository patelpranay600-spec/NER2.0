from pathlib import Path
from whitebox.whitebox_tools import WhiteboxTools

BASE = Path(
    r"D:\rag_types\NER\ner-routing\app\data\processed"
)

HYDRO = BASE / "hydrology"

FLOW_DIR = HYDRO / "test_flow_direction.tif"
FLOW_ACC = HYDRO / "test_flow_accumulation_v2.tif"

wbt = WhiteboxTools()
wbt.set_working_dir(str(HYDRO))

print("=" * 70)
print("D8 FLOW ACCUMULATION — POINTER MODE")
print("=" * 70)

result = wbt.d8_flow_accumulation(
    i=str(FLOW_DIR),
    output=str(FLOW_ACC),
    out_type="cells",
    pntr=True
)

print("\nWhitebox result:")
print(result)

print("\n" + "=" * 70)
print("COMPLETE")
print("=" * 70)