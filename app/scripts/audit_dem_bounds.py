from pathlib import Path
import rasterio
from collections import Counter

DEM_ROOT = Path(r"D:\rag_types\NER\ner-routing\satalite model")

dem_files = list(DEM_ROOT.rglob("*.tif")) + list(DEM_ROOT.rglob("*.tiff"))

print("=" * 70)
print("DEM FOOTPRINT AUDIT")
print("=" * 70)

print(f"\nTotal DEM files: {len(dem_files)}")

bounds_list = []

for i, path in enumerate(dem_files):

    try:
        with rasterio.open(path) as src:

            b = src.bounds

            bounds_list.append({
                "file": path.name,
                "path": str(path),
                "left": b.left,
                "bottom": b.bottom,
                "right": b.right,
                "top": b.top,
                "crs": str(src.crs),
                "width": src.width,
                "height": src.height,
                "res": src.res,
            })

    except Exception as e:
        print(f"ERROR: {path}")
        print(e)


# ---------------------------------------------------------
# UNION OF ALL DEM BOUNDS
# ---------------------------------------------------------

left = min(x["left"] for x in bounds_list)
bottom = min(x["bottom"] for x in bounds_list)
right = max(x["right"] for x in bounds_list)
top = max(x["top"] for x in bounds_list)

print("\n" + "=" * 70)
print("UNION OF ALL 78 DEMs")
print("=" * 70)

print(f"Left   : {left}")
print(f"Bottom : {bottom}")
print(f"Right  : {right}")
print(f"Top    : {top}")


# ---------------------------------------------------------
# PRINT EVERY DEM
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("INDIVIDUAL DEM EXTENTS")
print("=" * 70)

for i, x in enumerate(bounds_list):

    print(
        f"{i+1:02d}. "
        f"{x['file']:<45} "
        f""
        f"({x['left']:.4f}, {x['bottom']:.4f}) → "
        f"({x['right']:.4f}, {x['top']:.4f})"
    )


# ---------------------------------------------------------
# CHECK WHETHER DEM COVERAGE CAN COVER ALL ROADS
# ---------------------------------------------------------

ROAD_LEFT = 88.0867485
ROAD_BOTTOM = 22.08881252
ROAD_RIGHT = 96.83656372
ROAD_TOP = 28.83551999

print("\n" + "=" * 70)
print("ROAD COVERAGE CHECK")
print("=" * 70)

print("Road bounds:")
print(f"({ROAD_LEFT}, {ROAD_BOTTOM}) → ({ROAD_RIGHT}, {ROAD_TOP})")

print("\nDEM union:")
print(f"({left}, {bottom}) → ({right}, {top})")

if (
    left <= ROAD_LEFT
    and bottom <= ROAD_BOTTOM
    and right >= ROAD_RIGHT
    and top >= ROAD_TOP
):
    print("\n✓ ALL ROAD BOUNDS ARE COVERED BY DEM DATA")
else:
    print("\n✗ DEM DATA DOES NOT COVER ALL ROAD BOUNDS")

    if right < ROAD_RIGHT:
        print(f"  Missing eastward coverage: up to {ROAD_RIGHT:.4f}E")

    if top < ROAD_TOP:
        print(f"  Missing northward coverage: up to {ROAD_TOP:.4f}N")

    if left > ROAD_LEFT:
        print(f"  Missing westward coverage: down to {ROAD_LEFT:.4f}E")

    if bottom > ROAD_BOTTOM:
        print(f"  Missing southward coverage: down to {ROAD_BOTTOM:.4f}N")

print("\n" + "=" * 70)