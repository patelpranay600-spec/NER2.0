from pathlib import Path

import numpy as np
import rasterio


# ============================================================
# CONFIG
# ============================================================

DEM_ROOT = Path(
    r"D:\rag_types\NER\ner-routing\satalite model"
)

OUTPUT_DIR = Path(
    r"D:\rag_types\NER\ner-routing\app\data\processed\dem_normalized_v2"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_NODATA = -9999.0


# ============================================================
# FIND DEM FILES
# ============================================================

dem_files = sorted(
    DEM_ROOT.rglob("*.tif")
)

print("=" * 70)
print("DEM NORMALIZATION")
print("=" * 70)

print(f"Found {len(dem_files)} DEM files")


if len(dem_files) != 78:
    print(
        f"WARNING: Expected 78 files, "
        f"found {len(dem_files)}"
    )


# ============================================================
# PROCESS
# ============================================================

for i, src_path in enumerate(
    dem_files,
    start=1
):

    output_path = (
        OUTPUT_DIR /
        src_path.name
    )

    print(
        f"\n[{i}/{len(dem_files)}] "
        f"{src_path.name}"
    )

    with rasterio.open(src_path) as src:

        data = src.read(1)

        print(
            f"  Input dtype : {data.dtype}"
        )

        print(
            f"  Input nodata: {src.nodata}"
        )

        # ----------------------------------------------------
        # Convert to Float32
        # ----------------------------------------------------

        data = data.astype(
            np.float32
        )

        # ----------------------------------------------------
        # Identify NoData
        # ----------------------------------------------------

        invalid = np.zeros(
            data.shape,
            dtype=bool
        )

        # Source-declared NoData
        if src.nodata is not None:

            invalid |= (
                data == src.nodata
            )

        # Legacy CARTOSAT/ERDAS tiles
        # containing -1024 fill values
        invalid |= (
            data == -1024
        )

        # ----------------------------------------------------
        # Replace invalid values
        # ----------------------------------------------------

        invalid_count = np.sum(
            invalid
        )

        data[invalid] = OUTPUT_NODATA

        # ----------------------------------------------------
        # Create output profile
        # ----------------------------------------------------

        profile = src.profile.copy()

        profile.update(
            dtype="float32",
            count=1,
            nodata=OUTPUT_NODATA,
            compress="deflate",
            predictor=2
        )

        # ----------------------------------------------------
        # Write
        # ----------------------------------------------------

        with rasterio.open(
            output_path,
            "w",
            **profile
        ) as dst:

            dst.write(
                data,
                1
            )

        print(
            f"  NoData pixels : "
            f"{invalid_count:,}"
        )

print("\n" + "=" * 70)
print("NORMALIZATION COMPLETE")
print("=" * 70)

print(
    f"Output directory:\n{OUTPUT_DIR}"
)