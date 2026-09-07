from pathlib import Path
import rasterio
import numpy as np
from rasterio.windows import Window


BASE = Path(
    r"D:\rag_types\NER\ner-routing\app\data\processed"
)

DEM = BASE / "dem_complete_v2.vrt"

OUT_DIR = BASE / "terrain"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT = OUT_DIR / "roughness_3x3.tif"


# ============================================================
# SETTINGS
# ============================================================

WINDOW_SIZE = 512
RADIUS = 1

NODATA = -9999.0


# ============================================================
# OPEN DEM
# ============================================================

with rasterio.open(DEM) as src:

    profile = src.profile.copy()

    profile.update(
        driver="GTiff",
        dtype="float32",
        count=1,
        nodata=NODATA,
        compress="LZW",
        tiled=True,
        BIGTIFF="YES",
        blockxsize=512,
        blockysize=512
    )

    width = src.width
    height = src.height

    print("=" * 70)
    print("TERRAIN ROUGHNESS GENERATION")
    print("=" * 70)

    print("Input:")
    print(DEM)

    print()
    print("Size:")
    print(width, "x", height)

    print()
    print("Output:")
    print(OUTPUT)

    with rasterio.open(OUTPUT, "w", **profile) as dst:

        total_windows = (
            ((width + WINDOW_SIZE - 1) // WINDOW_SIZE)
            *
            ((height + WINDOW_SIZE - 1) // WINDOW_SIZE)
        )

        processed = 0

        for row in range(0, height, WINDOW_SIZE):

            for col in range(0, width, WINDOW_SIZE):

                processed += 1

                # ------------------------------------------------
                # Core window
                # ------------------------------------------------

                w = min(WINDOW_SIZE, width - col)
                h = min(WINDOW_SIZE, height - row)

                core = Window(
                    col,
                    row,
                    w,
                    h
                )

                # ------------------------------------------------
                # Expanded window
                #
                # Need one-pixel border because we're calculating
                # a 3x3 neighbourhood.
                # ------------------------------------------------

                expanded = Window(
                    max(0, col - RADIUS),
                    max(0, row - RADIUS),
                    min(
                        width - max(0, col - RADIUS),
                        w + 2 * RADIUS
                    ),
                    min(
                        height - max(0, row - RADIUS),
                        h + 2 * RADIUS
                    )
                )

                dem = src.read(
                    1,
                    window=expanded,
                    boundless=True,
                    fill_value=NODATA
                ).astype(np.float32)

                # ------------------------------------------------
                # Valid mask
                # ------------------------------------------------

                valid = (
                    np.isfinite(dem)
                    &
                    (dem != NODATA)
                )

                # ------------------------------------------------
                # Pad array
                # ------------------------------------------------

                padded = np.pad(
                    dem,
                    1,
                    mode="edge"
                )

                valid_padded = np.pad(
                    valid,
                    1,
                    mode="constant",
                    constant_values=False
                )

                # ------------------------------------------------
                # Calculate local mean
                # ------------------------------------------------

                sum_values = np.zeros_like(
                    dem,
                    dtype=np.float32
                )

                sum_sq = np.zeros_like(
                    dem,
                    dtype=np.float32
                )

                count = np.zeros_like(
                    dem,
                    dtype=np.float32
                )

                for dy in range(3):

                    for dx in range(3):

                        values = padded[
                            dy:dy + dem.shape[0],
                            dx:dx + dem.shape[1]
                        ]

                        mask = valid_padded[
                            dy:dy + dem.shape[0],
                            dx:dx + dem.shape[1]
                        ]

                        sum_values += np.where(
                            mask,
                            values,
                            0
                        )

                        sum_sq += np.where(
                            mask,
                            values * values,
                            0
                        )

                        count += mask.astype(
                            np.float32
                        )

                # ------------------------------------------------
                # Variance
                # ------------------------------------------------

                roughness = np.full(
                    dem.shape,
                    NODATA,
                    dtype=np.float32
                )

                enough_data = count >= 3

                mean = np.zeros_like(
                    dem,
                    dtype=np.float32
                )

                mean[enough_data] = (
                    sum_values[enough_data]
                    /
                    count[enough_data]
                )

                variance = np.zeros_like(
                    dem,
                    dtype=np.float32
                )

                variance[enough_data] = (
                    sum_sq[enough_data]
                    /
                    count[enough_data]
                    -
                    mean[enough_data] ** 2
                )

                # Numerical safety
                variance = np.maximum(
                    variance,
                    0
                )

                roughness[enough_data] = np.sqrt(
                    variance[enough_data]
                )

                # ------------------------------------------------
                # Extract core region from expanded result
                # ------------------------------------------------

                row_offset = row - expanded.row_off
                col_offset = col - expanded.col_off

                core_result = roughness[
                    row_offset:row_offset + h,
                    col_offset:col_offset + w
                ]

                dst.write(
                    core_result,
                    1,
                    window=core
                )

                # ------------------------------------------------
                # Progress
                # ------------------------------------------------

                if (
                    processed % 100 == 0
                    or processed == total_windows
                ):

                    percent = (
                        processed /
                        total_windows *
                        100
                    )

                    print(
                        f"{percent:6.2f}% "
                        f"({processed}/{total_windows})"
                    )


print()
print("=" * 70)
print("ROUGHNESS GENERATION COMPLETE")
print("=" * 70)

print()
print("Output:")
print(OUTPUT)