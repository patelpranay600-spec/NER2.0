from __future__ import annotations

import math
import subprocess
from pathlib import Path
from typing import List

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from pyproj import CRS
from shapely.geometry import Point


class TerrainEnricher:

    DEM_EXTENSIONS = {".tif", ".tiff"}

    TERRAIN_COLUMNS = [
        "elevation_mean",
        "elevation_min",
        "elevation_max",
        "slope_mean",
        "slope_max",
        "curvature_mean",
        "curvature_max",
    ]

    def __init__(
        self,
        dem_root: str | Path,
        vrt_path: str | Path = "data/processed/dem.vrt",
        sample_spacing_m: float = 100.0,
    ):

        self.dem_root = Path(dem_root)
        self.vrt_path = Path(vrt_path)
        self.sample_spacing_m = sample_spacing_m

        if not self.dem_root.exists():
            raise FileNotFoundError(
                f"DEM directory does not exist:\n{self.dem_root}"
            )

        self.vrt_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

    # =========================================================
    # DEM DISCOVERY
    # =========================================================

    def discover_dem_files(self) -> List[Path]:

        print("\nSearching for DEM files...")

        files = sorted(
            {
                path
                for path in self.dem_root.rglob("*")
                if (
                    path.is_file()
                    and path.suffix.lower()
                    in self.DEM_EXTENSIONS
                )
            }
        )

        print(
            f"Found {len(files):,} DEM files."
        )

        if not files:
            raise FileNotFoundError(
                "No DEM .tif/.tiff files found."
            )

        return files

    # =========================================================
    # BUILD VRT
    # =========================================================

    def build_vrt(
        self,
        dem_files: List[Path]
    ):

        print("\nBuilding DEM VRT...")

        command = [
            "gdalbuildvrt",
            "-overwrite",
            str(self.vrt_path.resolve()),
        ]

        command.extend(
            str(path.resolve())
            for path in dem_files
        )

        try:

            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=False,
            )

        except FileNotFoundError:

            raise RuntimeError(
                "gdalbuildvrt was not found.\n"
                "Install GDAL using:\n"
                "conda install -c conda-forge gdal"
            )

        if result.returncode != 0:

            print(result.stdout)
            print(result.stderr)

            raise RuntimeError(
                "Failed to create DEM VRT."
            )

        print(
            f"VRT created:\n"
            f"{self.vrt_path.resolve()}"
        )

    # =========================================================
    # LOCAL PROJECTED CRS
    # =========================================================

    @staticmethod
    def get_local_utm_crs(
        geometry,
        source_crs
    ) -> CRS:

        """
        Determine the UTM CRS appropriate for the road.

        The road centroid is transformed to WGS84 first.
        """

        centroid = (
            gpd.GeoSeries(
                [geometry],
                crs=source_crs
            )
            .to_crs("EPSG:4326")
            .iloc[0]
            .centroid
        )

        lon = centroid.x
        lat = centroid.y

        zone = int(
            math.floor(
                (lon + 180) / 6
            ) + 1
        )

        if lat >= 0:
            epsg = 32600 + zone
        else:
            epsg = 32700 + zone

        return CRS.from_epsg(epsg)

    # =========================================================
    # SAMPLE POINTS
    # =========================================================

    def sample_points(
        self,
        geometry
    ):

        length = geometry.length

        count = max(
            2,
            int(
                math.ceil(
                    length /
                    self.sample_spacing_m
                )
            ) + 1
        )

        distances = np.linspace(
            0,
            length,
            count
        )

        return [
            geometry.interpolate(
                float(distance)
            )
            for distance in distances
        ], distances

    # =========================================================
    # ELEVATION SAMPLING
    # =========================================================

    @staticmethod
    def sample_elevation(
        src,
        points
    ) -> np.ndarray:

        coordinates = [
            (point.x, point.y)
            for point in points
        ]

        values = []

        for sample in src.sample(
            coordinates
        ):

            value = float(
                sample[0]
            )

            if not np.isfinite(value):
                value = np.nan

            values.append(value)

        return np.asarray(
            values,
            dtype=np.float32
        )

    # =========================================================
    # PROFILE CLEANING
    # =========================================================

    @staticmethod
    def clean_profile(
        elevations
    ):

        elevations = np.asarray(
            elevations,
            dtype=np.float64
        )

        valid = np.isfinite(
            elevations
        )

        if valid.sum() < 2:
            return None

        indices = np.arange(
            len(elevations)
        )

        # Interpolate small gaps in the DEM profile.
        cleaned = np.interp(
            indices,
            indices[valid],
            elevations[valid]
        )

        return cleaned

    # =========================================================
    # SLOPE
    # =========================================================

    @staticmethod
    def calculate_slope(
        elevations,
        distances
    ):

        dz_dx = np.gradient(
            elevations,
            distances
        )

        slope_degrees = np.degrees(
            np.arctan(
                np.abs(dz_dx)
            )
        )

        return slope_degrees

    # =========================================================
    # CURVATURE
    # =========================================================

    @staticmethod
    def calculate_curvature(
        elevations,
        distances
    ):

        first_derivative = np.gradient(
            elevations,
            distances
        )

        second_derivative = np.gradient(
            first_derivative,
            distances
        )

        return second_derivative

    # =========================================================
    # EMPTY RESULT
    # =========================================================

    @classmethod
    def empty_features(cls):

        return {
            column: np.nan
            for column in cls.TERRAIN_COLUMNS
        }

    # =========================================================
    # ENRICH ONE ROAD
    # =========================================================

    def enrich_road(
        self,
        geometry,
        source_crs,
        dem_src
    ):

        if geometry is None or geometry.is_empty:
            return self.empty_features()

        # -----------------------------------------------------
        # Convert this road to a local metre-based CRS
        # -----------------------------------------------------

        local_crs = self.get_local_utm_crs(
            geometry,
            source_crs
        )

        road = (
            gpd.GeoSeries(
                [geometry],
                crs=source_crs
            )
            .to_crs(local_crs)
            .iloc[0]
        )

        # -----------------------------------------------------
        # Generate points every ~100 m
        # -----------------------------------------------------

        points_local, distances = (
            self.sample_points(road)
        )

        # -----------------------------------------------------
        # Convert points back to DEM CRS
        # -----------------------------------------------------

        points_wgs84 = (
            gpd.GeoSeries(
                points_local,
                crs=local_crs
            )
            .to_crs(dem_src.crs)
        )

        # -----------------------------------------------------
        # Sample DEM
        # -----------------------------------------------------

        elevations = self.sample_elevation(
            dem_src,
            points_wgs84
        )

        cleaned = self.clean_profile(
            elevations
        )

        if cleaned is None:
            return self.empty_features()

        # -----------------------------------------------------
        # Terrain derivatives
        # -----------------------------------------------------

        slope = self.calculate_slope(
            cleaned,
            distances
        )

        curvature = self.calculate_curvature(
            cleaned,
            distances
        )

        return {

            "elevation_mean":
                float(np.mean(cleaned)),

            "elevation_min":
                float(np.min(cleaned)),

            "elevation_max":
                float(np.max(cleaned)),

            "slope_mean":
                float(np.mean(slope)),

            "slope_max":
                float(np.max(slope)),

            "curvature_mean":
                float(np.mean(curvature)),

            "curvature_max":
                float(np.max(np.abs(curvature))),
        }

    # =========================================================
    # FULL ENRICHMENT
    # =========================================================

    def enrich(
        self,
        roads: gpd.GeoDataFrame
    ) -> gpd.GeoDataFrame:

        print("\nStarting terrain enrichment...")

        dem_files = self.discover_dem_files()

        # -----------------------------------------------------
        # Create VRT if required
        # -----------------------------------------------------

        if not self.vrt_path.exists():

            self.build_vrt(
                dem_files
            )

        else:

            print(
                f"\nUsing existing VRT:\n"
                f"{self.vrt_path.resolve()}"
            )

        # -----------------------------------------------------
        # Open DEM
        # -----------------------------------------------------

        with rasterio.open(
            self.vrt_path
        ) as dem_src:

            print("\nDEM:")

            print(
                f"CRS        : {dem_src.crs}"
            )

            print(
                f"Resolution : {dem_src.res}"
            )

            print(
                f"Size       : "
                f"{dem_src.width:,} × "
                f"{dem_src.height:,}"
            )

            # -------------------------------------------------
            # Validate CRS
            # -------------------------------------------------

            if dem_src.crs != CRS.from_epsg(4326):

                raise ValueError(
                    "Expected DEM CRS EPSG:4326."
                )

            terrain_records = []

            total = len(roads)

            # -------------------------------------------------
            # Process roads
            # -------------------------------------------------

            for i, geometry in enumerate(
                roads.geometry
            ):

                try:

                    features = self.enrich_road(
                        geometry,
                        roads.crs,
                        dem_src
                    )

                except Exception as exc:

                    print(
                        f"\nWarning: road {i} "
                        f"failed: {exc}"
                    )

                    features = (
                        self.empty_features()
                    )

                terrain_records.append(
                    features
                )

                if (
                    (i + 1) % 100 == 0
                    or i == 0
                    or i + 1 == total
                ):

                    print(
                        f"Terrain: "
                        f"{i + 1:,}/{total:,}"
                    )

        # -----------------------------------------------------
        # Attach features
        # -----------------------------------------------------

        terrain_df = pd.DataFrame(
            terrain_records
        )

        result = roads.copy()

        for column in self.TERRAIN_COLUMNS:

            result[column] = (
                terrain_df[column].values
            )

        # -----------------------------------------------------
        # Statistics
        # -----------------------------------------------------

        print(
            "\nTerrain enrichment finished."
        )

        print(
            "\nMissing values:"
        )

        for column in self.TERRAIN_COLUMNS:

            missing = int(
                result[column].isna().sum()
            )

            percentage = (
                missing / len(result) * 100
            )

            print(
                f"{column:20s}"
                f"{missing:6d} "
                f"({percentage:5.2f}%)"
            )

        return result