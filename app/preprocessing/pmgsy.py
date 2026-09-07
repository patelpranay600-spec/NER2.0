from __future__ import annotations

from pathlib import Path
from typing import Optional

import geopandas as gpd
import pandas as pd


class PMGSYIngestor:
    """
    PMGSY road-data ingestion and preprocessing pipeline.

    Converts heterogeneous PMGSY spatial files into one
    clean GeoDataFrame suitable for downstream processing.
    """

    SUPPORTED_EXTENSIONS = {
        ".geojson",
        ".shp",
        ".gpkg",
    }

    # Broad Northeast India bounding box.
    LAT_BOUNDS = (21.0, 30.0)
    LON_BOUNDS = (88.0, 98.0)

    TARGET_CRS = "EPSG:4326"

    REQUIRED_COLUMNS = [
        "road_id",
        "road_name",
        "road_class",
        "surface_type",
        "pmgsy_phase",
        "state_name",
        "length_km",
        "geometry",
    ]

    def __init__(self, root_directory: str | Path):

        self.root_dir = Path(root_directory)

        if not self.root_dir.exists():
            raise FileNotFoundError(
                f"PMGSY directory does not exist: {self.root_dir}"
            )

    # ---------------------------------------------------------
    # FILE DISCOVERY
    # ---------------------------------------------------------

    def discover_files(self) -> list[Path]:

        files = []

        for path in self.root_dir.rglob("*"):

            if (
                path.is_file()
                and path.suffix.lower() in self.SUPPORTED_EXTENSIONS
            ):
                files.append(path)

        return sorted(files)

    # ---------------------------------------------------------
    # METADATA
    # ---------------------------------------------------------

    def extract_metadata(
        self,
        file_path: Path
    ) -> tuple[str, str]:

        """
        Extract PMGSY phase and state from parent folder.

        Example:

        Proposal_PMGSY-III_Assam

        -> PMGSY-III
        -> Assam
        """

        folder_name = file_path.parent.name

        parts = folder_name.split("_")

        phase = "UNKNOWN_PHASE"
        state = "UNKNOWN_STATE"

        for part in parts:

            if "PMGSY" in part.upper():

                phase = part

                break

        # State is normally the last component.
        if len(parts) >= 2:

            possible_state = parts[-1].strip()

            if possible_state:
                state = possible_state

        return phase, state

    # ---------------------------------------------------------
    # COLUMN STANDARDIZATION
    # ---------------------------------------------------------

    def _find_column(
        self,
        columns,
        candidates: list[str]
    ) -> Optional[str]:

        normalized = {
            str(c).strip().lower(): c
            for c in columns
        }

        for candidate in candidates:

            key = candidate.lower()

            if key in normalized:

                return normalized[key]

        return None

    def standardize_columns(
        self,
        gdf: gpd.GeoDataFrame,
        phase: str,
        state: str
    ) -> gpd.GeoDataFrame:

        result = gdf.copy()

        # -----------------------------
        # ROAD ID
        # -----------------------------

        road_id_col = self._find_column(
            result.columns,
            [
                "road_id",
                "roadid",
                "road_code",
                "roadcode",
                "link_id",
            ],
        )

        if road_id_col:

            result["road_id"] = (
                result[road_id_col]
                .astype(str)
                .str.strip()
            )

        else:

            result["road_id"] = [
                f"{state[:3].upper()}_{phase}_{i}"
                for i in range(len(result))
            ]

        # -----------------------------
        # ROAD NAME
        # -----------------------------

        road_name_col = self._find_column(
            result.columns,
            [
                "road_name",
                "roadname",
                "name",
                "link_name",
            ],
        )

        if road_name_col:

            result["road_name"] = (
                result[road_name_col]
                .fillna("PMGSY Rural Link")
                .astype(str)
            )

        else:

            result["road_name"] = (
                f"{state} Rural Link"
            )

        # -----------------------------
        # ROAD CLASS
        # -----------------------------

        road_class_col = self._find_column(
            result.columns,
            [
                "road_class",
                "roadcat",
                "road_cat",
                "category",
                "road_category",
            ],
        )

        if road_class_col:

            result["road_class"] = (
                result[road_class_col]
                .fillna("Habitation Link")
                .astype(str)
            )

        else:

            result["road_class"] = "Habitation Link"

        # -----------------------------
        # SURFACE
        # -----------------------------

        surface_col = self._find_column(
            result.columns,
            [
                "surface_type",
                "surface",
                "road_surface",
                "surf_type",
            ],
        )

        if surface_col:

            result["surface_type"] = (
                result[surface_col]
                .fillna("BT")
                .astype(str)
            )

        else:

            result["surface_type"] = "BT"

        # -----------------------------
        # METADATA
        # -----------------------------

        result["pmgsy_phase"] = phase
        result["state_name"] = state

        return result

    # ---------------------------------------------------------
    # GEOMETRY CLEANING
    # ---------------------------------------------------------

    def clean_geometry(
        self,
        gdf: gpd.GeoDataFrame
    ) -> gpd.GeoDataFrame:

        gdf = gdf.copy()

        # Remove missing geometries.
        gdf = gdf[
            gdf.geometry.notna()
        ].copy()

        # Remove empty geometries.
        gdf = gdf[
            ~gdf.geometry.is_empty
        ].copy()

        if gdf.empty:

            return gdf

        # Explode MultiLineStrings.
        gdf = (
            gdf
            .explode(
                index_parts=False
            )
            .reset_index(drop=True)
        )

        # Keep only LineStrings.
        gdf = gdf[
            gdf.geometry.geom_type == "LineString"
        ].copy()

        return gdf

    # ---------------------------------------------------------
    # BOUNDING BOX
    # ---------------------------------------------------------

    def filter_region(
        self,
        gdf: gpd.GeoDataFrame
    ) -> gpd.GeoDataFrame:

        min_lon, max_lon = self.LON_BOUNDS
        min_lat, max_lat = self.LAT_BOUNDS

        return gdf.cx[
            min_lon:max_lon,
            min_lat:max_lat
        ].copy()

    # ---------------------------------------------------------
    # LENGTH
    # ---------------------------------------------------------

    def calculate_length(
        self,
        gdf: gpd.GeoDataFrame
    ) -> gpd.GeoDataFrame:

        """
        Calculate approximate road length in kilometres.

        Uses geodesic length through GeoPandas' projected
        representation. For the final production version we
        can replace this with a more precise regional CRS
        strategy.
        """

        gdf = gdf.copy()

        # Use Web Mercator only as an intermediate fallback.
        # We will improve CRS handling in the next iteration.
        projected = gdf.to_crs("EPSG:3857")

        gdf["length_km"] = (
            projected.geometry.length / 1000.0
        ).round(4)

        return gdf

    # ---------------------------------------------------------
    # SINGLE FILE
    # ---------------------------------------------------------

    def process_file(
        self,
        file_path: Path
    ) -> Optional[gpd.GeoDataFrame]:

        try:

            print(f"Processing: {file_path}")

            phase, state = self.extract_metadata(
                file_path
            )

            gdf = gpd.read_file(file_path)

            if gdf.empty:

                print("  -> Empty file")

                return None

            # CRS
            if gdf.crs is None:

                print(
                    "  -> WARNING: CRS missing; "
                    "assuming EPSG:4326"
                )

                gdf = gdf.set_crs(
                    self.TARGET_CRS
                )

            elif gdf.crs.to_string() != self.TARGET_CRS:

                gdf = gdf.to_crs(
                    self.TARGET_CRS
                )

            # Geometry
            gdf = self.clean_geometry(gdf)

            if gdf.empty:

                print("  -> No valid LineString geometry")

                return None

            # Region
            gdf = self.filter_region(gdf)

            if gdf.empty:

                print("  -> Outside Northeast bounds")

                return None

            # Attributes
            gdf = self.standardize_columns(
                gdf,
                phase,
                state,
            )

            # Length
            gdf = self.calculate_length(gdf)

            return gdf[
                self.REQUIRED_COLUMNS
            ].copy()

        except Exception as exc:

            print(
                f"  -> ERROR: {file_path.name}: {exc}"
            )

            return None

    # ---------------------------------------------------------
    # COMPLETE PIPELINE
    # ---------------------------------------------------------

    def process(self) -> gpd.GeoDataFrame:

        files = self.discover_files()

        print(
            f"\nDiscovered {len(files)} spatial files."
        )

        if not files:

            raise ValueError(
                "No PMGSY spatial files found."
            )

        processed = []

        for file_path in files:

            result = self.process_file(
                file_path
            )

            if result is not None:

                processed.append(result)

        if not processed:

            raise ValueError(
                "No valid PMGSY road geometry was produced."
            )

        roads = gpd.GeoDataFrame(
            pd.concat(
                processed,
                ignore_index=True
            ),
            geometry="geometry",
            crs=self.TARGET_CRS,
        )

        # Remove duplicate geometries.
        roads = roads[
            ~roads.geometry.duplicated()
        ].copy()

        roads.reset_index(
            drop=True,
            inplace=True
        )

        print(
            f"\nTotal unified road segments: "
            f"{len(roads):,}"
        )

        return roads

    # ---------------------------------------------------------
    # SAVE
    # ---------------------------------------------------------

    def save(
        self,
        gdf: gpd.GeoDataFrame,
        output_path: str | Path
    ):

        output_path = Path(output_path)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        gdf.to_parquet(
            output_path,
            index=False
        )

        print(
            f"Saved processed network → "
            f"{output_path}"
        )