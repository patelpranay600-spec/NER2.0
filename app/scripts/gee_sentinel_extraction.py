import os
import ee
import geopandas as gpd
import json
import math

# Initialize Earth Engine
ee.Initialize(project='carecircle-496014')

def export_sentinel_features_chunked():
    # 1. Load local master roads
    roads_path = r"D:\rag_types\NER\ner-routing\app\data\processed\master_pmgsy_roads.gpkg"
    print(f"Loading all roads from {roads_path}...")
    roads = gpd.read_file(roads_path).to_crs("EPSG:4326")
    
    print(f"Preparing {len(roads)} road segments with 150m metric buffers...")
    buffered_geom = roads.to_crs("EPSG:32646").buffer(150).to_crs("EPSG:4326")
    roads['geometry'] = buffered_geom
    
    # Reduce chunk size to 150 to safely stay under the 10 MB payload limit
    chunk_size = 150
    total_chunks = math.ceil(len(roads) / chunk_size)
    
    for i in range(total_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, len(roads))
        chunk_df = roads.iloc[start_idx:end_idx].copy()
        
        print(f"Processing chunk {i + 1} of {total_chunks} (Rows {start_idx} to {end_idx})...")
        
        # Convert chunk to GeoJSON dictionary
        geojson_dict = json.loads(chunk_df[['road_id', 'geometry']].to_json())
        ee_roads = ee.FeatureCollection(geojson_dict)

        # 2. Sentinel-2 Optical Collection (NDVI, MNDWI)
        s2 = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED") \
            .filterDate('2025-01-01', '2025-12-31') \
            .filterBounds(ee_roads) \
            .filter(ee.Filter.lt('CLCLFY_CLOUD_COVERAGE_PERCENTAGE', 30))
        
        def mask_s2_clouds(image):
            qa = image.select('QA60')
            cloudBitMask = 1 << 10
            cirrusBitMask = 1 << 11
            mask = qa.bitwiseAnd(cloudBitMask).eq(0).And(qa.bitwiseAnd(cirrusBitMask).eq(0))
            return image.updateMask(mask).divide(10000)

        def add_s2_indices(image):
            ndvi = image.normalizedDifference(['B8', 'B4']).rename('ndvi')
            mndwi = image.normalizedDifference(['B3', 'B11']).rename('mndwi')
            return image.addBands([ndvi, mndwi])

        s2_median = s2.map(mask_s2_clouds).map(add_s2_indices).select(['ndvi', 'mndwi']).median()

        # 3. Sentinel-1 SAR Collection (VV, VH backscatter)
        s1 = ee.ImageCollection("COPERNICUS/S1_GRD") \
            .filterDate('2025-01-01', '2025-12-31') \
            .filterBounds(ee_roads) \
            .filter(ee.Filter.eq('instrumentMode', 'IW')) \
            .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV')) \
            .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))

        def add_s1_bands(image):
            vv = image.select('VV').rename('s1_vv')
            vh = image.select('VH').rename('s1_vh')
            ratio = image.select('VH').subtract(image.select('VV')).rename('vh_vv_ratio')
            return image.addBands([vv, vh, ratio])

        s1_median = s1.map(add_s1_bands).select(['s1_vv', 's1_vh', 'vh_vv_ratio']).median()

        # 4. Combine and Reduce Regions
        combined_image = s2_median.addBands(s1_median)

        extracted_data = combined_image.reduceRegions(
            collection=ee_roads,
            reducer=ee.Reducer.mean(),
            scale=30,
            crs='EPSG:4326'
        )

        # 5. Submit separate task per chunk
        task_desc = f'PMGSY_Sentinel_Chunk_{i + 1}'
        task = ee.batch.Export.table.toDrive(
            collection=extracted_data,
            description=task_desc,
            fileFormat='CSV',
            fileNamePrefix=f'pmgsy_sentinel_chunk_{i + 1}'
        )
        task.start()

    print("\nAll chunked export tasks successfully submitted to Google Drive!")

if __name__ == "__main__":
    export_sentinel_features_chunked()

