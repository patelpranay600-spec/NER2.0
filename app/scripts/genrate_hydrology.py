import os
import rasterio
import numpy as np

# --- Paths ---
PROJECT_ROOT = r"D:\rag_types\NER\ner-routing"
HYDRO_DIR = os.path.join(PROJECT_ROOT, "app", "data", "processed", "hydrology")

ACCUM_FILE = os.path.join(HYDRO_DIR, "flow_accumulation_d8_clean.tif")
SLOPE_FILE = os.path.join(HYDRO_DIR, "slope_wbt_clean.tif")
TWI_FILE = os.path.join(HYDRO_DIR, "twi.tif")

def calculate_twi_memory_safe():
    print("Calculating TWI using windowed reads (Memory-Safe)...")
    
    # Delete corrupted file from previous failed run if it exists
    if os.path.exists(TWI_FILE):
        print(f"Removing corrupted {TWI_FILE}...")
        os.remove(TWI_FILE)
    
    with rasterio.open(ACCUM_FILE) as src_acc, rasterio.open(SLOPE_FILE) as src_slp:
        profile = src_acc.profile
        
        # Explicitly set blockxsize and blockysize to 256 (multiples of 16)
        profile.update(
            dtype=rasterio.float32, 
            nodata=-9999.0, 
            compress='lzw', 
            tiled=True,
            blockxsize=256,
            blockysize=256,
            bigtiff='yes'
        )
        
        with rasterio.open(TWI_FILE, 'w', **profile) as dst:
            for ji, window in src_acc.block_windows(1):
                acc = src_acc.read(1, window=window).astype(np.float32)
                slp = src_slp.read(1, window=window).astype(np.float32)
                
                valid_mask = (acc != src_acc.nodata) & (slp != src_slp.nodata)
                twi = np.full(acc.shape, -9999.0, dtype=np.float32)
                
                if np.any(valid_mask):
                    slp_clamped = np.clip(slp[valid_mask], 0.001, 89.999)
                    slp_rad = np.radians(slp_clamped)
                    
                    a = acc[valid_mask] + 1.0
                    twi[valid_mask] = np.log(a / np.tan(slp_rad))
                
                dst.write(twi, 1, window=window)
                
    print(f"TWI successfully calculated and saved to: {TWI_FILE}")

if __name__ == "__main__":
    calculate_twi_memory_safe()