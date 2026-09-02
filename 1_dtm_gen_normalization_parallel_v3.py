import concurrent.futures
import io
import multiprocessing
import os
import sys
import time

import laspy
import numpy as np
import rasterio
from rasterio.transform import from_origin
from scipy.interpolate import (RBFInterpolator, RegularGridInterpolator,
                               griddata)
from scipy.spatial import cKDTree
from sklearn.neighbors import KNeighborsRegressor

# Try to import tqdm for progress bars
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, total=None, desc=""):
        return iterable

# Fix for Windows console encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# ==========================================
#               HELPER FUNCTIONS
# ==========================================

def interpolate_points(ground_coords, ground_z, target_coords, method='linear'):
    """Interpolates Z values for specific target coordinates (Raw calculation)."""
    if method == 'linear':
        return griddata(ground_coords, ground_z, target_coords, method='linear')
    elif method == 'rbf':
        rbf = RBFInterpolator(ground_coords, ground_z, kernel='linear')
        return rbf(target_coords)
    elif method == 'knn':
        knn = KNeighborsRegressor(n_neighbors=6, weights='distance')
        knn.fit(ground_coords, ground_z)
        return knn.predict(target_coords)
    elif method == 'idw':
        tree = cKDTree(ground_coords)
        dist, idx = tree.query(target_coords, k=6)
        weights = 1.0 / (dist ** 2 + 1e-12)
        z_idw = np.sum(weights * ground_z[idx], axis=1) / np.sum(weights, axis=1)
        return z_idw
    else:
        raise ValueError(f"Unsupported method: {method}")


def generate_dtm_grid(x, y, z, grid_xx, grid_yy, method='linear'):
    """Generates the Z-grid (The Raster)."""
    coords = np.column_stack((x, y))
    # Flatten grid for interpolation
    target_grid_coords = np.column_stack((grid_xx.ravel(), grid_yy.ravel()))
    
    # Interpolate using the raw method (IDW, KNN, etc) to fill the pixels
    z_grid_flat = interpolate_points(coords, z, target_grid_coords, method=method)
    
    # Reshape back to 2D grid
    grid_z = z_grid_flat.reshape(grid_xx.shape)
    
    # --- INTERNAL EDGE FIX FOR THE GRID ---
    # If linear method leaves NaNs at edges of the grid, fill them now
    # so the raster is complete before we use it for normalization.
    if method == 'linear' and np.isnan(grid_z).any():
        # Identify NaN pixels
        mask_nan = np.isnan(grid_z)
        # Identify valid pixels
        mask_valid = ~mask_nan
        
        if np.any(mask_valid):
            # Use Nearest Neighbor to fill gaps in the grid itself
            # We use griddata on the pixel coordinates
            valid_coords = np.column_stack((grid_xx[mask_valid], grid_yy[mask_valid]))
            valid_values = grid_z[mask_valid]
            target_nan_coords = np.column_stack((grid_xx[mask_nan], grid_yy[mask_nan]))
            
            filled_values = griddata(valid_coords, valid_values, target_nan_coords, method='nearest')
            grid_z[mask_nan] = filled_values

    return grid_z


def save_dtm_to_disk(grid_z, min_x, max_y, resolution, output_path, crs):
    """Saves the in-memory grid to a GeoTIFF."""
    transform = from_origin(min_x, max_y, resolution, resolution)
    
    with rasterio.open(
        output_path,
        'w',
        driver='GTiff',
        height=grid_z.shape[0],
        width=grid_z.shape[1],
        count=1,
        dtype=grid_z.dtype,
        crs=crs,
        transform=transform,
        nodata=np.nan
    ) as dst:
        dst.write(grid_z, 1)


# ==========================================
#           CORE WORKER FUNCTION
# ==========================================

def process_single_file(args):
    """
    Worker function.
    Logic Updated: Now samples normalization height FROM the generated DTM Grid.
    """
    input_path, output_path, dtm_output_folder, resolution, method, crs = args
    filename = os.path.basename(input_path)
    
    try:
        las = laspy.read(input_path)
        x, y, z = las.x, las.y, las.z
        cls = las.classification

        # 1. Extract Ground Points
        ground_mask = cls == 2
        ground_x, ground_y, ground_z = x[ground_mask], y[ground_mask], z[ground_mask]

        if len(ground_z) == 0:
            return {'success': False, 'file': filename, 'msg': "Skipped: no ground points."}

        # 2. Define the DTM Grid Bounds
        # We use ground bounds to define the raster area
        min_x, max_x = np.min(ground_x), np.max(ground_x)
        min_y, max_y = np.min(ground_y), np.max(ground_y)

        # Create the coordinate arrays for the grid
        # Note: We align strictly to resolution steps
        grid_x_vals = np.arange(min_x, max_x + resolution, resolution)
        grid_y_vals = np.arange(min_y, max_y + resolution, resolution)
        grid_xx, grid_yy = np.meshgrid(grid_x_vals, grid_y_vals)

        # 3. Generate the DTM Grid (Rasterize the Ground)
        # This creates the "Pixelated" surface based on the resolution
        dtm_grid_z = generate_dtm_grid(ground_x, ground_y, ground_z, grid_xx, grid_yy, method)

        # 4. Save the DTM GeoTIFF
        dtm_tif_path = os.path.join(
            dtm_output_folder,
            os.path.splitext(filename)[0] + f"_dtm_{method}.tif"
        )
        save_dtm_to_disk(dtm_grid_z, min_x, max_y, resolution, dtm_tif_path, crs)

        # 5. Create a Sampler from the DTM Grid
        # We use RegularGridInterpolator to sample the "Pixelated" height for every LAS point.
        # This ensures the normalization is based on the resolution-specific surface.
        
        # Note: RegularGridInterpolator expects (y_axis, x_axis) and ascending coords
        # grid_y_vals and grid_x_vals are already ascending from np.arange
        dtm_interpolator = RegularGridInterpolator(
            (grid_y_vals, grid_x_vals), 
            dtm_grid_z, 
            bounds_error=False, 
            fill_value=np.nan,
            method='linear' # Bilinear interpolation between pixels
        )

        # 6. Sample DTM Height for ALL LAS points
        # We pass (y, x) because of the grid order
        dtm_z_sampled = dtm_interpolator((y, x))

        # 7. Normalize
        valid = ~np.isnan(dtm_z_sampled)
        x, y, z = x[valid], y[valid], z[valid]
        norm_z = z - dtm_z_sampled[valid]
        cls = cls[valid]

        if len(norm_z) == 0:
            return {'success': False, 'file': filename, 'msg': "Skipped: points outside DTM bounds."}

        # 8. Save New LAS
        header = laspy.LasHeader(point_format=las.header.point_format, version=las.header.version)
        header.scales = las.header.scales
        header.offsets = las.header.offsets

        new_las = laspy.create(point_format=header.point_format, file_version=header.version)
        new_las.header.scales = header.scales
        new_las.header.offsets = header.offsets
        new_las.points.resize(len(norm_z))

        new_las.x = x
        new_las.y = y
        new_las.z = norm_z

        for dim in las.point_format.dimension_names:
            if dim not in ['X', 'Y', 'Z']:
                setattr(new_las, dim, getattr(las, dim)[valid])

        new_las.write(output_path)
        return {'success': True, 'file': filename, 'msg': "Success"}

    except Exception as e:
        return {'success': False, 'file': filename, 'msg': f"Error: {str(e)}"}


# ==========================================
#           BATCH & PARALLEL LOGIC
# ==========================================

def batch_normalize_parallel(input_folder, output_folder, dtm_output_folder, resolution, method, crs, cpu_usage_percent):
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(dtm_output_folder, exist_ok=True)
    
    log_file_path = os.path.join(output_folder, "processing_log.txt")
    
    files = [f for f in os.listdir(input_folder) if f.lower().endswith(".laz")]
    tasks = []
    
    for file in files:
        in_path = os.path.join(input_folder, file)
        out_path = os.path.join(output_folder, file)
        tasks.append((in_path, out_path, dtm_output_folder, resolution, method, crs))
    
    # --- Dynamic Worker Calculation ---
    total_cores = multiprocessing.cpu_count()
    max_workers = int(total_cores * cpu_usage_percent)
    
    if max_workers < 1: max_workers = 1
    if max_workers > len(tasks): max_workers = len(tasks)

    print(f"  > Launching {max_workers} parallel workers (Targeting {cpu_usage_percent*100:.0f}% of {total_cores} cores)...")

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_single_file, task): task for task in tasks}
        
        with open(log_file_path, 'w') as log:
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(tasks), desc=f"Res {resolution}m", unit="file"):
                result = future.result()
                if not result['success']:
                    log.write(f"{result['file']}: {result['msg']}\n")


def run_multiple_resolutions(input_folder, base_dtm_folder, base_output_folder, resolutions, method='linear', crs='EPSG:25832', cpu_usage_percent=0.8):
    # --- Start Timer ---
    start_time = time.time()
    
    for resolution in resolutions:
        print(f"\n{'='*60}")
        print(f"PROCESSING BATCH: Resolution {resolution}m")
        print(f"{'='*60}")
        
        resolution_str = str(resolution).replace('.', '_')
        dtm_output_folder = os.path.join(base_dtm_folder, f"res_{resolution_str}")
        output_folder = os.path.join(base_output_folder, f"res_{resolution_str}")
        
        batch_normalize_parallel(
            input_folder, 
            output_folder, 
            dtm_output_folder,
            resolution=resolution, 
            method=method, 
            crs=crs,
            cpu_usage_percent=cpu_usage_percent 
        )
    
    # --- End Timer & Print ---
    end_time = time.time()
    elapsed_seconds = end_time - start_time
    hours, rem = divmod(elapsed_seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    
    print(f"\n{'='*60}")
    print(f"ALL PROCESSING COMPLETE")
    print(f"Total Elapsed Time: {int(hours):02d}h {int(minutes):02d}m {seconds:05.2f}s")
    print(f"{'='*60}")


# ==========================================
#               MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    # --- CONFIGURATION ---
    input_folder = "1_ClippedPCD"
    base_dtm_folder = "2_DTM_Parallel_idw"
    base_output_folder = "3_Normalized_Parallel_idw"
    
    # 1. Target Coordinate System
    target_crs = 'EPSG:25832' 
    
    # 2. CPU Usage Settings
    cpu_usage_percent = 0.9 
    
    # 3. Resolutions to process
    resolutions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    
    # 4. Interpolation Method (used to create the DTM Grid)
    interpolation_method = 'idw' # Options: 'linear', 'rbf', 'knn', 'idw'

    # --- RUN ---
    run_multiple_resolutions(
        input_folder, 
        base_dtm_folder, 
        base_output_folder,
        resolutions, 
        interpolation_method, 
        target_crs,
        cpu_usage_percent
    )