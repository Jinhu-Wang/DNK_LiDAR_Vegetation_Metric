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

# Try to import tqdm for progress bars
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, total=None, desc="", unit="", leave=True):
        return iterable

# Fix for Windows console encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')



def res_str(res):
    """0.1 -> '0_1', 1.0 -> '1_0' (matches the res_X_Y folder convention used everywhere else)."""
    return str(res).replace('.', '_')


def write_geotiff(path, grid, transform, crs, metric_name):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with rasterio.open(
        path, 'w', driver='GTiff', height=grid.shape[0], width=grid.shape[1],
        count=1, dtype=rasterio.float32, crs=crs, transform=transform, nodata=np.nan
    ) as dst:
        dst.write(grid, 1)
        dst.set_band_description(1, f'Vegetation {metric_name}')


def metric_output_dir(base_output_folder, class_label, norm_res, metric, veg_res):
    return os.path.join(
        base_output_folder, class_label,
        f"res_{res_str(norm_res)}", metric, f"res_{res_str(veg_res)}"
    )


# ==========================================
#           CORE WORKER FUNCTION
# ==========================================

def process_single_file(args):
    """
    Worker: computes max/mean/median/p95 vegetation height for ONE file,
    ONE class filter, ONE (norm_res, veg_res) combination, in a single
    group-by-pixel pass (all 4 metrics share the same grouping).
    """
    (input_path, class_label, target_classes, norm_res, veg_res,
     crs, metrics_list, base_output_folder) = args
    filename = os.path.basename(input_path)
    file_stem = os.path.splitext(filename)[0]

    try:
        with laspy.open(input_path) as fh:
            las = fh.read()

        x = las.x
        y = las.y
        z = las.z
        classification = las.classification

        if len(x) == 0:
            return {'success': True, 'file': filename, 'msg': "Skipped (Empty file)"}

        # Grid snapped to veg_res, based on the FULL point extent (not just the
        # target-class subset) so class_4 / class_5 / class_4_5 rasters for the
        # same plot+resolution share identical grid origin/size.
        min_x = np.floor(np.min(x) / veg_res) * veg_res
        max_y = np.ceil(np.max(y) / veg_res) * veg_res
        max_x = np.ceil(np.max(x) / veg_res) * veg_res
        min_y = np.floor(np.min(y) / veg_res) * veg_res

        width = max(int(round((max_x - min_x) / veg_res)), 1)
        height = max(int(round((max_y - min_y) / veg_res)), 1)

        transform = from_origin(min_x, max_y, veg_res, veg_res)

        veg_mask = np.isin(classification, target_classes)

        # No points of this class: still write NaN rasters for every metric,
        # sized to the plot's full extent, so the output tree stays complete.
        if np.sum(veg_mask) == 0:
            empty_grid = np.full((height, width), np.nan, dtype=np.float32)
            for metric in metrics_list:
                out_path = os.path.join(
                    metric_output_dir(base_output_folder, class_label, norm_res, metric, veg_res),
                    f"{file_stem}_{metric}.tif"
                )
                write_geotiff(out_path, empty_grid, transform, crs, metric)
            return {'success': True, 'file': filename, 'msg': "Success (No Veg Points)"}

        x_veg = x[veg_mask]
        y_veg = y[veg_mask]
        z_veg = z[veg_mask]

        cols = np.clip(((x_veg - min_x) / veg_res).astype(int), 0, width - 1)
        rows = np.clip(((max_y - y_veg) / veg_res).astype(int), 0, height - 1)

        # Group points by pixel ONCE (sort by Z ascending within each pixel).
        flat_indices = rows * width + cols
        sort_order = np.lexsort((z_veg, flat_indices))
        flat_sorted = flat_indices[sort_order]
        z_sorted = z_veg[sort_order]

        unique_idx, counts = np.unique(flat_sorted, return_counts=True)
        u_rows = unique_idx // width
        u_cols = unique_idx % width

        run_starts = np.zeros(len(counts), dtype=int)
        run_starts[1:] = np.cumsum(counts[:-1])

        for metric in metrics_list:
            grid = np.full((height, width), np.nan, dtype=np.float32)

            if metric == 'max':
                idx = run_starts + counts - 1
                vals = z_sorted[idx]
            elif metric == 'mean':
                sums = np.add.reduceat(z_sorted, run_starts)
                vals = sums / counts
            elif metric == 'median':
                idx = run_starts + (counts // 2)
                vals = z_sorted[idx]
            elif metric == 'p95':
                offset = np.clip((counts * 0.95).astype(int), 0, counts - 1)
                idx = run_starts + offset
                vals = z_sorted[idx]
            else:
                continue

            grid[u_rows, u_cols] = vals

            out_path = os.path.join(
                metric_output_dir(base_output_folder, class_label, norm_res, metric, veg_res),
                f"{file_stem}_{metric}.tif"
            )
            write_geotiff(out_path, grid, transform, crs, metric)

        return {'success': True, 'file': filename, 'msg': "Success"}

    except Exception as e:
        return {'success': False, 'file': filename, 'msg': f"Error: {str(e)}"}


# ==========================================
#           SWEEP ORCHESTRATION
# ==========================================

def run_full_sweep(normalized_parent_folder, base_output_folder, resolutions,
                    class_configs, crs, cpu_usage_percent, metrics_list):
    """
    class_configs : dict of {class_label: [target_classification_codes]}
    Produces: base_output_folder/{class}/res_{norm}/{metric}/res_{veg}/*.tif
    Total raster sets = len(resolutions)^2 * len(metrics_list) * len(class_configs)
    """
    start_time = time.time()

    total_cores = multiprocessing.cpu_count()
    max_workers = max(1, int(total_cores * cpu_usage_percent))
    print(f"Workers per batch: {max_workers} ({cpu_usage_percent*100:.0f}% of {total_cores} cores)")

    for class_label, target_classes in class_configs.items():
        print(f"\n{'='*70}")
        print(f"CLASS: {class_label}  (target_classes={target_classes})")
        print(f"{'='*70}")

        for norm_res in resolutions:
            input_path = os.path.join(normalized_parent_folder, f"res_{res_str(norm_res)}")

            if not os.path.isdir(input_path):
                print(f"  [!] Missing input folder, skipping: {input_path}")
                continue

            files = [f for f in os.listdir(input_path) if f.lower().endswith(".laz")]
            if not files:
                print(f"  [!] No .laz files in {input_path}, skipping")
                continue

            print(f"\n  Normalization resolution {norm_res} m -- {len(files)} files ({input_path})")

            log_dir = os.path.join(base_output_folder, class_label, f"res_{res_str(norm_res)}")
            os.makedirs(log_dir, exist_ok=True)
            log_path = os.path.join(log_dir, "processing_log.txt")
            errors = []

            for veg_res in resolutions:
                # Pre-create output dirs in the main process to avoid worker race conditions
                # (write_geotiff also creates its own parent dir defensively, but doing it here
                # up front avoids every one of ~2568 workers racing to os.makedirs the same path).
                for metric in metrics_list:
                    os.makedirs(
                        metric_output_dir(base_output_folder, class_label, norm_res, metric, veg_res),
                        exist_ok=True
                    )

                tasks = [
                    (os.path.join(input_path, f), class_label, target_classes, norm_res, veg_res,
                     crs, metrics_list, base_output_folder)
                    for f in files
                ]

                # Fresh pool per batch (not one shared pool for the whole run): bounds how many
                # tasks any single worker process handles before being recycled, so a slow
                # per-task resource leak in GDAL/laspy can't accumulate across the full sweep.
                with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                    futures = [executor.submit(process_single_file, t) for t in tasks]

                    for fut in tqdm(concurrent.futures.as_completed(futures), total=len(futures),
                                     desc=f"    veg_res {veg_res}m", unit="file", leave=False):
                        result = fut.result()
                        if not result['success']:
                            errors.append(f"[veg_res={veg_res}] {result['file']}: {result['msg']}")

            if errors:
                with open(log_path, 'w') as f:
                    f.write("\n".join(errors))
                print(f"    {len(errors)} error(s) logged to {log_path}")

    elapsed_seconds = time.time() - start_time
    hours, rem = divmod(elapsed_seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"\n{'='*70}")
    print(f"ALL DONE. Total time: {int(hours):02d}h {int(minutes):02d}m {seconds:05.2f}s")
    print(f"{'='*70}")


# ==========================================
#               MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    # --- CONFIGURATION ---

    normalized_parent_folder = "3_Normalized_Parallel_idw"
    base_output_folder = "4_Vegetation_Metrics"

    resolutions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    target_crs = 'EPSG:25832'
    cpu_usage_percent = 0.90

    metrics_to_compute = ['mean', 'median', 'max', 'p95']

    # ASPRS classification: 4 = Medium Vegetation, 5 = High Vegetation
    class_configs = {
        'class_4':   [4],
        'class_5':   [5],
        'class_4_5': [4, 5],
    }

    if not os.path.isdir(normalized_parent_folder):
        print(f"ERROR: normalized_parent_folder not found: {normalized_parent_folder}")
        print("Update the path in the CONFIGURATION block above, then re-run.")
        sys.exit(1)

    run_full_sweep(
        normalized_parent_folder,
        base_output_folder,
        resolutions,
        class_configs,
        target_crs,
        cpu_usage_percent,
        metrics_to_compute
    )
