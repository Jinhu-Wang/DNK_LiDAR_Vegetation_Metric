import os
import random
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import seaborn as sns

# ==========================================
#        GLOBAL MEMORY FOR WORKERS
# ==========================================
# This variable will live in each worker process to store the GT data once.
GLOBAL_GT_DICT = {}

def init_worker(shared_gt_dict):
    """
    Initializes the worker process by loading the Ground Truth dictionary
    into its global scope. This avoids re-pickling the dict for every task.
    """
    global GLOBAL_GT_DICT
    GLOBAL_GT_DICT = shared_gt_dict


# ==========================================
#              HELPER FUNCTIONS
# ==========================================

def load_ground_truth_txt(filename):
    """
    Robust loader that finds the file relative to the script
    and normalizes filenames to ensure matches.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, filename)

    print(f"Looking for Ground Truth at: {file_path}", flush=True)

    if not os.path.exists(file_path):
        if os.path.exists(filename):
            file_path = filename
        else:
            print(f"[!] CRITICAL: File not found. Python is looking here:\n    {os.getcwd()}")
            return {}

    try:
        df = pd.read_csv(file_path, sep=' ', names=['filename', 'under', 'over', 'total'], header=None)

        gt_dict = {}
        for _, row in df.iterrows():
            raw_name = str(row['filename']).strip()
            fname = os.path.basename(raw_name)
            if not fname.lower().endswith('.laz'):
                fname += '.laz'

            gt_dict[fname] = (float(row['under']), float(row['over']), float(row['total']))

        print(f"Loaded {len(gt_dict)} rows. First key: '{list(gt_dict.keys())[0]}'", flush=True)
        return gt_dict

    except Exception as e:
        print(f"Error reading file: {e}")
        return {}


def scan_class_metric_work(metrics_root_folder, class_label, metric, resolutions, gt_dict):
    """
    Scans metrics_root_folder/{class_label}/res_{norm}/{metric}/res_{veg}/*.tif
    for ONE class + ONE metric across the full norm_res x veg_res grid.
    """
    work_items = []
    files_found = 0
    suffix = f"_{metric}.tif"

    for norm_res in resolutions:
        norm_folder = f"res_{str(norm_res).replace('.', '_')}"
        for veg_res in resolutions:
            veg_folder = f"res_{str(veg_res).replace('.', '_')}"
            folder_path = os.path.join(metrics_root_folder, class_label, norm_folder, metric, veg_folder)

            if os.path.exists(folder_path):
                with os.scandir(folder_path) as it:
                    for entry in it:
                        if entry.name.endswith(suffix) and entry.is_file():
                            files_found += 1
                            laz_key = entry.name[: -len(suffix)] + ".laz"
                            if laz_key in gt_dict:
                                work_items.append((entry.path, norm_res, veg_res, laz_key))

    print(f"    Scan: {files_found} tif files found, {len(work_items)} matched to ground truth.", flush=True)
    return work_items


def process_file_batch(batch):
    """
    Worker Function. Uses GLOBAL_GT_DICT (set once per worker via init_worker).
    Computes area diffs (LiDAR - GT) for under 1m / over 1m / total within a
    5m-radius circular window centered on the raster's own bounding-box centroid.
    """
    results = []
    radius_m = 5.0
    global GLOBAL_GT_DICT

    for filepath, norm_res, veg_res, laz_key in batch:
        if laz_key not in GLOBAL_GT_DICT:
            continue

        gt_vals = GLOBAL_GT_DICT[laz_key]
        pixel_area_m2 = veg_res * veg_res
        radius_px = radius_m / veg_res

        try:
            with rasterio.open(filepath) as src:
                # 1. Geometry Math
                left, bottom, right, top = src.bounds
                center_x = (left + right) / 2.0
                center_y = (top + bottom) / 2.0

                inv_tf = ~src.transform
                p_col, p_row = inv_tf * (center_x, center_y)

                # 2. Window Calculation
                height, width = src.shape
                min_row = max(0, int(np.floor(p_row - radius_px)))
                max_row = min(height, int(np.ceil(p_row + radius_px)))
                min_col = max(0, int(np.floor(p_col - radius_px)))
                max_col = min(width, int(np.ceil(p_col + radius_px)))

                if max_row <= min_row or max_col <= min_col:
                    continue

                # 3. Read Data (windowed read, not the whole raster)
                window = rasterio.windows.Window(col_off=min_col, row_off=min_row,
                                                  width=max_col - min_col, height=max_row - min_row)
                data = src.read(1, window=window)

                # 4. Circular Mask
                rows, cols = np.indices(data.shape)
                rel_row = p_row - min_row
                rel_col = p_col - min_col
                dist_sq = (rows - rel_row) ** 2 + (cols - rel_col) ** 2
                mask = dist_sq <= radius_px ** 2
                masked_data = data[mask]

                # 5. Valid pixels: v4's rasters are NaN for "no vegetation of this
                # class/metric here", not 0 -- so `~isnan`, not `> 0`.
                valid_pixels = masked_data[~np.isnan(masked_data)]

                count_total = valid_pixels.size
                if count_total > 0:
                    count_under = np.sum(valid_pixels < 1.0)
                    count_over = count_total - count_under
                else:
                    count_under = 0
                    count_over = 0

                # 6. Diff Calculation (LiDAR - GT)
                diff_u = (count_under * pixel_area_m2) - gt_vals[0]
                diff_o = (count_over * pixel_area_m2) - gt_vals[1]
                diff_t = (count_total * pixel_area_m2) - gt_vals[2]

                results.append((norm_res, veg_res, diff_u, diff_o, diff_t))

        except Exception:
            continue

    return results


def plot_heatmaps(output_folder, class_label, metric):
    """Generates the 3 heatmaps (under 1m / over 1m / total) for one class+metric combo."""
    files = {
        'Under 1m': 'diff_matrix_area_under_1m.csv',
        'Over 1m': 'diff_matrix_area_over_1m.csv',
        'Total': 'diff_matrix_area_total.csv',
    }

    for title, csv_file in files.items():
        csv_path = os.path.join(output_folder, csv_file)
        if not os.path.exists(csv_path):
            continue

        try:
            df = pd.read_csv(csv_path, index_col=0)
            if df.isnull().all().all():
                continue

            plt.figure(figsize=(10, 8))
            sns.heatmap(df, annot=True, fmt=".2f", cmap="RdBu", center=0, linewidths=.5)
            plt.title(f'{class_label} / {metric} -- {title}\nMean Difference (LiDAR - GT) [m²]', fontsize=14)
            plt.xlabel('Vegetation Metric Resolution (m)', fontsize=12)
            plt.ylabel('Normalization Resolution (m)', fontsize=12)

            out_name = f"heatmap_{csv_file.replace('.csv', '.png')}"
            save_path = os.path.join(output_folder, out_name)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"    Saved plot: {out_name}", flush=True)
        except Exception:
            pass


# ==========================================
#           CORE LOGIC WRAPPER
# ==========================================

def run_validation_for_combo(gt_dict, metrics_root_folder, output_dir, resolutions, class_label, metric):
    print(f"\n{'='*70}")
    print(f"CLASS: {class_label}  |  METRIC: {metric}")
    print(f"{'='*70}")

    combo_output_dir = os.path.join(output_dir, class_label, metric)
    os.makedirs(combo_output_dir, exist_ok=True)

    work_items = scan_class_metric_work(metrics_root_folder, class_label, metric, resolutions, gt_dict)
    if not work_items:
        print("    No matching files found. Skipping.")
        return

    random.shuffle(work_items)
    chunk_size = 1000
    batches = [work_items[i:i + chunk_size] for i in range(0, len(work_items), chunk_size)]

    max_workers = os.cpu_count()
    print(f"    Processing {len(work_items)} files in {len(batches)} batches using {max_workers} workers.", flush=True)
    start_time = time.time()

    acc_under = defaultdict(list)
    acc_over = defaultdict(list)
    acc_total = defaultdict(list)

    with ProcessPoolExecutor(max_workers=max_workers, initializer=init_worker, initargs=(gt_dict,)) as executor:
        futures = {executor.submit(process_file_batch, batch): batch for batch in batches}

        completed_files = 0
        for future in as_completed(futures):
            batch_results = future.result()
            for norm_res, veg_res, diff_u, diff_o, diff_t in batch_results:
                key = (norm_res, veg_res)
                acc_under[key].append(diff_u)
                acc_over[key].append(diff_o)
                acc_total[key].append(diff_t)

            completed_files += len(futures[future])
            if completed_files % 5000 < chunk_size:
                print(f"    Progress: {completed_files}/{len(work_items)} files processed...", flush=True)

    print(f"    Processing finished in {time.time() - start_time:.2f} seconds.", flush=True)

    matrix_under = pd.DataFrame(index=resolutions, columns=resolutions, dtype=float)
    matrix_over = pd.DataFrame(index=resolutions, columns=resolutions, dtype=float)
    matrix_total = pd.DataFrame(index=resolutions, columns=resolutions, dtype=float)

    for norm_res in resolutions:
        for veg_res in resolutions:
            key = (norm_res, veg_res)
            if key in acc_total and len(acc_total[key]) > 0:
                matrix_under.at[norm_res, veg_res] = np.mean(acc_under[key])
                matrix_over.at[norm_res, veg_res] = np.mean(acc_over[key])
                matrix_total.at[norm_res, veg_res] = np.mean(acc_total[key])
            else:
                matrix_under.at[norm_res, veg_res] = np.nan
                matrix_over.at[norm_res, veg_res] = np.nan
                matrix_total.at[norm_res, veg_res] = np.nan

    matrix_under.to_csv(os.path.join(combo_output_dir, 'diff_matrix_area_under_1m.csv'))
    matrix_over.to_csv(os.path.join(combo_output_dir, 'diff_matrix_area_over_1m.csv'))
    matrix_total.to_csv(os.path.join(combo_output_dir, 'diff_matrix_area_total.csv'))

    plot_heatmaps(combo_output_dir, class_label, metric)
    print(f"    Results saved to: {combo_output_dir}")


def run_all(gt_txt_path, metrics_root_folder, output_dir, resolutions, class_labels, metrics_list):
    overall_start = time.time()

    gt_dict = load_ground_truth_txt(gt_txt_path)
    if not gt_dict:
        print("Aborting: no ground truth loaded.")
        return

    for class_label in class_labels:
        for metric in metrics_list:
            run_validation_for_combo(gt_dict, metrics_root_folder, output_dir, resolutions, class_label, metric)

    elapsed_seconds = time.time() - overall_start
    hours, rem = divmod(elapsed_seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"\n{'='*70}")
    print(f"ALL VALIDATION COMPLETE")
    print(f"Total Elapsed Time: {int(hours):02d}h {int(minutes):02d}m {seconds:05.2f}s")
    print(f"{'='*70}")


# ==========================================
#               MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    # --- CONFIGURATION ---
    gt_txt_path = 'woody_plants_output_v2.txt'
    metrics_root_folder = "4_Vegetation_Metrics"   # Output from 2_compute_4_vegetation_metrics_v4.py
    output_dir = "5_Ground_Truth_Comparison"

    resolutions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    class_labels = ['class_4', 'class_5', 'class_4_5']
    metrics_to_compute = ['mean', 'median', 'max', 'p95']

    if not os.path.isdir(metrics_root_folder):
        print(f"ERROR: metrics_root_folder not found: {metrics_root_folder}")
        sys.exit(1)

    run_all(
        gt_txt_path,
        metrics_root_folder,
        output_dir,
        resolutions,
        class_labels,
        metrics_to_compute
    )
