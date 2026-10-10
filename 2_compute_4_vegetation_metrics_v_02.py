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

# ==========================================================================
# v_01 CHANGES (from v4)
# ==========================================================================
# ONE SUBSTANTIVE CHANGE: THE GRID. Everything else -- the metrics, the
# per-pixel grouping, the class configs, the output tree, the NaN-raster
# convention for plots with no points of a class, the sweep orchestration
# -- is v4 unchanged.
#
# v4 snapped each raster to the ABSOLUTE UTM grid:
#       min_x = floor(min(x) / veg_res) * veg_res
#       max_y = ceil (max(y) / veg_res) * veg_res
#   so pixel edges landed on multiples of veg_res in map coordinates, with
#   the plot centre falling wherever it happened to fall inside a pixel.
#   Because ~96% of the surveyed plot centres are exact multiples of 10 m,
#   that accidentally produced correct centre alignment at 0.1, 0.2, 0.5,
#   1, 2, 5 and 10 m -- and misalignment at 0.3, 0.4, 0.6, 0.7, 0.8, 0.9,
#   3, 4, 6, 7, 8 and 9 m, where it shifts which points fall in which
#   pixel and therefore every metric value.
#
# v_01 uses the same centre-symmetric grid as
# 1_dtm_gen_normalization_parallel_v_02.py:
#       centre  = bounding-box centre of ALL points in the tile,
#                 cx = (min_x + max_x)/2,  cy = (min_y + max_y)/2
#       n_x     = ceil(half_width / veg_res),  n_y likewise, min 1
#       raster  = 2*n_x by 2*n_y pixels, spanning cx +/- n_x*veg_res
#   Even cell count per axis, so the plot centre is always exactly the
#   shared CORNER of four pixels, and the pixel layout is point-symmetric
#   about it. A 20 x 20 m plot is 2x2 pixels at 10 m, 4x4 at 5 m, 20x20 at
#   1 m. Where veg_res does not divide the plot, padding is split equally
#   on both sides instead of piling up on one.
#
# THE GRID IS SHARED WITH THE DTM. Step 1 (v_01/v_02) drops no points --
# points beyond the pixel-centre domain are edge-clamped, not discarded --
# so a normalized cloud has exactly the point extent of its clipped tile,
# and the centre computed here is bit-identical to the one step 1 used.
# The metric raster and the DTM raster of the same plot at the same
# resolution therefore share an origin.
#
# The grid is still derived from the FULL point extent, not the
# target-class subset, so class_4 / class_5 / class_4_5 rasters for one
# plot and resolution remain identical in origin and size (v4's reason,
# unchanged) -- and now they also match across norm_res, which v4's
# data-dependent snapping did not guarantee.
#
# NOT CHANGED, BECAUSE THEY WERE NEVER WRONG HERE: v4's row order
# (rows = (max_y - y)/veg_res, row 0 = north) already agrees with
# from_origin(min_x, max_y, ...), so step 2 rasters were never vertically
# flipped, and v4 already treated grid cells as pixels rather than nodes,
# so there was no half-pixel offset either. Both of those bugs were
# specific to the step-1 DTM writer.
#
# ADDED: resume support, ported verbatim in spirit from
# 2_compute_4_vegetation_metrics_v9.py (is_valid_tif / filter_pending_files).
# It changes no output value; it only lets an interrupted sweep continue.
# A full 19 x 19 sweep is millions of files -- see the note in the
# CONFIGURATION block -- and restarting one from zero is not viable. Set
# resume = False for v4's exact behaviour (recompute everything).
# ==========================================================================


# ==========================================================================
# v_02 CHANGES (from v_01)
# ==========================================================================
# ONE ADDITION: a point-density check. For each plot, average point density
# (all points / point bounding-box area, in points per square metre) is
# computed once per normalization-resolution folder. Plots below
# MIN_POINT_DENSITY (see CONFIGURATION) are excluded from the whole sweep --
# no vegetation-metric rasters are written for them at any veg_res. This is
# a per-tile property, so it does not depend on veg_res and is measured
# once and reused across that folder's veg_res passes.
#
# A CSV of every plot's measured density and kept/excluded status is
# written to <base_output_folder>/plot_density_report.csv.
# ==========================================================================


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
#            POINT-DENSITY CHECK  (v_02)
# ==========================================

def compute_point_density(las_path):
    """
    Average point density of one tile: n_points / (point bbox area), in
    points per square metre. Reads only the LAS header (point count and
    x/y extent are both header fields), so this is fast even for millions
    of files.
    """
    with laspy.open(las_path) as fh:
        header = fh.header
        n_points = int(header.point_count)
        dx = float(header.maxs[0] - header.mins[0])
        dy = float(header.maxs[1] - header.mins[1])

    area = dx * dy
    if area <= 0 or not np.isfinite(area):
        return n_points, area, 0.0
    return n_points, area, n_points / area


def filter_by_density(files, input_path, min_density):
    """
    Splits `files` into (kept, records) where records is a list of
    (stem, n_points, area_m2, density, kept) for every file, used both to
    build the kept list and to write the CSV report.
    """
    kept = []
    records = []
    for f in files:
        stem = os.path.splitext(f)[0]
        try:
            n_points, area, density = compute_point_density(os.path.join(input_path, f))
        except Exception as e:
            print(f"    [!] Could not read {f} for density check: {e}")
            n_points, area, density = 0, 0.0, 0.0
        is_kept = density >= min_density
        if is_kept:
            kept.append(f)
        records.append((stem, n_points, area, density, is_kept))
    return kept, records


def write_density_report(base_output_folder, all_records):
    """plot_density_report.csv : norm_res, plot, n_points, area_m2, density_pts_per_m2, kept"""
    os.makedirs(base_output_folder, exist_ok=True)
    path = os.path.join(base_output_folder, "plot_density_report.csv")
    with open(path, 'w', encoding='utf-8') as f:
        f.write("norm_res,plot,n_points,area_m2,density_pts_per_m2,kept\n")
        for norm_res, stem, n_points, area, density, kept in all_records:
            f.write(f"{norm_res},{stem},{n_points},{area:.4f},{density:.6f},{int(kept)}\n")
    return path


# ==========================================
#        CENTRE-SYMMETRIC RASTER GRID
# ==========================================

def plot_geometric_centre(x, y):
    """
    The plot's geometric centre: the centre of the bounding box of its
    points. Identical definition to
    1_dtm_gen_normalization_parallel_v_02.py, so the metric raster and the
    DTM raster of a plot share the same anchor.
    """
    return (float(np.min(x) + np.max(x)) / 2.0, float(np.min(y) + np.max(y)) / 2.0)


def centre_symmetric_grid(centre, min_x, max_x, min_y, max_y, resolution):
    """
    Pixel grid holding n cells on EACH side of the plot centre along each
    axis: even cell counts, centre exactly on the shared corner of four
    pixels, layout point-symmetric about it. n is the smallest count that
    covers the point extent, n = ceil(half_extent / resolution), at least 1.

    One integer n is computed per axis and used for both sides, rather
    than flooring the low side and ceiling the high side independently:
    that makes the cell counts symmetric by construction instead of
    dependent on which way a float lands. With coordinates ~6.4e5 and
    resolution 0.1, (min_x - cx)/res is routinely one ULP from an integer,
    hence also the epsilon below.

    Returns (left, top, width, height) -- the NW corner of the raster and
    its size in pixels.
    """
    cx, cy = centre
    eps = 1e-9

    half_w = max(max_x - cx, cx - min_x)
    half_h = max(max_y - cy, cy - min_y)

    n_x = max(int(np.ceil(half_w / resolution - eps)), 1)
    n_y = max(int(np.ceil(half_h / resolution - eps)), 1)

    left = cx - n_x * resolution
    top = cy + n_y * resolution

    return left, top, 2 * n_x, 2 * n_y


# ==========================================
#              RESUME SUPPORT
# ==========================================

def is_valid_tif(path):
    """Existence isn't enough after an interrupted run -- try to actually
    read band 1 (same check as 2_compute_4_vegetation_metrics_v9.py)."""
    if not os.path.exists(path):
        return False
    try:
        with rasterio.open(path) as src:
            src.read(1)
        return True
    except Exception:
        return False


def is_file_done(args):
    """Worker: True if every expected metric output for this file already
    exists and is a valid, readable GeoTIFF."""
    file_stem, base_output_folder, class_label, norm_res, veg_res, metrics_list = args
    for metric in metrics_list:
        out_path = os.path.join(
            metric_output_dir(base_output_folder, class_label, norm_res, metric, veg_res),
            f"{file_stem}_{metric}.tif"
        )
        if not is_valid_tif(out_path):
            return (file_stem, False)
    return (file_stem, True)


def filter_pending_files(files, base_output_folder, class_label, norm_res, veg_res,
                         metrics_list, max_workers):
    """
    Subset of `files` still needing work for this (class, norm_res,
    veg_res) cell. Skips the per-file check entirely -- treating every
    file as pending -- when the cell's output folder is still empty, which
    is the common case for any cell an interrupted run never reached.
    """
    first_metric_dir = metric_output_dir(base_output_folder, class_label, norm_res,
                                         metrics_list[0], veg_res)
    with os.scandir(first_metric_dir) as it:
        has_any_output = any(it)
    if not has_any_output:
        return files

    stems = [os.path.splitext(f)[0] for f in files]
    tasks = [(stem, base_output_folder, class_label, norm_res, veg_res, metrics_list)
             for stem in stems]

    done_stems = set()
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = tqdm(executor.map(is_file_done, tasks, chunksize=20), total=len(tasks),
                       desc=f"    checking veg_res {veg_res}m", unit="file", leave=False)
        for file_stem, is_done in results:
            if is_done:
                done_stems.add(file_stem)

    return [f for f, stem in zip(files, stems) if stem not in done_stems]


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

        # np.asarray, as 3_ground_truth_comparison_v5_7.py does: laspy hands
        # back a ScaledArrayView, and a plain Python float minus such a view
        # raises TypeError (no __rsub__). v4 never hit this because its
        # min_x/max_y came out of np.floor/np.ceil as np.float64; the grid
        # origin here is derived from the plot centre and is a real float.
        x = np.asarray(las.x)
        y = np.asarray(las.y)
        z = np.asarray(las.z)
        classification = np.asarray(las.classification)

        if len(x) == 0:
            return {'success': True, 'file': filename, 'msg': "Skipped (Empty file)"}

        # Grid point-symmetric about the plot's own geometric centre, based
        # on the FULL point extent (not just the target-class subset) so
        # class_4 / class_5 / class_4_5 rasters for the same plot+resolution
        # share identical grid origin/size (v_01; see header).
        centre = plot_geometric_centre(x, y)
        min_x, top, width, height = centre_symmetric_grid(
            centre, np.min(x), np.max(x), np.min(y), np.max(y), veg_res
        )
        max_y = top

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
                   class_configs, crs, cpu_usage_percent, metrics_list, resume=True,
                   min_point_density=None):
    """
    class_configs : dict of {class_label: [target_classification_codes]}
    Produces: base_output_folder/{class}/res_{norm}/{metric}/res_{veg}/*.tif
    Total raster sets = len(resolutions)^2 * len(metrics_list) * len(class_configs)
    """
    start_time = time.time()

    total_cores = multiprocessing.cpu_count()
    max_workers = max(1, int(total_cores * cpu_usage_percent))
    print(f"Workers per batch: {max_workers} ({cpu_usage_percent*100:.0f}% of {total_cores} cores)")

    density_records = []   # (norm_res, stem, n_points, area_m2, density, kept) -- v_02

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

            # v_02: point-density check, once per norm_res folder (a
            # per-tile property that does not depend on veg_res).
            if min_point_density is not None:
                files, recs = filter_by_density(files, input_path, min_point_density)
                density_records.extend((norm_res,) + r for r in recs)
                n_total = len(recs)
                n_kept = len(files)
                print(f"    point-density check: {n_kept}/{n_total} plots >= "
                      f"{min_point_density} pts/m2 kept, {n_total - n_kept} excluded")
                if not files:
                    print(f"  [!] No plot passes the density threshold at norm_res {norm_res} m, skipping")
                    continue

            log_dir = os.path.join(base_output_folder, class_label, f"res_{res_str(norm_res)}")
            os.makedirs(log_dir, exist_ok=True)
            log_path = os.path.join(log_dir, "processing_log.txt")
            errors = []

            for veg_res in resolutions:
                # Pre-create output dirs in the main process to avoid worker race conditions
                # (write_geotiff also creates its own parent dir defensively, but doing it here
                # up front avoids every one of ~2626 workers racing to os.makedirs the same path).
                for metric in metrics_list:
                    os.makedirs(
                        metric_output_dir(base_output_folder, class_label, norm_res, metric, veg_res),
                        exist_ok=True
                    )

                # Resume support (v_01): drop files whose outputs are already
                # valid, so continuing an interrupted run doesn't redo them.
                if resume:
                    pending_files = filter_pending_files(
                        files, base_output_folder, class_label, norm_res, veg_res,
                        metrics_list, max_workers
                    )
                    skipped = len(files) - len(pending_files)
                    if not pending_files:
                        if skipped:
                            print(f"    veg_res {veg_res}m: all {skipped} file(s) already done, skipping")
                        continue
                    if skipped:
                        print(f"    veg_res {veg_res}m: {skipped}/{len(files)} already done, "
                              f"{len(pending_files)} pending")
                else:
                    pending_files = files

                tasks = [
                    (os.path.join(input_path, f), class_label, target_classes, norm_res, veg_res,
                     crs, metrics_list, base_output_folder)
                    for f in pending_files
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

    if min_point_density is not None and density_records:
        report_path = write_density_report(base_output_folder, density_records)
        seen = set()
        n_plots = 0
        n_kept = 0
        for norm_res, stem, n_points, area, density, kept in density_records:
            key = (norm_res, stem)
            if key in seen:
                continue
            seen.add(key)
            n_plots += 1
            n_kept += int(kept)
        print(f"\nPoint-density check: {n_kept}/{n_plots} (norm_res, plot) combinations "
              f">= {min_point_density} pts/m2")
        print(f"  full report: {report_path}")

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

    # Centre-aligned normalized clouds from step 1 v_02.
    normalized_parent_folder = "3_Normalized_Parallel_idw_v_01"
    base_output_folder = "4_Vegetation_Metrics_v_02"

    # Drives BOTH the norm_res loop and the veg_res loop, as in v4. The
    # full 19 values now that step 1 produces all of them.
    #
    # SIZE WARNING, so it is not a surprise mid-run: with class_4_5 alone,
    # 19 x 19 x 4 metrics = 1,444 raster sets x 2,626 tiles = ~3.8 MILLION
    # GeoTIFFs, ~50 GB (the fine veg_res dominate: a 0.1 m raster of a 20 m
    # plot is 200x200 float32 = 160 KB, and 0.1-0.3 m alone is ~90% of the
    # total). There is ~212 GB free on this disk, and it is ntfs-3g (FUSE),
    # where millions of small files are slow to create and slower to list.
    resolutions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
                   2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]

    target_crs = 'EPSG:25832'
    cpu_usage_percent = 0.90

    metrics_to_compute = ['mean', 'median', 'max', 'p95']

    # ASPRS classification: 4 = Medium Vegetation, 5 = High Vegetation.
    # class_4 and class_5 alone are not computed: 3_ground_truth_comparison_v5_7
    # only ever uses class_4_5, and the two extra classes tripled the output
    # (see the size note above). Add them back as [4] / [5] entries if a
    # per-stratum breakdown is ever needed.
    class_configs = {
        'class_4_5': [4, 5],
    }

    # Skip files whose 4 metric rasters already exist and open cleanly.
    # False = v4's behaviour (recompute everything).
    resume = True

    # --- POINT-DENSITY THRESHOLD (v_02) ---
    # Average point density = n_points / point-bbox area, in points per
    # square metre. A plot below this threshold is excluded from the whole
    # sweep at that norm_res: no vegetation-metric rasters are written for
    # it at any veg_res. Set to None for v_01's exact behaviour (no filter).
    min_point_density = 20.0

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
        metrics_to_compute,
        resume,
        min_point_density=min_point_density
    )
