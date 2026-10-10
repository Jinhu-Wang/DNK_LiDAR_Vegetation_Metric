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
    def tqdm(iterable, total=None, desc="", unit="", leave=True):
        return iterable

# Fix for Windows console encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# ==========================================================================
# v_02 CHANGES (from v_01)
# ==========================================================================
# MINIMUM-GROUND-POINT FILTER. A tile is now EXCLUDED unless it carries at
# least `min_ground_points` class-2 returns. One config value, one check;
# everything else below is v_01 unchanged.
#
# WHY: v_01 produced 2,627 outputs per resolution where v3/v4 produced
# 2,626. The extra one is 842959_20180928.laz -- 3 points in total, of
# which 2 are ground. v3/v4 never wrote it because cKDTree.query(k=6)
# against 2 points returns out-of-range indices and the worker died with
# an IndexError; v_01's k = min(6, n_ground) fix made it succeed, so a
# "DTM" interpolated from two ground points entered the pipeline. That is
# not data, it is an artefact of a crash being fixed, and it is now
# filtered out explicitly instead of by accident.
#
# THE THRESHOLD IS NOT DELICATE. Ground-point counts across the 2,641
# tiles: 14 tiles have 0, one tile (the above) has 2, and the next tile up
# has >= 50. Every cutoff from 3 to 50 therefore drops exactly the same
# single tile. 6 is used because that is the neighbourhood size the IDW /
# KNN interpolators ask for -- below it, the surface is not defined by the
# data at all. Raise or lower it in the CONFIGURATION block; nothing
# changes until you go past 50.
#
# The 14 zero-ground tiles that v_01 reported as "Skipped: no ground
# points" failures now fall under the same rule and are reported as
# EXCLUDED, not as errors -- they are ineligible input, not processing
# failures. Expected result: 15 excluded, 0 failed, 2,626 written per
# resolution, matching v3/v4.
#
# PURGE_EXCLUDED. If an excluded tile already has outputs on disk from an
# earlier run (v_01 wrote 38 such files for 842959_20180928: one DTM and
# one normalized cloud in each of the 19 resolutions), this version
# DELETES them, so that re-running over an existing output tree converges
# on exactly the intended file set instead of leaving stale files behind.
# Every deletion is printed and written to processing_log.txt. Set
# purge_excluded = False in the CONFIGURATION block to leave them alone.
#
# The k = min(6, n_ground) cap from v_01 is kept as a safety net, but with
# the default threshold it is unreachable.
# ==========================================================================

# ==========================================================================
# v_01 CHANGES (from v3 / v4) -- ALL STILL IN FORCE
# ==========================================================================
# THE DEFECT THIS FIXES -- GRID ALIGNMENT.
#   v3/v4 built the DTM grid with np.arange(min_x, max_x + res, res), where
#   min_x/max_x come from the tile's own ground points. The grid therefore
#   started at the lower-left of the DATA, which means:
#     (a) the plot centre fell at an arbitrary position inside a pixel,
#         different for every plot and every resolution;
#     (b) all padding piled up on one side. Measured on plot
#         816079_20180613 (20 x 20 m plot):
#             res  0.5 -> 41x41 = 20.5 m span   (right edge +10.51 m)
#             res  1.0 -> 21x21 = 21.0 m span   (right edge +11.01 m)
#             res  3.0 ->  8x8  = 24.0 m span   (right edge +14.01 m)
#             res  7.0 ->  4x4  = 28.0 m span   (right edge +18.01 m)
#             res 10.0 ->  3x3  = 30.0 m span   (right edge +20.01 m)
#         -- a 30 x 30 m DTM for a 20 x 20 m plot, one-sided.
#
#   v_01 anchors every grid on the plot's OWN GEOMETRIC CENTRE and makes
#   the pixel layout exactly point-symmetric about it:
#
#     centre  = bounding-box centre of the tile's points,
#               cx = (min_x + max_x)/2,  cy = (min_y + max_y)/2
#     n_x     = ceil(half_width  / res),  n_y = ceil(half_height / res)
#     raster  = 2*n_x by 2*n_y pixels, spanning cx +/- n_x*res, cy +/- n_y*res
#
#   Because the cell count per axis is 2n -- always EVEN, with n cells on
#   each side -- the centre is always exactly the shared CORNER of four
#   pixels, and the raster is its own image under a 180-degree rotation
#   about that corner. A 20 x 20 m plot gives 2x2 pixels at 10 m, 4x4 at
#   5 m, 20x20 at 1 m. Where res does not divide the plot (0.3, 3, 6, 7,
#   8, 9 m ...) padding is unavoidable, but it is split equally on both
#   sides instead of piling up on one.
#
#   No external table is read: the centre comes from the point cloud
#   itself, as requested. The bounding-box centre of ALL points is used
#   (not the centroid, not the ground-point bbox) -- over 300 tiles its
#   offset from the surveyed plot centre in filtered_plots_2018.csv is
#   median 0.005 m / p95 0.011 m / max 1.605 m, versus median 0.217 m for
#   the centroid and a 5.25 m worst case for the ground-only bbox. The
#   residual offset is the discreteness bias: points do not reach the exact
#   clip boundary (mean tile width 19.97 m, worst 16.75 m), so a tile that
#   was clipped short is centred on its points, not on the survey stake.
#   Same centre for every resolution of a given tile, so the resolution
#   sweep stays internally consistent.
#
# TWO RASTER-WRITING BUGS FIXED AT THE SAME TIME (they are inseparable from
# getting the alignment right in the OUTPUT file):
#   1. Y-FLIP. v3/v4 built grid_yy with np.meshgrid(x_vals, y_vals), whose
#      row 0 is min_y, then wrote it with from_origin(min_x, max_y, ...),
#      whose row 0 is max_y. Every GeoTIFF in 2_DTM_Parallel_idw is
#      upside down. Sampling the res-1.0 m DTM of 816079_20180613 at its
#      own ground points: RMSE 1.187 m as written, 0.080 m row-flipped.
#      v_01 writes grid_z[::-1] so row 0 is the northern row.
#      (The normalized point clouds were NEVER affected: normalization
#      went through RegularGridInterpolator on correctly ascending axes.)
#   2. HALF-PIXEL OFFSET. v3/v4 interpolated at grid NODES starting exactly
#      at min_x/min_y, then wrote those values with from_origin(min_x,...),
#      which declares them to be pixel UPPER-LEFT CORNERS -- so every value
#      was georeferenced res/2 away from where it was computed (5 m at
#      res 10). v_01 interpolates at PIXEL CENTRES and writes the pixel
#      EDGE as the origin, so raster value and location agree.
#
# EDGE HANDLING -- THE ONE DELIBERATE BEHAVIOUR CHANGE.
#   v3/v4 sampled the DTM for normalization with fill_value=np.nan and
#   DROPPED every point that fell outside the grid. With node-at-data-edge
#   grids almost nothing fell outside (measured: 0.032% of points lie
#   outside the ground-point bbox). With pixel-CENTRE sampling the
#   interpolation domain is inset by half a pixel on each side, and the
#   old rule would now discard the outer ring:
#       res  1.0 m ->  9.8% of all points outside the pixel-centre domain
#       res  5.0 m -> 44.4%
#       res 10.0 m -> 75.7%
#   Dropping three quarters of every tile at 10 m is obviously not an
#   option, so v_01 CLAMPS query coordinates into the domain before
#   sampling: points in the outer half-pixel ring take the DTM value of
#   the nearest pixel-centre row/column (edge replication -- what GDAL
#   does at raster edges for bilinear resampling). Interior points are
#   bilinearly interpolated exactly as before. The NaN drop is kept as a
#   safety net for genuinely undefined DTM values only.
#
# EXTENT NOW COMES FROM ALL POINTS, NOT ONLY GROUND POINTS. The symmetric
#   box must be built around the same point set that defines the centre,
#   or the layout is no longer exactly symmetric. This can only ENLARGE
#   the grid relative to v3/v4 (the ground points are a subset), so the
#   DTM still covers everything it covered before, and it now covers the
#   whole tile. The ground points themselves still define the SURFACE:
#   only class 2 is fed to the interpolator, exactly as before.
#
# ALSO: k for the IDW/KNN neighbour search is now min(6, n_ground) instead
#   of a hard 6, so a tile with fewer than 6 ground points no longer dies
#   with an IndexError. Same weights, same k=6 wherever 6 points exist.
#
# OUTPUT FOLDERS ARE NEW (2_DTM_Parallel_idw_v_01 / 3_Normalized_Parallel_idw_v_01)
#   so this run does not overwrite the existing misaligned results before
#   you have compared against them. 19 resolutions x 2,626 tiles is ~3 GB;
#   there is ~211 GB free. Point them at the old folders in the
#   CONFIGURATION block if you would rather replace in place.
#
# UNCHANGED FROM v3/v4: ground = ASPRS class 2; IDW with 6 nearest ground
#   points and 1/d^2 weights; the 'linear'/'rbf'/'knn' alternatives and the
#   nearest-neighbour fill for linear's NaN edges; bilinear sampling of the
#   DTM for normalization; normalized Z = Z - DTM; all LAS dimensions
#   carried across; the ProcessPoolExecutor layout, the per-resolution
#   processing_log.txt, the CRS and the timing report.
# ==========================================================================

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
        # k capped at the number of ground points available (v_01).
        k = min(6, len(ground_z))
        knn = KNeighborsRegressor(n_neighbors=k, weights='distance')
        knn.fit(ground_coords, ground_z)
        return knn.predict(target_coords)
    elif method == 'idw':
        tree = cKDTree(ground_coords)
        # k capped at the number of ground points available (v_01): querying
        # k=6 against fewer than 6 points returns out-of-range indices.
        k = min(6, len(ground_z))
        dist, idx = tree.query(target_coords, k=k)
        if k == 1:
            dist = dist[:, None]
            idx = idx[:, None]
        weights = 1.0 / (dist ** 2 + 1e-12)
        z_idw = np.sum(weights * ground_z[idx], axis=1) / np.sum(weights, axis=1)
        return z_idw
    else:
        raise ValueError(f"Unsupported method: {method}")


def plot_geometric_centre(x, y):
    """
    The plot's geometric centre: the centre of the bounding box of its
    points. Not the centroid -- the centroid follows point density (it sits
    a median 0.217 m from the surveyed centre over 300 tiles, versus
    0.005 m for this), and a box built around it would not be symmetric
    about the plot.
    """
    return (float(np.min(x) + np.max(x)) / 2.0, float(np.min(y) + np.max(y)) / 2.0)


def centre_symmetric_grid(centre, min_x, max_x, min_y, max_y, resolution):
    """
    Builds the point-symmetric raster geometry for one plot (v_01 core).

    The grid holds n cells on EACH side of the centre along each axis, so
    every axis has an even cell count (2n) and the centre is exactly the
    shared corner of four pixels; rotating the raster 180 degrees about
    that corner maps it onto itself. n is the smallest count that covers
    the point extent:  n = ceil(half_extent / resolution), at least 1.

    Computing n once and using -n / +n (rather than flooring and ceiling
    the two sides independently) makes the symmetry exact by construction
    instead of dependent on floating-point luck: with coordinates ~6.4e5
    and resolution 0.1, (min_x - cx)/res is routinely one ULP from an
    integer, hence also the epsilon below.

    Returns (left, top, x_centres, y_centres):
      left, top  -- NW corner of the raster, i.e. the GeoTIFF origin
      x_centres  -- pixel-centre X, ascending (west to east)
      y_centres  -- pixel-centre Y, ascending (south to north)
    """
    cx, cy = centre
    eps = 1e-9

    half_w = max(max_x - cx, cx - min_x)
    half_h = max(max_y - cy, cy - min_y)

    n_x = max(int(np.ceil(half_w / resolution - eps)), 1)
    n_y = max(int(np.ceil(half_h / resolution - eps)), 1)

    left = cx - n_x * resolution
    top = cy + n_y * resolution

    # Pixel centres at cx + (k + 0.5) * res for k = -n .. n-1
    x_centres = cx + (np.arange(-n_x, n_x) + 0.5) * resolution
    y_centres = cy + (np.arange(-n_y, n_y) + 0.5) * resolution

    return left, top, x_centres, y_centres


def generate_dtm_grid(x, y, z, grid_xx, grid_yy, method='linear'):
    """Generates the Z-grid (The Raster). Row 0 = southernmost (ascending Y)."""
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


def save_dtm_to_disk(grid_z, left, top, resolution, output_path, crs):
    """
    Saves the in-memory grid to a GeoTIFF.

    `grid_z` row 0 is the SOUTHERNMOST row (ascending Y, as built by
    np.meshgrid on ascending axes), while a GeoTIFF's row 0 is the
    NORTHERNMOST -- hence the [::-1]. v3/v4 omitted this, so every DTM
    they wrote is vertically mirrored. `left`/`top` are the NW corner of
    the raster (a pixel EDGE), matching the pixel-centre sampling.
    """
    transform = from_origin(left, top, resolution, resolution)

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
        dst.write(grid_z[::-1], 1)


def outputs_already_valid(dtm_path, las_path):
    """
    Resume support: True only if BOTH outputs exist and can actually be
    opened (existence alone is not enough after an interrupted run --
    same reasoning as is_valid_tif() in 2_compute_4_vegetation_metrics_v9.py).
    """
    if not (os.path.exists(dtm_path) and os.path.exists(las_path)):
        return False
    try:
        with rasterio.open(dtm_path) as src:
            src.read(1)
        with laspy.open(las_path) as fh:
            _ = fh.header.point_count
        return True
    except Exception:
        return False


# ==========================================
#           CORE WORKER FUNCTION
# ==========================================

def purge_outputs(dtm_tif_path, output_path):
    """
    Removes any outputs an earlier run left for a tile that is now
    excluded (v_02). Returns the list of paths actually deleted.
    """
    removed = []
    for path in (dtm_tif_path, output_path):
        if os.path.exists(path):
            try:
                os.remove(path)
                removed.append(path)
            except OSError:
                pass
    return removed


def process_single_file(args):
    """
    Worker function.
    Samples normalization height FROM the generated DTM grid, which is
    point-symmetric about the plot's own geometric centre (v_01), for
    every tile carrying at least `min_ground_points` ground returns (v_02).
    """
    (input_path, output_path, dtm_output_folder, resolution,
     method, crs, resume, min_ground_points, purge_excluded) = args
    filename = os.path.basename(input_path)

    dtm_tif_path = os.path.join(
        dtm_output_folder,
        os.path.splitext(filename)[0] + f"_dtm_{method}.tif"
    )

    try:
        las = laspy.read(input_path)
        x, y, z = las.x, las.y, las.z
        cls = las.classification

        # 1. Extract Ground Points (these alone define the SURFACE)
        ground_mask = cls == 2
        ground_x, ground_y, ground_z = x[ground_mask], y[ground_mask], z[ground_mask]

        # --- v_02 ELIGIBILITY FILTER -------------------------------------
        # Checked BEFORE the resume shortcut, so a tile that an earlier run
        # wrote is re-examined and, if it no longer qualifies, cleaned up
        # rather than silently kept.
        if len(ground_z) < min_ground_points:
            removed = purge_outputs(dtm_tif_path, output_path) if purge_excluded else []
            msg = (f"EXCLUDED: {len(ground_z)} ground point(s), "
                   f"minimum is {min_ground_points}.")
            if removed:
                msg += f" Deleted {len(removed)} stale output(s) from an earlier run."
            return {'success': True, 'file': filename, 'msg': msg, 'skipped': False,
                    'excluded': True, 'n_removed': len(removed),
                    'n_clamped': 0, 'n_points': 0}
        # -----------------------------------------------------------------

        if resume and outputs_already_valid(dtm_tif_path, output_path):
            return {'success': True, 'file': filename, 'msg': "Skipped (already done)",
                    'skipped': True, 'n_clamped': 0, 'n_points': 0}

        if len(x) == 0:
            return {'success': False, 'file': filename, 'skipped': False,
                    'n_clamped': 0, 'n_points': 0, 'msg': "Skipped: empty file."}

        # 2. Define the DTM Grid
        # Centre and extent both come from the tile's own points, so the
        # pixel layout is exactly symmetric about the plot centre and the
        # centre is the shared corner of four pixels (v_01).
        centre = plot_geometric_centre(x, y)

        left, top, grid_x_vals, grid_y_vals = centre_symmetric_grid(
            centre, np.min(x), np.max(x), np.min(y), np.max(y), resolution
        )

        # Pixel CENTRES, ascending in both axes (row 0 = southernmost).
        grid_xx, grid_yy = np.meshgrid(grid_x_vals, grid_y_vals)

        # 3. Generate the DTM Grid (Rasterize the Ground)
        # This creates the "Pixelated" surface based on the resolution
        dtm_grid_z = generate_dtm_grid(ground_x, ground_y, ground_z, grid_xx, grid_yy, method)

        # 4. Save the DTM GeoTIFF (flipped to north-up inside the writer)
        save_dtm_to_disk(dtm_grid_z, left, top, resolution, dtm_tif_path, crs)

        # 5. Create a Sampler from the DTM Grid
        # We use RegularGridInterpolator to sample the "Pixelated" height for every LAS point.
        # This ensures the normalization is based on the resolution-specific surface.

        # Note: RegularGridInterpolator expects (y_axis, x_axis) and ascending coords
        # grid_y_vals and grid_x_vals are already ascending from centre_symmetric_grid
        dtm_interpolator = RegularGridInterpolator(
            (grid_y_vals, grid_x_vals),
            dtm_grid_z,
            bounds_error=False,
            fill_value=np.nan,
            method='linear'  # Bilinear interpolation between pixels
        )

        # 6. Sample DTM Height for ALL LAS points
        # The pixel-centre lattice is inset by half a pixel from the raster
        # edge, so points in that outer ring are CLAMPED to the nearest
        # pixel centre (edge replication) instead of being dropped -- at
        # res 10 m that ring holds ~76% of the points (v_01; see header).
        x_query = np.clip(x, grid_x_vals[0], grid_x_vals[-1])
        y_query = np.clip(y, grid_y_vals[0], grid_y_vals[-1])
        n_clamped = int(np.count_nonzero((x_query != x) | (y_query != y)))

        dtm_z_sampled = dtm_interpolator((y_query, x_query))

        # 7. Normalize
        # NaN can no longer come from "outside the grid"; this is the
        # safety net for a genuinely undefined DTM value.
        valid = ~np.isnan(dtm_z_sampled)
        x, y, z = x[valid], y[valid], z[valid]
        norm_z = z - dtm_z_sampled[valid]
        cls = cls[valid]

        if len(norm_z) == 0:
            return {'success': False, 'file': filename, 'skipped': False,
                    'n_clamped': n_clamped, 'n_points': 0,
                    'msg': "Skipped: DTM undefined at every point."}

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
        return {'success': True, 'file': filename, 'msg': "Success", 'skipped': False,
                'n_clamped': n_clamped, 'n_points': int(len(valid))}

    except Exception as e:
        return {'success': False, 'file': filename, 'skipped': False,
                'n_clamped': 0, 'n_points': 0, 'msg': f"Error: {str(e)}"}


# ==========================================
#           BATCH & PARALLEL LOGIC
# ==========================================

def batch_normalize_parallel(input_folder, output_folder, dtm_output_folder,
                             resolution, method, crs, cpu_usage_percent, resume,
                             min_ground_points, purge_excluded):
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(dtm_output_folder, exist_ok=True)

    log_file_path = os.path.join(output_folder, "processing_log.txt")

    files = [f for f in os.listdir(input_folder) if f.lower().endswith(".laz")]
    tasks = []

    for file in files:
        in_path = os.path.join(input_folder, file)
        out_path = os.path.join(output_folder, file)
        tasks.append((in_path, out_path, dtm_output_folder, resolution, method, crs,
                      resume, min_ground_points, purge_excluded))

    # --- Dynamic Worker Calculation ---
    total_cores = multiprocessing.cpu_count()
    max_workers = int(total_cores * cpu_usage_percent)

    if max_workers < 1: max_workers = 1
    if max_workers > len(tasks): max_workers = len(tasks)

    print(f"  > Launching {max_workers} parallel workers (Targeting {cpu_usage_percent*100:.0f}% of {total_cores} cores)...")

    n_skipped = 0
    n_failed = 0
    n_excluded = 0
    n_removed = 0
    n_clamped = 0
    n_points = 0

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_single_file, task): task for task in tasks}

        with open(log_file_path, 'w') as log:
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(tasks), desc=f"Res {resolution}m", unit="file"):
                result = future.result()
                if not result['success']:
                    n_failed += 1
                    log.write(f"{result['file']}: {result['msg']}\n")
                elif result.get('excluded'):
                    n_excluded += 1
                    n_removed += result.get('n_removed', 0)
                    log.write(f"{result['file']}: {result['msg']}\n")
                    if result.get('n_removed'):
                        print(f"    [purge] {result['file']}: {result['msg']}")
                elif result.get('skipped'):
                    n_skipped += 1
                else:
                    n_clamped += result.get('n_clamped', 0)
                    n_points += result.get('n_points', 0)

    pct = (100.0 * n_clamped / n_points) if n_points else 0.0
    print(f"  > Done: {len(tasks) - n_failed - n_skipped - n_excluded} written, "
          f"{n_skipped} skipped (resume), {n_excluded} excluded (< {min_ground_points} ground points), "
          f"{n_failed} failed.")
    if n_removed:
        print(f"  > Purged {n_removed} stale output file(s) belonging to excluded tiles.")
    print(f"  > Edge-clamped points: {n_clamped:,}/{n_points:,} ({pct:.2f}%).")


def run_multiple_resolutions(input_folder, base_dtm_folder, base_output_folder, resolutions,
                             method='linear', crs='EPSG:25832', cpu_usage_percent=0.8, resume=True,
                             min_ground_points=6, purge_excluded=True):
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
            cpu_usage_percent=cpu_usage_percent,
            resume=resume,
            min_ground_points=min_ground_points,
            purge_excluded=purge_excluded
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

    # Pointed at the folders v_01 already filled. v_02 changes the eligible
    # TILE SET, not the geometry, so every tile v_01 wrote is still correct:
    # with resume = True this run re-checks eligibility, skips the 2,626
    # good tiles in seconds, and deletes the outputs of the excluded one.
    # Change to "..._v_02" to build a fresh tree from scratch instead.
    base_dtm_folder = "2_DTM_Parallel_idw_v_01"
    base_output_folder = "3_Normalized_Parallel_idw_v_01"

    # 1. Target Coordinate System
    target_crs = 'EPSG:25832'

    # 2. CPU Usage Settings
    cpu_usage_percent = 0.9

    # 3. Resolutions to process
    # The full 19-value sweep in one run: every old DTM is misaligned, so
    # there is nothing to reuse and no reason to split the list the way
    # v3 (0.1-1.0) and v4 (2-10) did.
    resolutions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
                   2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]

    # 4. Interpolation Method (used to create the DTM Grid)
    interpolation_method = 'idw' # Options: 'linear', 'rbf', 'knn', 'idw'

    # 5. Resume: skip tiles whose DTM and normalized cloud already exist
    #    AND open cleanly. Set False to force a full recompute.
    resume = True

    # 6. Eligibility filter (v_02): a tile needs at least this many class-2
    #    ground returns to get a DTM. 6 = the IDW/KNN neighbourhood size.
    #    Ground-point counts in this dataset are 0 (14 tiles), 2 (1 tile),
    #    then >= 50, so anything from 3 to 50 excludes the same 15 tiles.
    min_ground_points = 6

    # 7. Delete outputs an earlier run wrote for a now-excluded tile.
    #    Set False to leave stale files in place. THIS DELETES FILES from
    #    base_dtm_folder / base_output_folder -- only ever for tiles that
    #    fail the filter above (here: 842959_20180928, 38 files across the
    #    19 resolutions). Every deletion is printed and logged.
    purge_excluded = True

    # --- RUN ---
    run_multiple_resolutions(
        input_folder,
        base_dtm_folder,
        base_output_folder,
        resolutions,
        interpolation_method,
        target_crs,
        cpu_usage_percent,
        resume,
        min_ground_points,
        purge_excluded
    )
