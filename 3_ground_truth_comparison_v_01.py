import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib
import numpy as np
import pandas as pd
import rasterio
from rasterio.errors import RasterioIOError
from rasterio.windows import Window

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Try to import tqdm for progress bars
try:
    from tqdm import tqdm
except ImportError:
    class tqdm:
        """Minimal fallback: supports both `for x in tqdm(it)` and `with tqdm(total=N)`."""
        def __init__(self, iterable=None, total=None, desc="", unit="", leave=True):
            self.iterable = iterable
            self.total = total
            self.desc = desc
            self.n = 0

        def __iter__(self):
            for x in self.iterable:
                yield x
                self.update(1)

        def update(self, n=1):
            self.n += n
            if self.total:
                print(f"    {self.desc}: {self.n}/{self.total}", flush=True)

        def close(self):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            self.close()



RADIUS_M = 5.0
DISK_AREA_M2 = np.pi * RADIUS_M ** 2
MIN_CIRCLE_COVERAGE = 0.99
HEIGHT_THRESHOLD_M = 1.0


# ==========================================
#        GLOBAL MEMORY FOR WORKERS
# ==========================================
GLOBAL_CONFIG = {}


def init_worker(shared_config):
    global GLOBAL_CONFIG
    GLOBAL_CONFIG = shared_config


# ==========================================
#       EXACT PIXEL / DISK OVERLAP AREA
# ==========================================
# Verbatim from 3_ground_truth_comparison_v5_5..v5_7.

def _chord_primitive(x, r):
    """Antiderivative of sqrt(r^2 - x^2), with x clipped to [-r, r]."""
    x = np.clip(x, -r, r)
    return 0.5 * (x * np.sqrt(r * r - x * x) + r * r * np.arcsin(x / r))


def _area_above(x0, x1, h, r):
    """Area of disk(0, r) inside x0 <= X <= x1 and Y >= h, for h >= 0."""
    s = np.sqrt(np.maximum(r * r - h * h, 0.0))
    a = np.clip(x0, -s, s)
    b = np.clip(x1, -s, s)
    return (_chord_primitive(b, r) - _chord_primitive(a, r)) - h * (b - a)


def _area_at_or_above(x0, x1, y, r):
    """Area of disk(0, r) inside x0 <= X <= x1 and Y >= y, for any real y."""
    ay = np.abs(y)
    full_column = 2.0 * _area_above(x0, x1, 0.0, r)
    return np.where(y >= 0, _area_above(x0, x1, ay, r), full_column - _area_above(x0, x1, ay, r))


def disk_rect_area(x0, x1, y0, y1, r):
    """
    Exact area of the intersection of the axis-aligned rectangle(s)
    [x0, x1] x [y0, y1] with the disk of radius r centred at the origin.
    Coordinates are relative to the disk centre; inputs broadcast.
    """
    return _area_at_or_above(x0, x1, y0, r) - _area_at_or_above(x0, x1, y1, r)


# ==========================================
#              PATH CONVENTIONS
# ==========================================

def res_str(res):
    """'res_0_5' / 'res_10_0' -- identical to step 1 and step 2."""
    return str(res).replace('.', '_')


def metric_raster_path(metrics_parent, class_label, norm_res, metric, veg_res, stem):
    return os.path.join(
        metrics_parent, class_label, f"res_{res_str(norm_res)}", metric,
        f"res_{res_str(veg_res)}", f"{stem}_{metric}.tif"
    )


# ==========================================
#        DISK GEOMETRY OF ONE RASTER
# ==========================================

def raster_centre(transform, width, height):
    """Geometric centre of the raster's extent, in map coordinates."""
    cx = transform.c + 0.5 * width * transform.a
    cy = transform.f + 0.5 * height * transform.e      # transform.e is negative (north-up)
    return float(cx), float(cy)


def disk_overlap_weights(transform, width, height):
    """
    For a north-up raster, returns (window, weights, mask):

      window  : rasterio Window covering every pixel that can touch the disk
      mask    : bool array over that window, True where overlap > 0
      weights : 1-D float64, the exact pixel-disk overlap area of mask's True
                pixels, in map units squared, in row-major window order

    The disk centre is the raster's geometric centre (raster_centre()).

    The disk is centred on the raster, so the window is centred too; at
    veg_res 0.1 m it is 100 x 100 of a 200 x 200 raster.
    """
    res_x = transform.a
    res_y = -transform.e
    left = transform.c
    top = transform.f
    cx, cy = raster_centre(transform, width, height)

    col_lo = int(np.floor((cx - RADIUS_M - left) / res_x))
    col_hi = int(np.ceil((cx + RADIUS_M - left) / res_x))
    row_lo = int(np.floor((top - (cy + RADIUS_M)) / res_y))
    row_hi = int(np.ceil((top - (cy - RADIUS_M)) / res_y))

    col_lo = max(col_lo, 0)
    row_lo = max(row_lo, 0)
    col_hi = min(col_hi, width)
    row_hi = min(row_hi, height)
    if col_hi <= col_lo or row_hi <= row_lo:
        return None, np.empty(0), None

    cols = np.arange(col_lo, col_hi)
    rows = np.arange(row_lo, row_hi)
    x0 = left + cols * res_x - cx
    x1 = x0 + res_x
    y1 = top - rows * res_y - cy                 # pixel top edge
    y0 = y1 - res_y                              # pixel bottom edge

    w = disk_rect_area(x0[None, :], x1[None, :], y0[:, None], y1[:, None], RADIUS_M)
    mask = w > 0.0
    window = Window(col_lo, row_lo, col_hi - col_lo, row_hi - row_lo)
    return window, w[mask].astype(np.float64), mask


# ==========================================
#                 WORKER
# ==========================================

def save_checkpoint(cache_path, areas, done, stems):
    """
    Atomic checkpoint write. np.savez leaves a truncated, unreadable file if
    the run is killed mid-flush, which would throw away every cell computed
    so far; writing to a sibling and renaming cannot.
    """
    tmp = cache_path + '.tmp.npz'          # np.savez appends .npz unless present
    np.savez(tmp, areas=areas, done=done, stems=np.array(stems))
    os.replace(tmp, cache_path)


def process_task_batch(batch):
    """
    batch: list of (plot_idx, veg_idx). For each, reads the
    n_norm x n_metric rasters of that (plot, veg_res) and returns the
    ">= 1 m" area clipped to the disk.

    Returns (results, n_missing, n_failed) where results is a list of
    (plot_idx, veg_idx, areas[n_norm, n_metric] float32, n_failed_in_cell).
    A cell with n_failed_in_cell > 0 lost at least one raster -- absent or
    unreadable -- and is left un-done so the next run retries it. Absent is
    not treated as permanent: step 2's tree is complete, so a miss on this
    ntfs-3g mount is more likely a transient read than a file that is not
    there, and a second run distinguishes the two (the count repeats exactly).
    """
    cfg = GLOBAL_CONFIG
    stems = cfg['stems']
    resolutions = cfg['resolutions']
    metrics = cfg['metrics']
    class_label = cfg['class_label']
    metrics_parent = cfg['metrics_parent']
    n_norm = len(resolutions)
    n_metric = len(metrics)

    results = []
    n_missing = 0
    n_failed = 0

    for plot_idx, veg_idx in batch:
        stem = stems[plot_idx]
        veg_res = resolutions[veg_idx]
        areas = np.full((n_norm, n_metric), np.nan, dtype=np.float32)

        window = None
        weights = None
        mask = None
        cell_failed = 0

        for i_norm, norm_res in enumerate(resolutions):
            for i_metric, metric in enumerate(metrics):
                path = metric_raster_path(metrics_parent, class_label, norm_res,
                                          metric, veg_res, stem)
                try:
                    with rasterio.open(path) as src:
                        if window is None:
                            window, weights, mask = disk_overlap_weights(
                                src.transform, src.width, src.height)
                            if window is None:
                                raise ValueError("disk does not intersect raster")
                        values = src.read(1, window=window)[mask]
                except RasterioIOError:
                    # rasterio raises RasterioIOError -- an OSError, but NOT a
                    # FileNotFoundError -- both for an absent file and for a
                    # read error, so the two are told apart by the filesystem.
                    if os.path.exists(path):
                        n_failed += 1
                    else:
                        n_missing += 1
                    cell_failed += 1
                    continue
                except Exception:
                    n_failed += 1
                    cell_failed += 1
                    continue

                areas[i_norm, i_metric] = weights[values >= HEIGHT_THRESHOLD_M].sum()

        results.append((plot_idx, veg_idx, areas, cell_failed))

    return results, n_missing, n_failed


# ==========================================
#              GROUND TRUTH
# ==========================================

def load_ground_truth_txt(filename):
    """
    woody_plants_output_v2.txt: '<plot>.laz under over total', space separated,
    no header. Returns {'<plot>': over_m2}, keyed by plot stem. `over` -- the
    woody cover area in m2 above 1 m inside the 5 m circle -- is the only
    column this comparison uses; `under` and `total` are read and dropped.
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

    df = pd.read_csv(file_path, sep=' ', names=['filename', 'under', 'over', 'total'], header=None)
    gt_dict = {}
    for _, row in df.iterrows():
        stem = os.path.basename(str(row['filename']).strip())
        if stem.lower().endswith('.laz'):
            stem = stem[:-4]
        gt_dict[stem] = float(row['over'])
    print(f"Loaded {len(gt_dict)} rows. First key: '{next(iter(gt_dict))}'", flush=True)
    return gt_dict


def load_plot_centres(plots_csv):
    """Surveyed centres, keyed '<aktId>_<dato>.laz'. DIAGNOSTIC ONLY in v_01."""
    if not os.path.isfile(plots_csv):
        print(f"    [!] Plot centre table not found ({plots_csv}); centre-offset diagnostic skipped.")
        return {}
    df = pd.read_csv(plots_csv)
    keys = df['aktId'].astype(str) + '_' + df['dato'].astype(str) + '.laz'
    centres = dict(zip(keys, zip(df['UTM_X'].astype(float), df['UTM_Y'].astype(float))))
    print(f"Loaded {len(centres)} surveyed plot centres from {plots_csv} (diagnostic only)", flush=True)
    return centres


# ==========================================
#          INVENTORY AND ELIGIBILITY
# ==========================================

def scan_plots(metrics_parent, class_label, resolutions, metrics, gt_dict):
    """
    Plots that have a raster in the reference cell (finest norm_res, first
    metric, finest veg_res) AND a ground-truth row. Returns sorted stems.
    """
    ref_dir = os.path.join(metrics_parent, class_label, f"res_{res_str(resolutions[0])}",
                           metrics[0], f"res_{res_str(resolutions[0])}")
    if not os.path.isdir(ref_dir):
        print(f"ERROR: reference cell not found: {ref_dir}")
        return []
    suffix = f"_{metrics[0]}.tif"
    on_disk = {f[:-len(suffix)] for f in os.listdir(ref_dir) if f.endswith(suffix)}
    gt_stems = set(gt_dict)

    stems = sorted(on_disk & gt_stems)
    print(f"    Rasters in reference cell: {len(on_disk)}   ground-truth rows: {len(gt_stems)}   "
          f"both: {len(stems)}", flush=True)
    missing_gt = sorted(on_disk - gt_stems)
    missing_ras = sorted(gt_stems - on_disk)
    if missing_gt:
        print(f"    [!] {len(missing_gt)} raster(s) without ground truth, excluded: {missing_gt[:5]}")
    if missing_ras:
        print(f"    [!] {len(missing_ras)} ground-truth row(s) without rasters, excluded: {missing_ras[:5]}")
    return stems


def survey_geometry(metrics_parent, class_label, resolutions, metrics, stems, surveyed_centres):
    """
    Reads the header of each plot's finest-veg_res raster once: disk centre,
    coverage of the disk by the raster extent, and the offset from the
    surveyed centre. Returns a DataFrame indexed like `stems`.
    """
    rows = []
    for stem in tqdm(stems, desc="    raster headers", unit="file", leave=False):
        path = metric_raster_path(metrics_parent, class_label, resolutions[0],
                                  metrics[0], resolutions[0], stem)
        with rasterio.open(path) as src:
            t, width, height = src.transform, src.width, src.height
        cx, cy = raster_centre(t, width, height)
        # Fraction of the disk inside the raster's own extent (t.e < 0, north-up).
        cov = float(disk_rect_area(t.c - cx, t.c + width * t.a - cx,
                                   t.f + height * t.e - cy, t.f - cy, RADIUS_M)) / DISK_AREA_M2
        sx, sy = surveyed_centres.get(stem + '.laz', (np.nan, np.nan))
        rows.append((stem, cx, cy, sx, sy, np.hypot(cx - sx, cy - sy), cov))
    return pd.DataFrame(rows, columns=['plot', 'raster_cx', 'raster_cy', 'survey_x', 'survey_y',
                                       'centre_offset_m', 'circle_coverage'])


def check_centre_invariance(metrics_parent, class_label, resolutions, metrics, stems, n_sample=25):
    """
    The disk centre must not depend on norm_res, veg_res or metric. Checks a
    random sample of plots across the corners and middle of the sweep.
    """
    rng = np.random.default_rng(0)
    sample = [stems[i] for i in rng.choice(len(stems), size=min(n_sample, len(stems)), replace=False)]
    probe_res = [resolutions[0], resolutions[len(resolutions) // 2], resolutions[-1]]
    worst = 0.0
    for stem in sample:
        seen = []
        for norm_res in probe_res:
            for veg_res in probe_res:
                for metric in (metrics[0], metrics[-1]):
                    path = metric_raster_path(metrics_parent, class_label, norm_res,
                                              metric, veg_res, stem)
                    with rasterio.open(path) as src:
                        seen.append(raster_centre(src.transform, src.width, src.height))
        arr = np.asarray(seen)
        worst = max(worst, float(np.abs(arr - arr[0]).max()))
    print(f"    Disk-centre invariance over {len(sample)} plots x {len(probe_res)**2} resolution pairs "
          f"x 2 metrics: max deviation {worst:.3e} m", flush=True)
    return worst


# ==========================================
#      PER-CELL STATISTIC DEFINITIONS
# ==========================================
STATS = ['mean', 'std', 'min', 'max', 'range']
STAT_LABELS = {'mean': 'Mean', 'std': 'Std Dev', 'min': 'Min', 'max': 'Max', 'range': 'Range'}
SIGNED_STATS = {'mean', 'min', 'max'}


def plot_heatmaps(output_folder, class_label, metric):
    """One '>1m' heatmap per statistic (mean/std/min/max/range)."""
    for stat in STATS:
        csv_path = os.path.join(output_folder, f'diff_matrix_area_over_1m_{stat}.csv')
        if not os.path.exists(csv_path):
            continue
        try:
            df = pd.read_csv(csv_path, index_col=0)
            if df.isnull().all().all():
                continue

            n = len(df)
            plt.figure(figsize=(max(10, n * 1.0), max(8, n * 0.8)))
            if stat in SIGNED_STATS:
                sns.heatmap(df, annot=True, fmt=".2f", cmap="RdBu", center=0, linewidths=.5,
                            annot_kws={"size": 6})
            else:
                sns.heatmap(df, annot=True, fmt=".2f", cmap="viridis", linewidths=.5,
                            annot_kws={"size": 6})

            # Normalization resolution increases upward (smallest at the bottom).
            plt.gca().invert_yaxis()
            plt.title(f'{class_label} / {metric} -- Woody Cover (>1m)\n'
                      f'{STAT_LABELS[stat]} of Difference (LiDAR - GT) [m²]', fontsize=14)
            plt.xlabel('Vegetation Metric Resolution (m)', fontsize=12)
            plt.ylabel('Normalization Resolution (m)', fontsize=12)

            out_name = f'heatmap_diff_matrix_area_over_1m_{stat}.png'
            plt.savefig(os.path.join(output_folder, out_name), dpi=300, bbox_inches='tight')
            plt.close()
            print(f"    Saved plot: {out_name}", flush=True)
        except Exception as exc:
            print(f"    [!] Heatmap {stat} failed: {exc}", flush=True)


# ==========================================
#                 OUTPUTS
# ==========================================

def write_combo_outputs(areas, gt_over, stems, resolutions, output_dir, class_label, metric, i_metric):
    """
    Per-plot table, the 5 per-cell diff matrices and their heatmaps for ONE
    class + metric. `areas` is (n_veg, n_plot, n_norm, n_metric).
    """
    combo_output_dir = os.path.join(output_dir, class_label, metric)
    os.makedirs(combo_output_dir, exist_ok=True)

    n_veg, n_plot, n_norm = areas.shape[0], areas.shape[1], areas.shape[2]
    block = areas[:, :, :, i_metric]                      # (n_veg, n_plot, n_norm)
    lidar = np.transpose(block, (2, 0, 1)).reshape(-1)    # norm-major, then veg, then plot
    gt_col = np.tile(gt_over, n_norm * n_veg)
    per_plot = pd.DataFrame({
        'norm_res': np.repeat(resolutions, n_veg * n_plot),
        'veg_res': np.tile(np.repeat(resolutions, n_plot), n_norm),
        'plot': np.tile(np.asarray(stems, dtype=object), n_norm * n_veg),
        'lidar_area_m2': lidar.astype(np.float64),
        'gt_area_m2': gt_col,
    })
    per_plot['diff_m2'] = per_plot['lidar_area_m2'] - per_plot['gt_area_m2']
    per_plot['plot'] = per_plot['plot'].astype(str) + '.laz'
    per_plot.to_csv(os.path.join(combo_output_dir, 'per_plot_area_over_1m.csv.gz'), index=False)

    matrices = {stat: pd.DataFrame(index=resolutions, columns=resolutions, dtype=float)
                for stat in STATS}
    n_cells_incomplete = 0
    for i_norm, norm_res in enumerate(resolutions):
        for i_veg, veg_res in enumerate(resolutions):
            diff = block[i_veg, :, i_norm] - gt_over
            diff = diff[np.isfinite(diff)]
            if diff.size < n_plot:
                n_cells_incomplete += 1
            if diff.size:
                matrices['mean'].at[norm_res, veg_res] = np.mean(diff)
                matrices['std'].at[norm_res, veg_res] = np.std(diff)
                matrices['min'].at[norm_res, veg_res] = np.min(diff)
                matrices['max'].at[norm_res, veg_res] = np.max(diff)
                matrices['range'].at[norm_res, veg_res] = np.max(diff) - np.min(diff)
            else:
                for stat in STATS:
                    matrices[stat].at[norm_res, veg_res] = np.nan

    for stat in STATS:
        matrices[stat].to_csv(os.path.join(combo_output_dir, f'diff_matrix_area_over_1m_{stat}.csv'))

    finite = np.isfinite(block)
    print(f"\n    {class_label}/{metric}: {len(per_plot)} rows  |  "
          f"{finite.sum()}/{block.size} areas computed  |  "
          f"{n_cells_incomplete}/{n_norm * n_veg} cells incomplete  |  "
          f"max LiDAR area {np.nanmax(block):.3f} m² (disk area {DISK_AREA_M2:.3f} m²)", flush=True)
    plot_heatmaps(combo_output_dir, class_label, metric)
    print(f"    Results saved to: {combo_output_dir}")


# ==========================================
#                SELF-TEST
# ==========================================

def _mc_overlap(x0, x1, y0, y1, r, n=400000, seed=0):
    """Monte-Carlo area of rect ∩ disk(0, r), for checking disk_rect_area."""
    rng = np.random.default_rng(seed)
    px = rng.uniform(x0, x1, n)
    py = rng.uniform(y0, y1, n)
    inside = (px * px + py * py) <= r * r
    return (x1 - x0) * (y1 - y0) * inside.mean()


def run_self_test(metrics_parent, class_label, resolutions, metrics, stems, n_plots=6):
    """
    Checks, in order:
      A  disk_rect_area against a closed form and against Monte-Carlo;
      B  the weights of a whole raster sum to the disk area;
      C  an independent, per-pixel reimplementation of the clipped area
         reproduces the worker's value bit-for-bit;
      D  every area lies in [0, disk area];
      E  a plot's disk centre is the same for every norm_res / veg_res / metric.
    """
    print(f"\n{'='*70}\nSELF-TEST\n{'='*70}", flush=True)
    ok = True

    # ---- A: exact overlap ------------------------------------------------
    full = disk_rect_area(-9.0, 9.0, -9.0, 9.0, RADIUS_M)
    err_full = abs(full - DISK_AREA_M2)
    half = disk_rect_area(-9.0, 0.0, -9.0, 9.0, RADIUS_M)
    quarter = disk_rect_area(0.0, 9.0, 0.0, 9.0, RADIUS_M)
    print(f"  A1 whole disk      : {full:.9f} vs {DISK_AREA_M2:.9f}   err {err_full:.2e}")
    print(f"  A2 half / quarter  : {half:.9f} (exp {DISK_AREA_M2/2:.9f})  "
          f"{quarter:.9f} (exp {DISK_AREA_M2/4:.9f})")
    ok &= err_full < 1e-9 and abs(half - DISK_AREA_M2 / 2) < 1e-9 and abs(quarter - DISK_AREA_M2 / 4) < 1e-9

    rng = np.random.default_rng(1)
    worst_mc = 0.0
    for k in range(12):
        x0, y0 = rng.uniform(-8, 4, 2)
        x1, y1 = x0 + rng.uniform(0.1, 9), y0 + rng.uniform(0.1, 9)
        exact = float(disk_rect_area(x0, x1, y0, y1, RADIUS_M))
        mc = _mc_overlap(x0, x1, y0, y1, RADIUS_M, seed=k)
        worst_mc = max(worst_mc, abs(exact - mc))
    print(f"  A3 vs Monte-Carlo  : worst |exact - MC| = {worst_mc:.4f} m² over 12 random rectangles "
          f"(MC noise ~0.01 m²)")
    ok &= worst_mc < 0.05

    # ---- B/C/D: against the rasters --------------------------------------
    sample = stems[:: max(len(stems) // n_plots, 1)][:n_plots]
    probe_veg = [resolutions[0], 0.3, 1.0, 7.0, resolutions[-1]]
    probe_veg = [v for v in probe_veg if v in resolutions]
    worst_wsum = 0.0
    worst_indep = 0.0
    n_checked = 0
    for stem in sample:
        for veg_res in probe_veg:
            path = metric_raster_path(metrics_parent, class_label, resolutions[0],
                                      metrics[0], veg_res, stem)
            with rasterio.open(path) as src:
                transform, width, height = src.transform, src.width, src.height
                window, weights, mask = disk_overlap_weights(transform, width, height)
                values = src.read(1, window=window)[mask]
                full_arr = src.read(1)

            # B: the weights must account for the whole disk
            worst_wsum = max(worst_wsum, abs(weights.sum() - DISK_AREA_M2))

            area = weights[values >= HEIGHT_THRESHOLD_M].sum()

            # C: independent per-pixel reimplementation, no windowing, plain loops
            cx, cy = raster_centre(transform, width, height)
            res_x, res_y = transform.a, -transform.e
            indep = 0.0
            for r in range(height):
                y_top = transform.f - r * res_y - cy
                for c in range(width):
                    v = full_arr[r, c]
                    if not (v >= HEIGHT_THRESHOLD_M):
                        continue
                    x_left = transform.c + c * res_x - cx
                    indep += float(disk_rect_area(x_left, x_left + res_x,
                                                  y_top - res_y, y_top, RADIUS_M))
            worst_indep = max(worst_indep, abs(indep - area))

            # D: range
            if not (-1e-9 <= area <= DISK_AREA_M2 + 1e-9):
                print(f"  [!] D area out of range: {stem} veg_res {veg_res}: {area}")
                ok = False
            n_checked += 1

    print(f"  B  sum of weights == disk area      : worst err {worst_wsum:.2e} m²  "
          f"({n_checked} rasters)")
    print(f"  C  independent reimplementation     : worst |Δ| {worst_indep:.2e} m²  "
          f"({n_checked} rasters)")
    print(f"  D  0 <= area <= {DISK_AREA_M2:.3f} m²          : "
          f"{'pass' if ok else 'FAIL'}")
    ok &= worst_wsum < 1e-6 and worst_indep < 1e-6

    # ---- E: centre invariance -------------------------------------------
    dev = check_centre_invariance(metrics_parent, class_label, resolutions, metrics, stems, n_sample=8)
    ok &= dev < 1e-6

    print(f"{'='*70}\nSELF-TEST {'PASSED' if ok else 'FAILED'}\n{'='*70}", flush=True)
    return ok


# ==========================================
#           CORE LOGIC WRAPPER
# ==========================================

def run_all(gt_txt_path, plots_csv, metrics_parent, output_dir, resolutions,
            class_label, metrics_list, checkpoint_seconds, batch_size, max_workers,
            self_test):
    overall_start = time.time()

    print(f"\n{'#'*70}")
    print("NOTE on the '>1m' comparison in v_01:")
    print(f"  The {RADIUS_M:g} m-radius circle is centred on the GEOMETRIC CENTRE of each")
    print("  vegetation-metric raster (= the plot's bounding-box centre, the same centre")
    print("  steps 1 and 2 build their grids around). Pixel values are read from")
    print(f"  {metrics_parent} as written by step 2; every pixel with a value >= "
          f"{HEIGHT_THRESHOLD_M:g} m")
    print(f"  contributes its EXACT overlap area with the circle (<= {DISK_AREA_M2:.2f} m² total).")
    print("  Points outside the circle still influence a pixel's value -- the rasters are")
    print("  compared as published. This is v5_6 semantics, NOT v5_7's clipped-points ones.")
    print(f"{'#'*70}", flush=True)

    gt_dict = load_ground_truth_txt(gt_txt_path)
    if not gt_dict:
        print("Aborting: no ground truth loaded.")
        return
    surveyed = load_plot_centres(plots_csv)

    stems = scan_plots(metrics_parent, class_label, resolutions, metrics_list, gt_dict)
    if not stems:
        print("Aborting: no plots to process.")
        return

    os.makedirs(output_dir, exist_ok=True)

    if self_test and not run_self_test(metrics_parent, class_label, resolutions, metrics_list, stems):
        print("Aborting: self-test failed.")
        return

    # ---- geometry, eligibility, diagnostics ------------------------------
    print("\n    Reading raster headers (disk centre, coverage, offset from surveyed centre)...",
          flush=True)
    geom = survey_geometry(metrics_parent, class_label, resolutions, metrics_list, stems, surveyed)
    # One table: disk centre, its offset from the surveyed centre, and coverage.
    geom.sort_values('circle_coverage').to_csv(
        os.path.join(output_dir, 'plot_circle_coverage.csv'), index=False)

    off = geom['centre_offset_m'].to_numpy(dtype=float)
    off = off[np.isfinite(off)]
    if off.size:
        print(f"    Raster centre vs surveyed centre (diagnostic only): median {np.median(off):.4f} m, "
              f"p95 {np.percentile(off, 95):.4f} m, max {off.max():.4f} m "
              f"({(off > 0.5).sum()} plot(s) > 0.5 m)", flush=True)

    eligible = geom['circle_coverage'] >= MIN_CIRCLE_COVERAGE
    n_excluded = int((~eligible).sum())
    stems = geom.loc[eligible, 'plot'].tolist()
    print(f"    Eligible plots: {len(stems)} / {len(geom)}  ({n_excluded} excluded: raster extent "
          f"covers < {MIN_CIRCLE_COVERAGE:.0%} of the circle)", flush=True)
    if not stems:
        print("Aborting: no eligible plots.")
        return

    gt_over = np.array([gt_dict[s] for s in stems], dtype=np.float64)

    n_res = len(resolutions)
    n_plot = len(stems)
    n_metric = len(metrics_list)

    # ---- checkpoint ------------------------------------------------------
    cache_dir = os.path.join(output_dir, '_cache')
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, 'areas.npz')

    areas = np.full((n_res, n_plot, n_res, n_metric), np.nan, dtype=np.float32)   # veg, plot, norm, metric
    done = np.zeros((n_res, n_plot), dtype=bool)
    if os.path.isfile(cache_path):
        try:
            cached = np.load(cache_path, allow_pickle=True)
            if (list(cached['stems']) == stems and cached['areas'].shape == areas.shape):
                areas = cached['areas']
                done = cached['done']
                print(f"    Resuming from checkpoint: {int(done.sum())}/{done.size} "
                      f"(plot, veg_res) cells already done.", flush=True)
            else:
                print("    [!] Checkpoint does not match this configuration; starting fresh.", flush=True)
        except Exception as exc:
            print(f"    [!] Could not read checkpoint ({exc}); starting fresh.", flush=True)

    tasks = [(p, v) for v in range(n_res) for p in range(n_plot) if not done[v, p]]
    if not tasks:
        print("    All cells already computed; writing outputs from the checkpoint.", flush=True)
    else:
        config = {'stems': stems, 'resolutions': list(resolutions), 'metrics': list(metrics_list),
                  'class_label': class_label, 'metrics_parent': metrics_parent}
        batches = [tasks[i:i + batch_size] for i in range(0, len(tasks), batch_size)]
        n_rasters = len(tasks) * n_res * n_metric
        print(f"\n    Processing {len(tasks)} (plot, veg_res) cells in {len(batches)} batches using "
              f"{max_workers} workers -- {n_rasters:,} rasters "
              f"({n_res} norm_res x {n_metric} metrics each).", flush=True)

        start_time = time.time()
        last_flush = start_time
        n_missing_total = 0
        n_failed_total = 0

        with ProcessPoolExecutor(max_workers=max_workers, initializer=init_worker,
                                 initargs=(config,)) as executor:
            futures = {executor.submit(process_task_batch, b): b for b in batches}
            with tqdm(total=len(tasks), desc="    cells", unit="cell", leave=False) as pbar:
                for future in as_completed(futures):
                    results, n_missing, n_failed = future.result()
                    n_missing_total += n_missing
                    n_failed_total += n_failed
                    for plot_idx, veg_idx, cell, cell_failed in results:
                        areas[veg_idx, plot_idx] = cell
                        done[veg_idx, plot_idx] = (cell_failed == 0)
                    pbar.update(len(futures[future]))

                    now = time.time()
                    if now - last_flush >= checkpoint_seconds:
                        save_checkpoint(cache_path, areas, done, stems)
                        last_flush = now

        save_checkpoint(cache_path, areas, done, stems)
        elapsed = time.time() - start_time
        print(f"    Processing finished in {elapsed:.2f} s "
              f"({n_rasters / max(elapsed, 1e-9):,.0f} rasters/s).  "
              f"Missing rasters: {n_missing_total}   failed reads: {n_failed_total}", flush=True)
        n_retry = int((~done).sum())
        if n_retry:
            print(f"    [!] {n_retry} cell(s) lost at least one raster (missing or unreadable) "
                  f"and were left un-done; re-run to retry exactly those.", flush=True)

    # ---- outputs ---------------------------------------------------------
    for i_metric, metric in enumerate(metrics_list):
        write_combo_outputs(areas, gt_over, stems, resolutions, output_dir,
                            class_label, metric, i_metric)

    elapsed_seconds = time.time() - overall_start
    hours, rem = divmod(elapsed_seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"\n{'='*70}")
    print("ALL VALIDATION COMPLETE")
    print(f"Total Elapsed Time: {int(hours):02d}h {int(minutes):02d}m {seconds:05.2f}s")
    print(f"{'='*70}")


# ==========================================
#               MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    # --- CONFIGURATION ---
    gt_txt_path = 'woody_plants_output_v2.txt'
    plots_csv = os.path.join('0_Shapefiles_2018', 'filtered_plots_2018.csv')  # diagnostic only
    metrics_parent = "/directory/to/LiDAR-derived_metrics/"      # rasters written by step 2
    output_dir = "/directory/to/output"

    # True = only redraw heatmaps from the CSVs already in output_dir.
    replot_only = False
    # True = run the geometry / area self-test before processing.
    self_test = True

    # Full 19-value grid -- drives both the norm_res and veg_res axes.
    resolutions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
                   2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]

    class_label = 'class_4_5'
    metrics_to_compute = ['mean', 'median', 'max', 'p95']

    batch_size = 8                 # (plot, veg_res) cells per task
    checkpoint_seconds = 300
    max_workers = os.cpu_count()

    if replot_only:
        for metric in metrics_to_compute:
            combo_dir = os.path.join(output_dir, class_label, metric)
            if os.path.isdir(combo_dir):
                print(f"Replotting {combo_dir}", flush=True)
                plot_heatmaps(combo_dir, class_label, metric)
        sys.exit(0)

    if not os.path.isdir(metrics_parent):
        print(f"ERROR: folder not found: {metrics_parent}")
        sys.exit(1)

    run_all(gt_txt_path, plots_csv, metrics_parent, output_dir, resolutions,
            class_label, metrics_to_compute, checkpoint_seconds, batch_size,
            max_workers, self_test)
