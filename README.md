# DNK LiDAR Vegetation Metric

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub issues](https://img.shields.io/github/issues/Jinhu-Wang/DNK_LiDAR_Vegetation_Metric)](https://github.com/Jinhu-Wang/DNK_LiDAR_Vegetation_Metric/issues)
[![GitHub forks](https://img.shields.io/github/forks/Jinhu-Wang/DNK_LiDAR_Vegetation_Metric)](https://github.com/Jinhu-Wang/DNK_LiDAR_Vegetation_Metric/network)
[![GitHub stars](https://img.shields.io/github/stars/Jinhu-Wang/DNK_LiDAR_Vegetation_Metric)](https://github.com/Jinhu-Wang/DNK_LiDAR_Vegetation_Metric/stargazers)

## Table of Contents
- [Overview](#overview)
- [Processing Pipeline](#processing-pipeline)
- [File Structure](#file-structure)
- [Requirements](#requirements)
- [Usage Instructions](#usage-instructions)
- [License](#license)
- [Contact](#contact)

---

## Overview

**DNK_LiDAR_Vegetation_Metric** is a set of parallel Python scripts that derive vegetation
structure metrics from Airborne Laser Scanning (ALS) point clouds over Denmark, and validate
them against field-surveyed woody-cover ground truth.

All three stages share one raster geometry: every grid (DTM and vegetation metric) is
**point-symmetric about the plot's own geometric centre** — the bounding-box centre of all
points in the tile. Each axis has an even number of cells (`2n`), so the plot centre is always
the shared corner of four pixels, and any padding is split equally on both sides. Because the
centre does not depend on resolution, a plot has the same anchor in every DTM, every metric
raster and every ground-truth comparison cell.

---

## Processing Pipeline

```text
1_ClippedPCD/ (input .laz plots, ASPRS-classified)
        |
        |  1_dtm_gen_normalization_parallel_v_02.py
        v
2_DTM_Parallel_idw_v_01/res_{0_1 .. 10_0}/          (DTM GeoTIFFs, one per plot per resolution)
3_Normalized_Parallel_idw_v_01/res_{0_1 .. 10_0}/   (height-normalized .laz, Z = height above ground)
        |
        |  2_compute_4_vegetation_metrics_v_02.py
        v
4_Vegetation_Metrics_v_02/{class}/res_{norm}/{metric}/res_{veg}/{plot}_{metric}.tif
4_Vegetation_Metrics_v_02/plot_density_report.csv
        |
        |  3_ground_truth_comparison_v_01.py  (+ woody_plants_output_v2.txt)
        v
5_Ground_Truth_Comparison_v_01/{class}/{metric}/   (difference matrices, per-plot table, heatmaps)
```

The default resolution sweep is 19 values on both the normalization and the metric axis:
`0.1, 0.2, ... 1.0` m and `2, 3, ... 10` m.

### Stage 1 — DTM generation and height normalization
`1_dtm_gen_normalization_parallel_v_02.py`

* **Eligibility filter.** A tile is processed only if it has at least `min_ground_points`
  (default `6`, the IDW/KNN neighbourhood size) ground returns (ASPRS class `2`). Tiles below
  the threshold are reported as `EXCLUDED` in `processing_log.txt`, not as failures.
* **Purge of stale outputs.** With `purge_excluded = True` (default), any DTM or normalized
  cloud left on disk by an earlier run for an excluded tile is **deleted**. Every deletion is
  printed and logged. Set it to `False` to leave existing files untouched.
* **Centre-symmetric grid.** The DTM grid is built around the bounding-box centre of *all*
  points, with `n = ceil(half_extent / res)` cells on each side. Only class-2 points define the
  surface; the extent comes from the full tile.
* **Interpolation** at pixel centres. Supported methods: `idw` (default; 6 nearest ground
  points, `1/d²` weights), `linear` (with nearest-neighbour fill of edge NaNs), `rbf`, `knn`.
* **GeoTIFF writing** is north-up (row 0 = northernmost) with the origin on the pixel edge, so
  values and georeferencing agree.
* **Normalization** samples the *rasterized* DTM bilinearly (`RegularGridInterpolator`) at
  every point, so the normalized heights reflect the resolution-specific surface. Points in the
  outer half-pixel ring are **edge-clamped** to the nearest pixel centre rather than dropped,
  so the normalized cloud keeps the full point extent of the input tile. The number of clamped
  points is reported per resolution.
* Writes a normalized `.laz` with `Z` = height above ground, preserving all other dimensions.
* **Resume:** with `resume = True`, tiles whose DTM and normalized cloud both exist *and open
  cleanly* are skipped.

### Stage 2 — Vegetation metrics
`2_compute_4_vegetation_metrics_v_02.py`

* **Point-density filter.** For each tile, average density = point count / bounding-box area
  (pts/m²), read from the LAS header only. Tiles below `min_point_density` (default `20.0`)
  get no rasters at any metric resolution. Every tile's density and kept/excluded status is
  written to `plot_density_report.csv`. Set `min_point_density = None` to disable the filter.
* Rasterizes the normalized clouds and computes four per-pixel height statistics —
  **`mean`**, **`median`**, **`max`** and **`p95`** — in a single group-by-pixel pass
  (`lexsort` + `reduceat`).
* Uses the **same centre-symmetric grid as stage 1**, derived from the full point extent (not
  the target-class subset). Since stage 1 drops no points, the metric raster and the DTM of a
  plot at the same resolution share an origin, and rasters are aligned across classes and
  normalization resolutions.
* Default class configuration is `class_4_5` (ASPRS `4` medium + `5` high vegetation) only.
  `class_4` / `class_5` can be re-added in `class_configs` for a per-stratum breakdown.
* Pixels without points of the target class are `NaN`; a plot with no target-class points
  still gets all-`NaN` rasters so the output tree stays complete.
* Every combination of normalization resolution × metric resolution is produced:
  `len(resolutions)² × len(metrics) × len(classes)` raster sets.
* **Resume:** with `resume = True`, plots whose four metric rasters already exist and open
  cleanly are skipped, per (class, norm_res, veg_res) cell.

### Stage 3 — Ground truth comparison
`3_ground_truth_comparison_v_01.py`

* Reads the space-separated ground truth file (`filename under over total`). Only the
  **`over`** column — woody cover above 1 m inside the 5 m plot circle, in m² — is used.
* The **5 m radius circle is centred on the geometric centre of each metric raster** (the same
  plot centre stages 1 and 2 built their grids around). The surveyed coordinates in
  `0_Shapefiles_2018/filtered_plots_2018.csv` are optional and used **only as a diagnostic**
  (offset between raster centre and surveyed centre); they do not affect any area.
* Reads the stage-2 rasters as written. Every pixel with a value `>= 1.0 m` contributes its
  **exact** pixel-circle intersection area (closed-form, not pixel counting). Pixel values
  still reflect all points in the pixel, including those just outside the circle.
* **Eligibility:** a plot is used only if it has both rasters and a ground-truth row, and its
  raster extent covers at least 99 % (`MIN_CIRCLE_COVERAGE`) of the circle at the finest
  metric resolution. The same plot set is used in every cell.
* For each (normalization resolution, metric resolution) cell it computes the signed difference
  `LiDAR − GT` per plot and summarizes it with five statistics: **mean, std, min, max, range**.
* **Self-test** (`self_test = True`, default): validates the exact overlap formula against
  closed-form and Monte-Carlo values, checks that pixel weights sum to the circle area, compares
  against an independent per-pixel reimplementation, and verifies the circle centre is
  invariant across resolutions and metrics. The run aborts if it fails.
* **Checkpointing:** results are flushed to `_cache/areas.npz` every `checkpoint_seconds`
  (default 300 s); a restart recomputes only unfinished (plot, veg_res) cells. Cells with a
  missing or unreadable raster are left un-done and retried on the next run.
* **Replot mode:** `replot_only = True` only redraws heatmaps from existing CSVs.

Outputs per `{class}/{metric}/` folder:

| File | Content |
|------|---------|
| `diff_matrix_area_over_1m_{mean,std,min,max,range}.csv` | Per-cell statistic of `LiDAR − GT` (m²); rows = normalization res., columns = metric res. |
| `heatmap_diff_matrix_area_over_1m_{stat}.png` | Heatmap of each matrix (diverging `RdBu` centred on 0 for mean/min/max, `viridis` for std/range) |
| `per_plot_area_over_1m.csv.gz` | One row per (norm_res, veg_res, plot): LiDAR area, GT area, difference |

Plus, at the top of the output folder, `plot_circle_coverage.csv` (circle centre, offset from
the surveyed centre, and circle coverage per plot).

---

## File Structure

```text
DNK_LiDAR_Vegetation_Metric/
│
├── 1_dtm_gen_normalization_parallel_v_02.py   # Stage 1: DTM + height normalization
├── 2_compute_4_vegetation_metrics_v_02.py     # Stage 2: mean/median/max/p95 rasters
├── 3_ground_truth_comparison_v_01.py          # Stage 3: validation vs. field data
│
├── .gitignore                                 # Files to be ignored by Git
├── LICENSE                                    # MIT License file
└── README.md                                  # Project documentation (this file)
```

Data (`1_ClippedPCD/`, `2_DTM_Parallel_idw_v_01/`, `3_Normalized_Parallel_idw_v_01/`,
`4_Vegetation_Metrics_v_02/`, `5_Ground_Truth_Comparison_v_01/`, `0_Shapefiles_2018/`,
`woody_plants_output_v2.txt`) is **not** tracked in this repository; output folders are created
at runtime next to the scripts.

---

## Requirements

A working Python environment (Python 3.8+ recommended). Required libraries:

* `numpy`
* `scipy` (interpolation: `griddata`, `RBFInterpolator`, `RegularGridInterpolator`, `cKDTree`)
* `laspy` (reading/writing `.las` and `.laz` point clouds)
* `lazrs` or `laszip` (LAZ backend for `laspy`)
* `rasterio` (GeoTIFF I/O)
* `scikit-learn` (imported by stage 1; used by the `knn` interpolation option)
* `pandas`
* `matplotlib` and `seaborn` (heatmaps)
* `tqdm` (optional — progress bars; the scripts fall back to a no-op if it is missing)

Install with `pip`:

```bash
pip install numpy scipy "laspy[lazrs]" rasterio scikit-learn pandas matplotlib seaborn tqdm
```

All stages use `ProcessPoolExecutor`, and each worker opens its own files — memory use scales
with the number of workers times plot size. Reduce `cpu_usage_percent` (stages 1–2) or
`max_workers` (stage 3) if you run out of RAM.

---

## Usage Instructions

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Jinhu-Wang/DNK_LiDAR_Vegetation_Metric.git
   cd DNK_LiDAR_Vegetation_Metric
   ```

2. **Prepare the input data.** Place the classified, clipped `.laz` plots in `1_ClippedPCD/`
   (or point `input_folder` at your own directory). The point clouds must carry standard ASPRS
   classification codes — `2` for ground, `4`/`5` for medium/high vegetation.

3. **Edit the `CONFIGURATION` block** at the bottom of each script (paths, CRS, resolutions,
   CPU share, thresholds). The default CRS is `EPSG:25832` (ETRS89 / UTM zone 32N, Denmark).
   Make sure each stage's input folder matches the previous stage's output folder:

   | Stage | Reads | Writes |
   |-------|-------|--------|
   | 1 | `input_folder` | `base_dtm_folder`, `base_output_folder` |
   | 2 | `normalized_parent_folder` (= stage 1 `base_output_folder`) | `base_output_folder` |
   | 3 | `metrics_parent` (= stage 2 `base_output_folder`) | `output_dir` |

   > **Note:** as shipped, stage 2 writes to `4_Vegetation_Metrics_v_02` while stage 3 reads
   > `metrics_parent = "4_Vegetation_Metrics_v_01"`. Set `metrics_parent` to
   > `4_Vegetation_Metrics_v_02` to compare against the density-filtered output.

4. **Run the stages in order:**
   ```bash
   python 1_dtm_gen_normalization_parallel_v_02.py
   python 2_compute_4_vegetation_metrics_v_02.py
   python 3_ground_truth_comparison_v_01.py
   ```
   Stages 2 and 3 exit with an error message if their input folder is missing. With resume
   enabled (default), an interrupted stage can simply be re-run.

5. **Provide the ground truth file** before running stage 3. `woody_plants_output_v2.txt` is
   expected next to the script (or in the working directory), space-separated, without a header:
   ```text
   plot_0001.laz  12.5  31.0  43.5
   plot_0002.laz   8.2  27.4  35.6
   ```
   Columns are: plot filename, surveyed area under 1 m, area over 1 m, total area (m²). Only the
   third column (`over`) is used. The `.laz` extension is optional. Optionally, place
   `0_Shapefiles_2018/filtered_plots_2018.csv` (columns `aktId`, `dato`, `UTM_X`, `UTM_Y`) to
   get the centre-offset diagnostic.

6. **Inspect the results** in `5_Ground_Truth_Comparison_v_01/{class}/{metric}/`. In each
   heatmap the rows are normalization resolutions (increasing upward), the columns are metric
   resolutions, and the value is the chosen statistic of the signed area difference in m². In
   the mean heatmap, blue/red indicate under- and over-estimation relative to the field survey.

> **Note on runtime and disk use:** the default sweep is 19 × 19 resolutions × 4 metrics. For
> `class_4_5` alone and ~2,600 plots that is roughly 3.8 million GeoTIFFs (~50 GB, dominated by
> the 0.1–0.3 m metric resolutions), and stage 3 reads every one of them. Start with a short
> `resolutions` list (e.g. `[0.5, 1.0, 5.0]`) and a handful of plots to verify the setup before
> launching the full run. Use the same `resolutions` list in all three stages.

---

## License

This project is licensed under the **MIT License**.

You are free to use, modify, and distribute this software, provided that the original copyright
notice and permission notice are included in all copies or substantial portions of the software.
See the `LICENSE` file for more details.

---

## Contact

**Author:** Jinhu Wang  
**GitHub:** [@Jinhu-Wang](https://github.com/Jinhu-Wang)  

If you encounter any issues, have questions, or want to contribute to the project, please feel
free to [open an issue](https://github.com/Jinhu-Wang/DNK_LiDAR_Vegetation_Metric/issues) or
submit a pull request!
