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
- [Configuration Reference](#configuration-reference)
- [Related Repositories](#related-repositories)
- [License](#license)
- [Contact](#contact)

---

## Overview

**DNK_LiDAR_Vegetation_Metric** is a set of parallel Python scripts that derive vegetation
structure metrics from Airborne Laser Scanning (ALS) point clouds over Denmark, and validate
them against field-surveyed ground truth.

The core question the repository is built around is a **resolution sensitivity analysis**: how
much does the choice of (a) the DTM resolution used for height normalization and (b) the raster
resolution used for the vegetation metric affect the estimated vegetation area? The pipeline
therefore sweeps both resolutions independently over a 10 x 10 grid (0.1 m to 1.0 m) and reports
the deviation from ground truth for every combination as a heatmap.

All three stages are batch-parallel (`ProcessPoolExecutor`) and are meant to be run in order on
a folder of tiled/clipped `.laz` plots.

---

## Processing Pipeline

```text
1_ClippedPCD/ (input .laz plots, ASPRS-classified)
        |
        |  1_dtm_gen_normalization_parallel_v3.py
        v
2_DTM_Parallel_idw/res_{0_1 .. 1_0}/      (DTM GeoTIFFs, one per plot per resolution)
3_Normalized_Parallel_idw/res_{0_1 .. 1_0}/ (height-normalized .laz, Z = height above ground)
        |
        |  2_compute_4_vegetation_metrics_v4.py
        v
4_Vegetation_Metrics/{class}/res_{norm}/{metric}/res_{veg}/*.tif
        |
        |  3_ground_truth_comparison_v5.py  (+ woody_plants_output_v2.txt)
        v
5_Ground_Truth_Comparison/{class}/{metric}/  (difference matrices + heatmaps)
```

### Stage 1 — DTM generation and height normalization
`1_dtm_gen_normalization_parallel_v3.py`

* Extracts ground points (ASPRS class `2`) from each `.laz` file.
* Interpolates them onto a regular grid at each requested resolution and writes the DTM as a
  GeoTIFF. Supported interpolators: `idw` (default), `linear`, `rbf`, `knn`.
* Samples the **rasterized** DTM (bilinear, via `RegularGridInterpolator`) at every point of the
  original cloud, so the normalization reflects the resolution-specific surface rather than the
  raw ground points.
* Writes a normalized `.laz` in which `Z` is the height above ground, preserving all other point
  dimensions. Points falling outside the DTM extent are dropped.
* Failures are logged per batch to `processing_log.txt` in the output folder.

### Stage 2 — Vegetation metrics
`2_compute_4_vegetation_metrics_v4.py`

* Rasterizes the normalized clouds and computes four per-pixel height statistics:
  **`mean`**, **`median`**, **`max`** and **`p95`**.
* Runs for three class configurations — `class_4` (medium vegetation), `class_5`
  (high vegetation) and `class_4_5` (both).
* All four metrics share a single group-by-pixel pass (`lexsort` + `reduceat`), so the extra
  metrics cost almost nothing beyond the first.
* The grid is snapped to the metric resolution using the **full** point extent, so the
  `class_4`, `class_5` and `class_4_5` rasters of a plot are pixel-aligned and directly
  comparable. Pixels without points of the target class are written as `NaN`, not `0`.
* Every combination of normalization resolution x metric resolution is produced, giving
  `len(resolutions)^2 x len(metrics) x len(classes)` raster sets.

### Stage 3 — Ground truth comparison
`3_ground_truth_comparison_v5.py`

* Reads a space-separated ground truth file (`filename under over total`) listing surveyed
  woody-plant area per plot, split into vegetation below and above 1 m.
* For each raster, counts valid pixels inside a **5 m radius circular window** centred on the
  raster's bounding-box centroid, splits them at the 1 m height threshold, converts pixel counts
  to area, and computes the signed difference `LiDAR - GT`.
* Aggregates the mean difference per (normalization resolution, metric resolution) cell and
  writes `diff_matrix_area_under_1m.csv`, `diff_matrix_area_over_1m.csv` and
  `diff_matrix_area_total.csv`, plus a diverging heatmap (`seaborn`, centred on 0) for each.

---

## File Structure

```text
DNK_LiDAR_Vegetation_Metric/
│
├── 1_dtm_gen_normalization_parallel_v3.py   # Stage 1: DTM + height normalization
├── 2_compute_4_vegetation_metrics_v4.py     # Stage 2: mean/median/max/p95 rasters
├── 3_ground_truth_comparison_v5.py          # Stage 3: validation vs. field data
│
├── .gitignore                               # Files to be ignored by Git
├── LICENSE                                  # MIT License file
└── README.md                                # Project documentation (this file)
```

Data folders (`1_ClippedPCD/`, `2_DTM_Parallel_idw/`, `3_Normalized_Parallel_idw/`,
`4_Vegetation_Metrics/`, `5_Ground_Truth_Comparison/`) are **not** tracked in this repository;
they are created at runtime next to the scripts.

---

## Requirements

A working Python environment (Python 3.8+ recommended). Required libraries:

* `numpy`
* `scipy` (interpolation: `griddata`, `RBFInterpolator`, `RegularGridInterpolator`, `cKDTree`)
* `laspy` (reading/writing `.las` and `.laz` point clouds)
* `lazrs` or `laszip` (LAZ backend for `laspy`)
* `rasterio` (GeoTIFF I/O)
* `scikit-learn` (only for the `knn` interpolation option)
* `pandas`
* `matplotlib` and `seaborn` (heatmaps)
* `tqdm` (optional — progress bars; the scripts fall back to a no-op if it is missing)

Install with `pip`:

```bash
pip install numpy scipy "laspy[lazrs]" rasterio scikit-learn pandas matplotlib seaborn tqdm
```

The scripts use `multiprocessing` with the default start method, and each worker opens its own
files — expect memory use to scale with the number of workers times plot size. Reduce
`cpu_usage_percent` if you run out of RAM.

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
   CPU share). The default CRS is `EPSG:25832` (ETRS89 / UTM zone 32N, Denmark).

4. **Run the stages in order:**
   ```bash
   python 1_dtm_gen_normalization_parallel_v3.py
   python 2_compute_4_vegetation_metrics_v4.py
   python 3_ground_truth_comparison_v5.py
   ```
   Stage 2 requires the output of stage 1, and stage 3 requires the output of stage 2; each
   script exits with an error message if its input folder is missing.

5. **Provide the ground truth file** before running stage 3. `woody_plants_output_v2.txt` is
   expected next to the script, space-separated, without a header:
   ```text
   plot_0001.laz  12.5  31.0  43.5
   plot_0002.laz   8.2  27.4  35.6
   ```
   Columns are: plot filename, surveyed area under 1 m, area over 1 m, total area (m²).
   The `.laz` extension is added automatically if omitted.

6. **Inspect the results** in `5_Ground_Truth_Comparison/{class}/{metric}/`. In each heatmap the
   rows are normalization resolutions, the columns are metric resolutions, and the value is the
   mean signed area difference in m² — blue/red indicate under- and over-estimation relative to
   the field survey.

> **Note on runtime:** the full sweep is a 10 x 10 resolution grid x 4 metrics x 3 class
> configurations. On a large plot collection this is hours of compute and produces a very large
> number of GeoTIFFs. Start with a short `resolutions` list (e.g. `[0.25, 0.5, 1.0]`) and a
> handful of plots to verify the setup before launching the full run.

---

## Configuration Reference

| Parameter | Script | Default | Description |
|---|---|---|---|
| `input_folder` | 1 | `1_ClippedPCD` | Folder of input `.laz` plots |
| `base_dtm_folder` | 1 | `2_DTM_Parallel_idw` | DTM GeoTIFF output root |
| `base_output_folder` | 1 | `3_Normalized_Parallel_idw` | Normalized `.laz` output root |
| `interpolation_method` | 1 | `idw` | `idw`, `linear`, `rbf` or `knn` |
| `resolutions` | 1, 2, 3 | `0.1 … 1.0` | Resolution sweep, in metres |
| `target_crs` | 1, 2 | `EPSG:25832` | CRS written to the output rasters |
| `cpu_usage_percent` | 1, 2 | `0.9` | Fraction of cores used by the worker pool |
| `metrics_to_compute` | 2, 3 | `mean, median, max, p95` | Per-pixel height statistics |
| `class_configs` / `class_labels` | 2, 3 | `class_4, class_5, class_4_5` | ASPRS vegetation classes |
| `gt_txt_path` | 3 | `woody_plants_output_v2.txt` | Ground truth table |
| `output_dir` | 3 | `5_Ground_Truth_Comparison` | Comparison output root |

The 5 m sampling radius and the 1 m height threshold used in stage 3 are set in
`process_file_batch` (`radius_m`, and the `< 1.0` comparison).

---

## Related Repositories

* [LiDAR_Vegetation_Metrics](https://github.com/Jinhu-Wang/LiDAR_Vegetation_Metrics) —
  lightweight Jupyter Notebooks (local and Google Colab) for computing vegetation metrics from
  ALS point clouds.

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
