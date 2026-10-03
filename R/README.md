# LiDAR processing

This folder contains the codes to process the LiDAR data to create the Canopy Height Model (CHM) to apply the mask refinement.

## R setup

The R package versions used to produce the results are recorded in `renv.lock` (R 4.6.1). To install the same versions, run from the repository root:

```r
install.packages("renv")
renv::restore()
```

The lockfile covers these packages and their dependencies: `lidR`, `RMCC`, `terra`, `sf`, `dplyr`, `ggplot2`, `patchwork`, `data.table`, `rstudioapi`, `tidyr`.

> Note: `lidR`, `rlas` and `RMCC` are installed from GitHub, so `renv::restore()` needs [Rtools](https://cran.r-project.org/bin/windows/Rtools/) on Windows to compile them.

Each script finds the repository root by itself when it is opened in RStudio. Otherwise, it uses the working directory, so run the scripts from the repository root:

```bash
Rscript R/ground_cls.R
```

## Ground classification

`ground_cls.R` reads `data/point_cloud/encinacorba_cloud.laz` and splits it into 250 m tiles with a 50 m buffer. It removes the noise points with a statistical outlier filter (SOR) and then classifies the ground points with three algorithms:

| Algorithm | Parameters |
| --- | --- |
| Multiscale Curvature Classification (MCC) | `mcc(1, 0.3)`, from Montealegre et al. (2015) |
| Progressive Morphological Filter (PMF) | Window sizes 3 to 12 m, height thresholds 0.1 to 1.5 m |
| Progressive TIN Densification (PTD) | Three angle/distance combinations: 30°/2 m, 30°/3 m and 40°/3 m |

Then, the script `validate_ground_cls.R` compares each classified point cloud with the GNSS orthometric height of the field targets (`data/spectral_signatures/spec_locations.geojson`).

For every classification, it builds a DTM with TIN method on the orthomosaic grid. It then samples each DTM at the targets with bilinear interpolation. The error is the DTM height minus the GNSS height. Only the targets covered by every DTM are used, so all methods are compared on the same targets. It writes the following files inside `results/evaluation/lidar`:

- `comparative_stats.csv`: one row per point cloud, with the number of targets, mean error (bias), standard deviation, MAE, RMSE, maximum absolute error and the percentage of targets with an absolute error above 0.5 m.
- `target_errors.csv`: the DTM height and the error of every target for each point cloud.
- `transect_plot.svg` and `transects.geojson`: four transects comparing MCC, PMF and the best PTD combination.

## DTM and canopy height model

`create_dtm_derived_products.R` selects the point cloud with the lowest RMSE in `comparative_stats.csv` and creates the following products inside `data/rasters/`:

- `encinacorba_dtm.tif`: the digital terrain model, on the same grid as the orthomosaic.
- `encinacorba_slope.tif`, `encinacorba_aspect.tif` and `encinacorba_hillshade.tif`.
- `encinacorba_chm.tif`: the canopy height model at 0.25 m, built from the height-normalized point cloud.

The CHM is not interpolated, so a cell without a LiDAR point stays empty instead of receiving an estimated height. Its 0.25 m grid is aligned with the 5 cm orthomosaic. Set `overwrite <- TRUE` at the top of the script to recompute products that already exist.

## Height of the annotated masks

`mask_height_stats.R` extracts the CHM height of every mask in `data/encinacorba_annotations.gpkg` and summarizes it by species and by growth form (tree or shrub). These statistics were used to choose the refinement thresholds. It writes the following files inside `results/evaluation/lidar/`:

- `mask_height_stats.csv`
- `species_height_stats.csv`
- `growth_form_height_stats.csv`

## References

A. L. Montealegre, M. T. Lamelas and J. de la Riva, "A Comparison of Open-Source LiDAR Filtering Algorithms in a Mediterranean Forest Environment," in IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, vol. 8, no. 8, pp. 4072-4085, Aug. 2015, doi: 10.1109/JSTARS.2015.2436974.
