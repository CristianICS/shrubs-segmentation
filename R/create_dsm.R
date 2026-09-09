# Create DTM, terrain-derived products, and canopy height model
library(lidR)
library(terra)

# Config ----------------------------------------------------------------------
# Get the current script path if running in RStudio
if (rstudioapi::isAvailable()) {
  root <- dirname(dirname(rstudioapi::getSourceEditorContext()$path))
} else {
  # Insert the ROOT dir manually
  root <- getwd()
}

# Set to TRUE to recompute and overwrite existing terrain-derived products;
# set to FALSE to reuse existing products when available.
overwrite <- FALSE

data_dir <- file.path(root, "data")

# Select the classified point cloud with the lowest RMSE against ground
# reference data (see results/evaluation/lidar/comparative_stats.csv, computed
# in R/validate_ground_cls.R)
stats_path <- file.path(
  root, "results", "evaluation", "lidar", "comparative_stats.csv"
)
comparative_stats <- read.csv(stats_path)

best_point_cloud <- comparative_stats$point_cloud[
  which.min(comparative_stats$mean_rmse)
]
message("Best classified point cloud: ", best_point_cloud)

cloud_dir <- file.path(data_dir, "las_outlier_filter_tiles")
cls_best_path <- if (best_point_cloud == "MCC") {
  file.path(cloud_dir, "las_mcc_cls_tiles")
} else if (best_point_cloud == "PMF") {
  file.path(cloud_dir, "las_pmf_cls_tiles")
} else {
  # PTD variants are named "PTD_<window_size>_<slope_threshold>"
  file.path(
    cloud_dir,
    paste0("las_ptd_cls_tiles_", sub("^PTD_", "", best_point_cloud))
  )
}

cls_best <- readLAScatalog(cls_best_path)

# Create DTM ----------------------------------------------------------------
dtm_path <- file.path(data_dir, "encinacorba_dtm.tif")

# Match the DTM resolution to the orthoimage
ortho_path <- file.path(data_dir, "encinacorba_orto.tif")
ortho_res <- res(rast(ortho_path))[1]

if (overwrite || !file.exists(dtm_path)) {

  dtm_tin <- rasterize_terrain(
    cls_best,
    res = ortho_res,
    algorithm = tin()
  )

  writeRaster(
    dtm_tin,
    dtm_path,
    gdal = c("COMPRESS=DEFLATE", "PREDICTOR=2"),
    overwrite = TRUE
  )

  gc()

} else {

  dtm_tin <- rast(dtm_path)

}

# Compute derived products  ---------------------------------------------------

# Slope in degrees (standard for analysis)
slope_path <- file.path(data_dir, "encinacorba_slope.tif")

if (overwrite || !file.exists(slope_path)) {

  slope_deg <- terrain(dtm_tin, v = "slope", unit = "degrees")
  writeRaster(
    slope_deg,
    slope_path,
    gdal = c("COMPRESS=DEFLATE", "PREDICTOR=2")
  )
  rm(slope_deg)
  gc()

}

# Aspect
aspect_path <- file.path(data_dir, "encinacorba_aspect.tif")

if (overwrite || !file.exists(aspect_path)) {
  aspect_deg <- terrain(dtm_tin, v = "aspect", unit = "degrees")
  writeRaster(
    aspect_deg,
    aspect_path,
    gdal = c("COMPRESS=DEFLATE", "PREDICTOR=2")
  )
  rm(aspect_deg)
  gc()
}

# Compute Hillshade raster using the radian layers
hillshade_path <- file.path(data_dir, "encinacorba_hillshade.tif")

if (overwrite || !file.exists(hillshade_path)) {
  slope_rad <- terrain(dtm_tin, v = "slope", unit = "radians")
  aspect_rad <- terrain(dtm_tin, v = "aspect", unit = "radians")

  hillshade <- shade(
    slope = slope_rad,
    aspect = aspect_rad,
    angle = 45,
    direction = 315
  )

  writeRaster(
    hillshade,
    hillshade_path,
    gdal = c("COMPRESS=DEFLATE", "PREDICTOR=2")
  )

  rm(hillshade, slope_rad, aspect_rad)
  gc()
}

# Height normalization --------------------------------------------------------
normalization_dir <- file.path(cls_best_path, "las_nrm_tiles")
if (!dir.exists(normalization_dir)) {
  dir.create(normalization_dir)
}

opt_output_files(cls_best) <- file.path(
  normalization_dir, "encinacorba_{XLEFT}_{YTOP}_normalized"
)
opt_laz_compression(cls_best) <- TRUE

nlas <- normalize_height(cls_best, algorithm = dtm_tin)
gc()

# Create a canopy height model (CHM). Interpolate heights of the points -------
chm_path <- file.path(data_dir, "encinacorba_chm.tif")
# Save the processed tiles in R
opt_output_files(nlas) <- ""

chm <- rasterize_canopy(nlas, res = ortho_res, p2r(0.15, na.fill = tin()))

writeRaster(
  chm,
  chm_path,
  gdal = c("COMPRESS=DEFLATE", "PREDICTOR=2")
)

gc()
