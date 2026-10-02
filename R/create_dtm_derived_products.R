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

data_dir <- file.path(root, "data", "rasters")

# Select the classified point cloud with the lowest RMSE against ground
# reference data (see results/evaluation/lidar/comparative_stats.csv, computed
# in R/validate_ground_cls.R)
stats_path <- file.path(
  root, "results", "evaluation", "lidar", "comparative_stats.csv"
)
comparative_stats <- read.csv(stats_path)

best_point_cloud <- comparative_stats$point_cloud[
  which.min(comparative_stats$rmse)
]
message("Best classified point cloud: ", best_point_cloud)

cloud_dir <- file.path(root, "data", "point_cloud", "las_outlier_filter_tiles")
cls_best_path <- if (best_point_cloud == "MCC") {
  file.path(cloud_dir, "las_mcc_cls_tiles")
} else if (best_point_cloud == "PMF") {
  file.path(cloud_dir, "las_pmf_cls_tiles")
} else {
  # PTD variants are named "PTD_<angle>_<distance>"
  file.path(
    cloud_dir,
    paste0("las_ptd_cls_tiles_", sub("^PTD_", "", best_point_cloud))
  )
}

cls_best <- readLAScatalog(cls_best_path)

# Create DTM ----------------------------------------------------------------
dtm_path <- file.path(data_dir, "encinacorba_dtm.tif")

# Use the orthoimage as the rasterization template so that all products share
# its CRS, origin, extent, and cell alignment (a single layer is enough, as
# only the grid geometry is used)
ortho_path <- file.path(data_dir, "orthomosaic_clip.tif")
template <- rast(ortho_path)[[1]]

if (overwrite || !file.exists(dtm_path)) {

  dtm_tin <- rasterize_terrain(
    cls_best,
    res = template,
    algorithm = tin()
  )

  writeRaster(
    dtm_tin,
    dtm_path,
    gdal = c("COMPRESS=DEFLATE", "PREDICTOR=3"),
    overwrite = TRUE
  )

  gc()

} else {

  dtm_tin <- rast(dtm_path)

  # An existing DTM may predate the template-based grid
  if (!compareGeom(dtm_tin, template, stopOnError = FALSE)) {
    stop(
      "Existing DTM is not aligned with the orthomosaic; ",
      "set overwrite <- TRUE to recompute it."
    )
  }

}

# Compute derived products  ---------------------------------------------------

# Slope in degrees (standard for analysis)
slope_path <- file.path(data_dir, "encinacorba_slope.tif")

if (overwrite || !file.exists(slope_path)) {

  slope_deg <- terrain(dtm_tin, v = "slope", unit = "degrees")
  writeRaster(
    slope_deg,
    slope_path,
    gdal = c("COMPRESS=DEFLATE", "PREDICTOR=3")
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
    gdal = c("COMPRESS=DEFLATE", "PREDICTOR=3")
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
    gdal = c("COMPRESS=DEFLATE", "PREDICTOR=3")
  )

  rm(hillshade, slope_rad, aspect_rad)
  gc()
}

# Height normalization --------------------------------------------------------
normalization_dir <- file.path(cls_best_path, "las_nrm_tiles")
if (!dir.exists(normalization_dir)) {
  dir.create(normalization_dir)
}

# Reuse the normalized tiles when there is one for every classified tile
# (normalization is slow); otherwise compute them
normalized_files <- list.files(
  normalization_dir,
  pattern = "\\.(laz|las)$",
  full.names = TRUE
)

if (!overwrite && length(normalized_files) == nrow(cls_best@data)) {

  message("Reusing normalized tiles in: ", normalization_dir)
  nlas <- readLAScatalog(normalization_dir)

} else {

  opt_output_files(cls_best) <- file.path(
    normalization_dir, "encinacorba_{XLEFT}_{YTOP}_normalized"
  )
  opt_laz_compression(cls_best) <- TRUE

  nlas <- normalize_height(cls_best, knnidw(), use_class = c(2L))
  gc()

}

# Create a canopy height model (CHM). Interpolate heights of the points -------
chm_path <- file.path(data_dir, "encinacorba_chm.tif")
# Save the processed tiles in R
opt_output_files(nlas) <- ""

# The CHM is computed at a coarser resolution than the orthomosaic (5 cm): with
# a non-interpolated CHM only ~5% of the 5 cm cells hold a LiDAR point, whereas
# at 0.25 m most cells do. The template is the orthomosaic grid aggregated by an
# integer factor, so it keeps the same CRS, origin and alignment
chm_res <- 0.25
chm_fact <- chm_res / res(template)[1]
if (abs(chm_fact - round(chm_fact)) > 1e-6) {
  stop("CHM resolution must be an integer multiple of the orthomosaic resolution.")
}
chm_template <- aggregate(template, fact = round(chm_fact))

# Avoid na.fill = tin() inside p2r function to skip interpolation
chm <- rasterize_canopy(nlas, res = chm_template, p2r())

writeRaster(
  chm,
  chm_path,
  gdal = c("COMPRESS=DEFLATE", "PREDICTOR=3"),
  overwrite = TRUE
)

gc()
