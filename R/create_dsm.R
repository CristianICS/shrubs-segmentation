# Create normalized Digital Surface Model and DEM derived products
library(lidR)
library(terra)

# Config ----------------------------------------------------------------------
# Get the current script path if running in RStudio
if (rstudioapi::isAvailable()) {
  ROOT <- dirname(dirname(rstudioapi::getSourceEditorContext()$path))
} else {
  # Insert the ROOT dir manually
  ROOT <- "../../"
}

data_dir <- file.path(ROOT, "data")

# Select the best classified point cloud according to ground reference data
cls_pmf_path <- file.path(data_dir, "las_pmf_cls_tiles")
cls_pmf <- readLAScatalog(cls_pmf_path)

# Create DEM ----------------------------------------------------------------
dtm_path <- file.path(data_dir, "encinacorba_dem.tif")

if (file.exists(dtm_path)) {
  dtm_tin <- rast(dtm_path)
} else {
  dtm_tin <- rasterize_terrain(cls_pmf, res = 0.20, algorithm = tin())
  
  writeRaster(
    dtm_tin, 
    dtm_path,
    gdal=c("COMPRESS=DEFLATE", "PREDICTOR=2")
  )
  gc()
}

# Compute derived products

# Slope in degrees (standard for analysis)
slope_path <- file.path(data_dir, "encinacorba_slope.tif")
if (!file.exists(slope_path)) {
  slope_deg <- terrain(dtm_tin, v = "slope", unit = "degrees")
  writeRaster(
    slope_deg,
    slope_path,
    gdal=c("COMPRESS=DEFLATE", "PREDICTOR=2")
  )
  rm(slope_deg)
  gc()
}

# Aspect
aspect_path <- file.path(data_dir, "encinacorba_aspect.tif")
if (!file.exists(aspect_path)) {
  aspect_deg <- terrain(dtm_tin, v = "aspect", unit = "degrees")
  writeRaster(
    aspect_deg,
    aspect_path,
    gdal=c("COMPRESS=DEFLATE", "PREDICTOR=2")
  )
  rm(aspect_deg)
  gc()
}

# Compute Hillshade raster using the radian layers
hillshade_path <- file.path(data_dir, "encinacorba_hillshade.tif")
if (!file.exists(hillshade_path)) {
  
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
    gdal=c("COMPRESS=DEFLATE", "PREDICTOR=2")
  )
  
  rm(hillshade, slope_rad, aspect_rad)
  gc()
}

# Height normalization --------------------------------------------------------
normalization_dir <- file.path(data_dir, "las_pmf_nrm_tiles")
if (!dir.exists(normalization_dir)) {
  dir.create(normalization_dir)
}

opt_output_files(cls_pmf) <- file.path(
  normalization_dir, "hillside_{XLEFT}_{YTOP}_normalized"
)
opt_laz_compression(cls_pmf) <- TRUE

nlas <- normalize_height(cls_pmf, algorithm = dtm_tin)
gc()

# Create an nDSF by interpolating heights of the points -----------------------
dsm_path <- file.path(data_dir, "encinacorba_dsm.tif")
# Save the processed tiles in R
opt_output_files(nlas) <- ""

if (!file.exists(dsm_path)) {
  dsm <- rasterize_canopy(nlas, res = 0.20, p2r(0.15, na.fill = tin()))
  
  writeRaster(
    dsm, 
    dsm_path,
    gdal=c("COMPRESS=DEFLATE", "PREDICTOR=2")
  )
}

gc()
