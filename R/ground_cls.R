# Conduct a ground classification
library(lidR)

# Get the current script path if running in RStudio
if (rstudioapi::isAvailable()) {
  root <- dirname(dirname(rstudioapi::getSourceEditorContext()$path))
} else {
  # Insert the ROOT dir manually
  root <- getwd()
}

data_dir <- file.path(root, "data", "point_cloud")

# The original point cloud contains 1/6 of the raw cloud (50 pnts/m2)
las_ctg <- readLAScatalog(
  file.path(data_dir, "encinacorba_cloud.laz"),
  filter = "xyz"
)

# Retile the original LAS file to allow fit the calculus in one computer
opt_chunk_buffer(las_ctg) <- 50
opt_chunk_size(las_ctg) <- 250
opt_laz_compression(las_ctg) <- TRUE
plot(las_ctg, chunk_pattern = TRUE)

# Statistical Outlier Removal (SOR)
# Flags points whose average distance to k nearest neighbors
# is beyond m standard deviations from the mean, then drops them (class 18)
# before writing the cleaned tile. The buffer is used only to get correct
# neighbor statistics at tile edges and is stripped before writing.
filter_sor_noise <- function(chunk, k = 10, m = 3) {
  las <- lidR::readLAS(chunk)
  if (lidR::is.empty(las)) return(NULL)

  las <- lidR::classify_noise(las, lidR::sor(k = k, m = m))
  las <- lidR::filter_poi(las, Classification != lidR::LASNOISE)
  lidR::filter_poi(las, buffer == 0L)
}

outlier_filter_dir <- file.path(data_dir, "las_outlier_filter_tiles")
if (!dir.exists(outlier_filter_dir)) {
  dir.create(outlier_filter_dir)
}

opt_output_files(las_ctg) <- file.path(
  outlier_filter_dir, "encinacorba_{XLEFT}_{YTOP}_sor"
)

# Skip the outlier filter step if the outlier filter folder already
# contains filtered LAZ tiles, and reuse them instead
existing_sor_tiles <- list.files(
  outlier_filter_dir, pattern = "\\.laz$", full.names = TRUE
)
if (length(existing_sor_tiles) > 0) {
  las_ctg <- readLAScatalog(existing_sor_tiles, filter = "xyz")
} else {
  # Merge the per-tile outputs into a single LAScatalog so it can be reused
  # directly by the ground classification steps below
  las_ctg <- lidR::catalog_apply(
    las_ctg,
    filter_sor_noise,
    k = 10,
    m = 3,
    .options = list(automerge = TRUE)
  )
}
gc()

# Re-apply the processing options on the filtered catalog before running
# the ground classification algorithms
opt_chunk_buffer(las_ctg) <- 50
opt_chunk_size(las_ctg) <- 250
opt_laz_compression(las_ctg) <- TRUE

# MCC ground classification
# The parameters used are the ones found optimal for areas like the studied one
# by Montealegre et al. 2015: COMPARISON OF OPEN SOURCE LiDAR FILTERING...
retile_mcc_dir <- file.path(outlier_filter_dir, "las_mcc_cls_tiles")
if (!dir.exists(retile_mcc_dir)) {
  dir.create(retile_mcc_dir)
}

opt_output_files(las_ctg) <- file.path(
  retile_mcc_dir, "encinacorba_{XLEFT}_{YTOP}_cls"
)

las_ctg_cls <- classify_ground(las_ctg, mcc(1, 0.3))
# Perform a garbage collection operation to free disk space
gc()

# Progressive Morphological Filter (PMF) classification method
# https://r-lidar.github.io/lidRbook/gnd.html#sec-pmf
retile_pmf_dir <- file.path(outlier_filter_dir, "las_pmf_cls_tiles")
if (!dir.exists(retile_pmf_dir)) {
  dir.create(retile_pmf_dir)
}

opt_output_files(las_ctg) <- file.path(
  retile_pmf_dir, "encinacorba_{XLEFT}_{YTOP}_cls"
)

ws <- seq(3, 12, 3)
th <- seq(0.1, 1.5, length.out = length(ws))
las_ctg_cls <- classify_ground(las_ctg, algorithm = pmf(ws = ws, th = th))

gc()

# Progressive TIN Densification (PTD) algorithm
# https://r-lidar.github.io/lidRbook/gnd.html#sec-ptd
# Three parameter combinations are tested. Each output folder name encodes
# "<angle>_<distance>", matching the folders read by validate_ground_cls.R
ptd_params <- data.frame(
  angle    = c(30, 30, 40),
  distance = c(2, 3, 3)
)

for (i in seq_len(nrow(ptd_params))) {
  angle <- ptd_params$angle[i]
  distance <- ptd_params$distance[i]

  retile_ptd_dir <- file.path(
    outlier_filter_dir,
    paste0("las_ptd_cls_tiles_", angle, "_", distance)
  )

  # Skip this combination if its folder already holds classified tiles
  if (length(list.files(retile_ptd_dir, pattern = "\\.laz$")) > 0) {
    message("Skipping PTD ", angle, "/", distance, ": tiles already exist")
    next
  }
  dir.create(retile_ptd_dir, showWarnings = FALSE)

  opt_output_files(las_ctg) <- file.path(
    retile_ptd_dir, "encinacorba_{XLEFT}_{YTOP}_cls"
  )

  classify_ground(
    las_ctg,
    algorithm = ptd(10, angle = angle, distance = distance)
  )

  gc()
}
