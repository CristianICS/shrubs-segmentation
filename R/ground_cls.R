# Conduct a ground classification
library(lidR)

# Get the current script path if running in RStudio
if (rstudioapi::isAvailable()) {
  ROOT <- dirname(dirname(rstudioapi::getSourceEditorContext()$path))
} else {
  # Insert the ROOT dir manually
  ROOT <- "../../"
}

data_dir <- file.path(ROOT, "data")

# The original point cloud contains 1/6 of the raw cloud (50 pnts/m2)
las_ctg <- readLAScatalog(
  file.path(data_dir, "encinacorba_hillside.las"),
  filter="xyz"
)

# Retile the original LAS file to allow fit the calculus in one computer
opt_chunk_buffer(las_ctg) <- 50
opt_chunk_size(las_ctg) <- 250
opt_laz_compression(las_ctg) <- TRUE
plot(las_ctg, chunk_pattern=TRUE)

# MCC ground classification
# The parameters used are the ones found optimal for areas like the studied one
# by Montealegre et al. 2015: COMPARISON OF OPEN SOURCE LiDAR FILTERING ALGORITHMS
retile_mcc_dir <- file.path(data_dir, "las_mcc_cls_tiles")
if (!dir.exists(retile_mcc_dir)) {
  dir.create(retile_mcc_dir)
}

opt_output_files(las_ctg) <- file.path(
  retile_mcc_dir, "hillside_{XLEFT}_{YTOP}_cls"
)

las_ctg_cls <- classify_ground(las_ctg, mcc(1, 0.3))
# Perform a garbage collection operation to free disk space
gc()

# A faster approach using Progressive Morphological Filter
# https://r-lidar.github.io/lidRbook/gnd.html#sec-pmf
retile_pmg_dir <- file.path(data_dir, "las_pmf_cls_tiles")
if (!dir.exists(retile_pmf_dir)) {
  dir.create(retile_pmf_dir)
}

opt_output_files(las_ctg) <- file.path(
  retile_pmf_dir, "hillside_{XLEFT}_{YTOP}_cls"
)

ws <- seq(3, 12, 3)
th <- seq(0.1, 1.5, length.out = length(ws))
las_ctg_cls <- classify_ground(las_ctg, algorithm = pmf(ws = ws, th = th))

gc()
