# Check the accuracy of the ground classification with GNSS data
library(lidR)
library(terra)
library(dplyr)
library(ggplot2)
library(patchwork)
library(sf)

# Get the current script path if running in RStudio
if (rstudioapi::isAvailable()) {
  root <- dirname(dirname(rstudioapi::getSourceEditorContext()$path))
} else {
  # Insert the ROOT dir manually
  root <- getwd()
}

# Set to TRUE to recompute and overwrite the DTMs of each classification;
# set to FALSE to reuse the existing ones when available.
overwrite <- FALSE

data_dir <- file.path(root, "data")
cloud_dir <- file.path(data_dir, "point_cloud", "las_outlier_filter_tiles")

cls_mcc_path <- file.path(cloud_dir, "las_mcc_cls_tiles")
cls_pmf_path <- file.path(cloud_dir, "las_pmf_cls_tiles")

# Three parameter combinations tested for the PTD filter, encoded in each
# folder name as "<angle>_<distance>"
ptd_variants <- c("30_2", "30_3", "40_3")
cls_ptd_paths <- setNames(
  file.path(cloud_dir, paste0("las_ptd_cls_tiles_", ptd_variants)),
  paste0("PTD_", ptd_variants)
)

cls_mcc <- readLAScatalog(cls_mcc_path)
cls_pmf <- readLAScatalog(cls_pmf_path)
cls_ptd_list <- lapply(cls_ptd_paths, readLAScatalog)

# Use the GNSS elevation data to assess the ground classification
gnss_data_path <- file.path(
  data_dir,
  "spectral_signatures",
  "spec_locations.geojson"
)

gnss_data <- st_read(gnss_data_path)
elev_col <- "height_ortho"
id_col <- "id"

# Height difference (m) above which a DTM value is considered incoherent
# with the GNSS reference height
incoh_thresh_m <- 0.5

# Create one DTM per classification ---------------------------------------
# Each DTM is built exactly as data/rasters/encinacorba_dtm.tif (TIN on the
# orthomosaic grid, see R/create_dtm_derived_products.R), so the selected
# classification is validated through the same product used downstream.
# Sampling a continuous DTM gives every method a value at the same targets,
# whereas the ground points around a target can be missing for some methods.
dtm_dir <- file.path(data_dir, "rasters", "ground_cls_dtms")
if (!dir.exists(dtm_dir)) {
  dir.create(dtm_dir)
}

# A single layer is enough, as only the grid geometry is used
template <- rast(file.path(data_dir, "rasters", "orthomosaic_clip.tif"))[[1]]

get_dtm <- function(ctg, pnt_cloud_name) {
  dtm_path <- file.path(dtm_dir, paste0(pnt_cloud_name, "_dtm.tif"))

  if (!overwrite && file.exists(dtm_path)) {
    return(rast(dtm_path))
  }

  message("Creating DTM for ", pnt_cloud_name)
  dtm <- rasterize_terrain(ctg, res = template, algorithm = tin())

  writeRaster(
    dtm,
    dtm_path,
    gdal = c("COMPRESS=DEFLATE", "PREDICTOR=3"),
    overwrite = TRUE
  )
  gc()

  rast(dtm_path)
}

catalogs <- c(list(MCC = cls_mcc, PMF = cls_pmf), cls_ptd_list)
dtms <- Map(get_dtm, catalogs, names(catalogs))

# Compare each DTM with the GNSS targets -----------------------------------
# Bilinear interpolation of the DTM at each target location. Error is
# DTM minus GNSS, so a positive error means the DTM is above the ground.
targets_errors <- function(dtm, sf_pnts, elev_col, id_col, pnt_cloud_name) {
  sf_pnts <- st_transform(sf_pnts, crs(dtm))

  dtm_z <- terra::extract(
    dtm,
    vect(sf_pnts),
    method = "bilinear",
    ID = FALSE
  )[[1]]

  data.frame(
    point_cloud = pnt_cloud_name,
    id = sf_pnts[[id_col]],
    gnss_z = sf_pnts[[elev_col]],
    dtm_z = dtm_z,
    error = dtm_z - sf_pnts[[elev_col]]
  )
}

target_errors <- bind_rows(
  unname(Map(
    targets_errors,
    dtms,
    MoreArgs = list(sf_pnts = gnss_data, elev_col = elev_col, id_col = id_col),
    pnt_cloud_name = names(dtms)
  ))
)

# Only keep the targets covered by every DTM, so all methods are compared on
# the same set of targets (targets outside the orthomosaic grid are dropped)
common_ids <- target_errors |>
  group_by(id) |>
  filter(all(!is.na(error))) |>
  pull(id) |>
  unique()

message(
  length(common_ids), " of ", nrow(gnss_data),
  " GNSS targets are covered by every DTM"
)

target_errors <- target_errors |>
  mutate(common_target = id %in% common_ids)

# Compute comparative stats
# - n_targets: number of targets used
# - mean_error: bias of the DTM (DTM - GNSS)
# - sd_error, mae, rmse, max_abs_error: dispersion of the errors
# - pct_incoherent_targets: targets whose absolute error exceeds the threshold
comparative_stats <- target_errors |>
  filter(common_target) |>
  group_by(point_cloud) |>
  summarise(
    n_targets = n(),
    mean_error = mean(error),
    sd_error = sd(error),
    mae = mean(abs(error)),
    rmse = sqrt(mean(error^2)),
    max_abs_error = max(abs(error)),
    pct_incoherent_targets = 100 * mean(abs(error) > incoh_thresh_m)
  ) |>
  # Keep the order in which the classifications were defined
  arrange(match(point_cloud, names(dtms)))

comparative_stats

write.csv(
  comparative_stats,
  file.path(root, "results", "evaluation", "lidar", "comparative_stats.csv"),
  row.names = FALSE
)

write.csv(
  target_errors,
  file.path(root, "results", "evaluation", "lidar", "target_errors.csv"),
  row.names = FALSE
)

# Keep the PTD parameterization with the lowest RMSE to compare against
# MCC and PMF in the transect plot
best_ptd_name <- comparative_stats |>
  filter(point_cloud %in% names(cls_ptd_list)) |>
  slice_min(rmse, n = 1) |>
  pull(point_cloud)

message("Best PTD parameterization: ", best_ptd_name[1])
cls_ptd_best <- cls_ptd_list[[best_ptd_name[1]]]

#' Plot consecutive transects from the classified LAS catalogs
#'
#' Creates five equally spaced points along the diagonal of a selected catalog
#' tile and extracts the four transects connecting consecutive points. The
#' transects from the classification results are displayed in a four-row,
#' three-column layout, with one column per catalog and one row per transect.
#'
#' The diagonal extends from the upper-left to the lower-right corner of the
#' selected tile. Each LAS file is read once and reused for all four transects.
#' Each transect is shortened to `length_fraction` of the distance between its
#' two points, keeping it centred on the segment midpoint.
#'
#' @param cls_mcc `lidR::LAScatalog` containing the MCC classification.
#' @param cls_pmf `lidR::LAScatalog` containing the PMF classification.
#' @param cls_ptd `lidR::LAScatalog` containing the PTD classification.
#' @param tile_number Integer index of the catalog tile to use. Defaults to
#'   `8`.
#' @param width Numeric width of each transect, in the coordinate units of the
#'   LAS data. Defaults to `5`.
#'
#' @param length_fraction Numeric in (0, 1] giving the fraction of each
#'   segment between consecutive points that is kept. Defaults to `0.5`.
#'
#' @return A list with `plot` (a `patchwork` object containing twelve transect
#'   plots arranged in four rows and three columns), `coords` (a data frame
#'   with the start/end coordinates of each transect) and `lines` (an `sf`
#'   object holding one `LINESTRING` per transect).
create_transect_plot <- function(
  cls_mcc,
  cls_pmf,
  cls_ptd,
  tile_number = 8L,
  width = 2,
  length_fraction = 0.5
) {

  get_five_points <- function(ctg, tile_number) {
    geom <- ctg@data$geometry[tile_number]
    bbox <- sf::st_bbox(geom)

    top_left <- c(
      bbox["xmin"],
      bbox["ymax"]
    )

    center <- c(
      (bbox["xmin"] + bbox["xmax"]) / 2,
      (bbox["ymin"] + bbox["ymax"]) / 2
    )

    diagonal <- sf::st_sfc(
      sf::st_linestring(
        rbind(top_left, center) |>
          unname()
      ),
      crs = sf::st_crs(geom)
    )

    diagonal |>
      sf::st_line_sample(sample = seq(0, 1, length.out = 5)) |>
      sf::st_coordinates() |>
      # st_coordinates() is returning three columns: X, Y, and a line idx
      # As clip_transect() requires exactly two coordinates, select X,Y
      (\(x) unname(x[, c("X", "Y"), drop = FALSE]))()
  }

  get_point_pairs <- function(points) {
    lapply(
      seq_len(nrow(points) - 1L),
      \(i) points[c(i, i + 1L), , drop = FALSE]
    )
  }

  read_tile <- function(ctg, tile_number) {
    lidR::readLAS(
      ctg@data$filename[tile_number],
      select = "xyzc"
    )
  }

  shrink_segment <- function(points, fraction) {
    midpoint <- matrix(colMeans(points), nrow = 2L, ncol = 2L, byrow = TRUE)
    midpoint + (points - midpoint) * fraction
  }

  get_transect <- function(las, points, width) {
    points <- shrink_segment(points, length_fraction)
    lidR::clip_transect(
      las,
      points[1, ],
      points[2, ],
      width = width,
      xz = TRUE
    )
  }

  plot_transect <- function(transect, catalog, transect_id, col_index) {
    data <- lidR::payload(transect)

    data$Class <- factor(
      ifelse(data$Classification == 2L, "Ground", "Other"),
      levels = c("Ground", "Other")
    )

    ggplot2::ggplot(
      data,
      ggplot2::aes(X, Z, color = Class)
    ) +
      ggplot2::geom_point(size = 0.25) +
      ggplot2::coord_equal() +
      ggplot2::scale_color_manual(
        name = "Classification",
        values = c(
          Ground = "#8B4513",
          Other = "#2E86C1"
        )
      ) +
      ggplot2::labs(
        title = if (transect_id == 1L) catalog else NULL,
        # Only the leftmost column carries the row's "Transect N" label
        subtitle = if (col_index == 1L) paste("Transect", transect_id) else NULL,
        x = "Distance along transect",
        y = "Elevation"
      ) +
      ggplot2::theme_minimal()
  }

  catalogs <- list(MCC = cls_mcc, PMF = cls_pmf, PTD = cls_ptd)

  points <- get_five_points(cls_mcc, tile_number)
  point_pairs <- get_point_pairs(points)
  tile_crs <- sf::st_crs(cls_mcc@data$geometry[tile_number])
  las_tiles <- lapply(catalogs, read_tile, tile_number = tile_number)

  plots <- unlist(
    lapply(
      seq_along(point_pairs),
      \(i) {
        transects <- lapply(
          las_tiles,
          get_transect,
          points = point_pairs[[i]],
          width = width
        )

        Map(
          plot_transect,
          transect = transects,
          catalog = names(transects),
          col_index = seq_along(transects),
          MoreArgs = list(transect_id = i)
        )
      }
    ),
    recursive = FALSE
  )

  transect_coords <- do.call(
    rbind,
    lapply(seq_along(point_pairs), \(i) {
      pts <- point_pairs[[i]]
      data.frame(
        transect_id = i,
        x_start = pts[1, 1],
        y_start = pts[1, 2],
        x_end = pts[2, 1],
        y_end = pts[2, 2]
      )
    })
  )

  transect_lines <- sf::st_sf(
    transect_id = seq_along(point_pairs),
    geometry = sf::st_sfc(
      lapply(point_pairs, \(pts) sf::st_linestring(pts)),
      crs = tile_crs
    )
  )

  list(
    plot = patchwork::wrap_plots(
      plots,
      ncol = length(catalogs),
      guides = "collect",
      axes = "collect",
      axis_titles = "collect"
    ),
    coords = transect_coords,
    lines = transect_lines
  )
}

transects <- create_transect_plot(cls_mcc, cls_pmf, cls_ptd_best, width = 0.5)

print(transects$coords)
print(transects$plot)

ggplot2::ggsave(
  file.path(root, "results", "evaluation", "lidar", "transect_plot.svg"),
  plot = transects$plot,
  width = 10,
  height = 12
)

sf::st_write(
  transects$lines,
  file.path(root, "results", "evaluation", "lidar", "transects.geojson"),
  delete_dsn = TRUE
)

gc()
