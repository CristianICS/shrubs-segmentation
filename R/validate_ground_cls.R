# Check the accuracy of the ground classification with GNSS data
library(lidR)
library(dplyr)
library(ggplot2)
library(patchwork)
library(sf)
# Custom package to compute target metrics
library(lidaynight)

# Get the current script path if running in RStudio
if (rstudioapi::isAvailable()) {
  root <- dirname(dirname(rstudioapi::getSourceEditorContext()$path))
} else {
  # Insert the ROOT dir manually
  root <- getwd()
}

data_dir <- file.path(root, "data")
cloud_dir <- file.path(data_dir, "las_outlier_filter_tiles")

cls_mcc_path <- file.path(cloud_dir, "las_mcc_cls_tiles")
cls_pmf_path <- file.path(cloud_dir, "las_pmf_cls_tiles")
cls_ptd_path <- file.path(cloud_dir, "las_ptd_cls_tiles")

cls_mcc <- readLAScatalog(cls_mcc_path)
cls_pmf <- readLAScatalog(cls_pmf_path)
cls_ptd <- readLAScatalog(cls_ptd_path)

#' Plot consecutive transects from the classified LAS catalogs
#'
#' Creates five equally spaced points along the diagonal of a selected catalog
#' tile and extracts the four transects connecting consecutive points. The
#' transects from the classification results are displayed in a four-row,
#' three-column layout, with one column per catalog and one row per transect.
#'
#' The diagonal extends from the upper-left to the lower-right corner of the
#' selected tile. Each LAS file is read once and reused for all four transects.
#'
#' @param cls_mcc `lidR::LAScatalog` containing the MCC classification.
#' @param cls_pmf `lidR::LAScatalog` containing the PMF classification.
#' @param cls_ptd `lidR::LAScatalog` containing the PTD classification.
#' @param tile_number Integer index of the catalog tile to use. Defaults to
#'   `8`.
#' @param width Numeric width of each transect, in the coordinate units of the
#'   LAS data. Defaults to `5`.
#'
#' @return A `patchwork` object containing twelve transect plots arranged in
#'   four rows and three columns.
create_transect_plot <- function(
  cls_mcc,
  cls_pmf,
  cls_ptd,
  tile_number = 8L,
  width = 2
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

  get_transect <- function(las, points, width) {
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

  patchwork::wrap_plots(
    plots,
    ncol = length(catalogs),
    guides = "collect",
    axes = "collect",
    axis_titles = "collect"
  )
}

create_transect_plot(cls_mcc, cls_pmf, cls_ptd)
ggplot2::ggsave(
  file.path(root, "results", "evaluation", "lidar", "transect_plot.svg"),
  width = 10,
  height = 12
)
gc()

# Use the GNSS elevation data to assess the ground classification
gnss_data_path <- file.path(
  data_dir,
  "spectral_signatures",
  "spec_locations.geojson"
)

gnss_data <- st_read(gnss_data_path)
elev_col <- "height_ortho"
id_col <- "id"

# Get the average point cloud height within a buffer from each GNSS target
buffer_size_m <- 0.20

targets_rmse <- function(
  sf_pnts,
  elev_col,
  id_col,
  buffer_m,
  ctg,
  pnt_cloud_name
) {
  # Select only ground points
  opt_filter(ctg) <- "-keep_class 2"

  pnt_buffers <- st_buffer(sf_pnts, dist = buffer_m)

  # Select only the points inside the buffers.
  # Returns a list of LAS objects, one per buffer
  las_list <- clip_roi(ctg, pnt_buffers)

  # Construct the table where extracted point cloud data will be stored
  ref_table <- st_drop_geometry(pnt_buffers[, c(id_col, elev_col)])

  metrics <- data.table::rbindlist(
    Map(
      compute_target_metrics,
      las_list,
      ref_table[[id_col]],
      ref_table[[elev_col]],
      # Note: the incoherent threshold acts as the heigh value to consider
      # a point invalid based on the difference between the point and the
      # reference height. Value in meters
      MoreArgs = list(incoh_thresh = 0.5)
    )
  )

  # RMSE of LiDAR last returns within each buffer relative to that
  # target's reference height.
  metrics$rmse_last <- ifelse(
    metrics$n_pnts_last > 0L,
    sqrt(metrics$sum_errs_last_sq / metrics$n_pnts_last),
    NA_real_
  )

  summarize_target_metrics <- metrics |>
    summarise(
      point_cloud = pnt_cloud_name,

      pct_incoherent_points =
        100 * sum(n_pnts_last_incoherent, na.rm = TRUE) /
        sum(n_pnts_last, na.rm = TRUE),

      mean_rmse = mean(rmse_last, na.rm = TRUE),

      min_thickness = min(thickness, na.rm = TRUE),
      max_thickness = max(thickness, na.rm = TRUE),
      mean_thickness = mean(thickness, na.rm = TRUE)
    )

  list("target_metrics" = metrics, "summarize" = summarize_target_metrics)
}

cls_mcc_metrics <- targets_rmse(gnss_data, elev_col, id_col, buffer_size_m, cls_mcc, "MCC")
cls_pmf_metrics <- targets_rmse(gnss_data, elev_col, id_col, buffer_size_m, cls_pmf, "PMF")
cls_ptd_metrics <- targets_rmse(gnss_data, elev_col, id_col, buffer_size_m, cls_ptd, "PTD")

# Compute comparative stats
# - Percentage of incoherent points: column "n_pnts_last_incoherent"
# - Mean RMSE: column "rmse_last"
# - Max, Min and Mean thickness: column thickness
comparative_stats <- bind_rows(
  cls_mcc_metrics[["summarize"]],
  cls_pmf_metrics[["summarize"]],
  cls_ptd_metrics[["summarize"]]
)

comparative_stats

write.csv(
  comparative_stats,
  file.path(root, "results", "evaluation", "lidar", "comparative_stats.csv"),
  row.names = FALSE
)
