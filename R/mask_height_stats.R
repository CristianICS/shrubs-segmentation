# Compute min, max and mean normalized height of each annotated mask
library(terra)

# Config ----------------------------------------------------------------------
# Get the current script path if running in RStudio
if (rstudioapi::isAvailable()) {
  root <- dirname(dirname(rstudioapi::getSourceEditorContext()$path))
} else {
  # Insert the ROOT dir manually
  root <- getwd()
}

annotations_path <- file.path(root, "data", "encinacorba_annotations.gpkg")
chm_path <- file.path(root, "data", "rasters", "encinacorba_chm.tif")

out_dir <- file.path(root, "results", "evaluation", "lidar")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
out_path <- file.path(out_dir, "mask_height_stats.csv")

# Load data -------------------------------------------------------------------
masks <- vect(annotations_path, layer = "field_work_masks")
chm <- rast(chm_path)

if (!same.crs(masks, chm)) {
  masks <- project(masks, crs(chm))
}

# Extract height statistics per mask ------------------------------------------
# Cells are included when their centre falls inside the polygon; NA cells
# (no LiDAR return) are ignored
height_stats <- extract(
  chm,
  masks,
  fun = function(x) {
    x <- x[!is.na(x)]
    if (length(x) == 0) {
      return(c(NA_real_, NA_real_, NA_real_, 0))
    }
    c(min(x), max(x), mean(x), length(x))
  },
  ID = FALSE
)
height_stats <- as.data.frame(height_stats)
colnames(height_stats) <- c(
  "min_height",
  "max_height",
  "mean_height",
  "n_cells"
)

# Combine with the mask attributes
mask_table <- cbind(as.data.frame(masks), height_stats)

print(mask_table)
write.csv(mask_table, out_path, row.names = FALSE)
message("Table saved to: ", out_path)

# Group height stats by species -----------------------------------------------
# Masks without valid CHM cells are excluded. min/max are taken across all the
# cells of the species' masks; mean_height is weighted by the number of cells
# (equivalent to pooling all cells), while mean_of_mask_means gives each mask
# the same weight regardless of its size
valid_masks <- mask_table[mask_table$n_cells > 0, ]

species_table <- do.call(
  rbind,
  lapply(split(valid_masks, valid_masks$esp_code), function(df) {
    data.frame(
      esp_code = df$esp_code[1],
      n_masks = nrow(df),
      n_cells = sum(df$n_cells),
      min_height = min(df$min_height),
      max_height = max(df$max_height),
      mean_height = weighted.mean(df$mean_height, df$n_cells),
      mean_of_mask_means = mean(df$mean_height)
    )
  })
)
rownames(species_table) <- NULL

# Species name and growth form of each species code, from the mask attributes.
# The `shrub` column codes shrubs as 1 and trees as 0
species_info <- unique(
  as.data.frame(masks)[, c("esp_code", "esp_scientific_name", "shrub")]
)
if (anyDuplicated(species_info$esp_code)) {
  stop("Some esp_code values have more than one species name or shrub value.")
}
species_info$growth_form <- ifelse(species_info$shrub == 1, "shrub", "tree")
names(species_info)[
  names(species_info) == "esp_scientific_name"
] <- "scientific_name"

# Add the species names
species_table <- merge(
  species_table,
  species_info[, c("esp_code", "scientific_name")],
  by = "esp_code",
  all.x = TRUE,
  sort = FALSE
)
species_table <- species_table[order(species_table$esp_code), ]
rownames(species_table) <- NULL

print(species_table)

species_out_path <- file.path(out_dir, "species_height_stats.csv")
write.csv(species_table, species_out_path, row.names = FALSE)
message("Table saved to: ", species_out_path)

# Group height stats by growth form (tree / shrub) ----------------------------
valid_masks$growth_form <- ifelse(valid_masks$shrub == 1, "shrub", "tree")

growth_form_table <- do.call(
  rbind,
  lapply(split(valid_masks, valid_masks$growth_form), function(df) {
    data.frame(
      growth_form = df$growth_form[1],
      n_masks = nrow(df),
      n_cells = sum(df$n_cells),
      min_height = min(df$min_height),
      max_height = max(df$max_height),
      mean_height = weighted.mean(df$mean_height, df$n_cells),
      mean_of_mask_means = mean(df$mean_height)
    )
  })
)
rownames(growth_form_table) <- NULL

print(growth_form_table)

growth_form_out_path <- file.path(out_dir, "growth_form_height_stats.csv")
write.csv(growth_form_table, growth_form_out_path, row.names = FALSE)
message("Table saved to: ", growth_form_out_path)

# Plot mean height by species -------------------------------------------------
library(ggplot2)

# Use the cell-weighted information
species_table$scientific_name <- reorder(
  species_table$scientific_name,
  species_table$mean_height
)

# Colour the bars by growth form: dark green for trees, pastel green for shrubs
species_table$growth_form <- species_info$growth_form[
  match(species_table$esp_code, species_info$esp_code)
]
growth_form_colours <- c(tree = "#2E6B34", shrub = "#B7DDB0")

# Labels sit inside the bars, except for those too short to hold the text;
# white text is only legible on the dark green (tree) bars
species_table$label_inside <- species_table$mean_height > 0.3
species_table$label_colour <- ifelse(
  species_table$label_inside & species_table$growth_form == "tree",
  "white",
  "black"
)

species_plot <- ggplot(
  species_table,
  aes(x = mean_height, y = scientific_name)
) +
  geom_col(aes(fill = growth_form)) +
  geom_text(
    aes(
      label = sprintf("%.2f", mean_height),
      hjust = ifelse(label_inside, 1.15, -0.15),
      colour = label_colour
    ),
    size = 4,
    show.legend = FALSE
  ) +
  scale_colour_identity() +
  scale_fill_manual(values = growth_form_colours, name = "Growth form") +
  labs(x = "Mean height (m)", y = "Species") +
  theme_minimal(base_size = 12)

print(species_plot)

plot_path <- file.path(out_dir, "species_mean_height.png")
ggsave(plot_path, species_plot, width = 7, height = 4, dpi = 300, bg = "white")
message("Plot saved to: ", plot_path)

# Plot mean height by species, including every mask ---------------------------
# Same bars as above, with the mean height of each individual mask overlaid as
# points. Only vertical jitter is used so the x position of each point keeps the
# exact mean height of its mask
mask_points <- valid_masks
mask_points$scientific_name <- factor(
  mask_points$esp_scientific_name,
  levels = levels(species_table$scientific_name)
)

set.seed(1)
species_points_plot <- ggplot(
  species_table,
  aes(x = mean_height, y = scientific_name)
) +
  geom_col(aes(fill = growth_form)) +
  geom_jitter(
    data = mask_points,
    aes(x = mean_height, y = scientific_name),
    height = 0.2,
    width = 0,
    size = 1.5,
    alpha = 0.7,
    shape = 21,
    fill = "white",
    colour = "black"
  ) +
  scale_fill_manual(values = growth_form_colours, name = "Growth form") +
  labs(x = "Mean height (m)", y = "Species") +
  theme_minimal(base_size = 12)

print(species_points_plot)

points_plot_path <- file.path(out_dir, "species_mean_height_points.png")
ggsave(
  points_plot_path,
  species_points_plot,
  width = 7,
  height = 4,
  dpi = 300,
  bg = "white"
)
message("Plot saved to: ", points_plot_path)
