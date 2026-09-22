# Create the graphics to characterize the field work plots

# Config ----------------------------------------------------------------------
# Get the current script path if running in RStudio
if (rstudioapi::isAvailable()) {
  root <- dirname(dirname(rstudioapi::getSourceEditorContext()$path))
} else {
  # Insert the ROOT dir manually
  root <- getwd()
}

# Load custom functions
source(file.path(root, "R", "utils/plot.R"))

data_dir <- file.path(root, "data")
spec_dir <- file.path(data_dir, "spectral_signatures")

specs <- readRDS(file.path(spec_dir, "preprocessed_spectral_signatures.rds"))
rois <- sf::st_read(
  file.path(spec_dir, "spec_database.gpkg"),
  layer = "square_rois"
)
rois_df <- sf::st_drop_geometry(rois)

# Compositions of the plots
dist_plots <- species_occupation_plot(rois_df)
dist_plots[[1]]

# Spectral signatures by species -----------------------------------------------
# ASD FieldSpec4 wavelength range (350-2500 nm, 1 nm resolution)
waves <- 350:2500

# Expand each plot into one row per collected spectrum (defined by the
# 'start'/'end' spec id range), so the table matches the 'plot_id'/'spec_id'
# schema expected by plotVeg().
meta <- do.call(rbind, lapply(seq_len(nrow(rois_df)), function(i) {
  data.frame(
    spec_id = rois_df$start[i]:rois_df$end[i],
    plot_id = rois_df$id[i],
    esp1 = rois_df$esp1[i],
    esp1_name = rois_df$esp1_name[i]
  )
}))

# Compare the averaged spectral signature of a shrub-dominated plot...
plotVeg(meta, specs, waves, veg_code = 3, plot_id = 36, average = TRUE)
# ...against a tree-dominated plot
plotVeg(meta, specs, waves, veg_code = 5, plot_id = 14, average = TRUE)
