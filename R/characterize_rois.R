# Get the current script path if running in RStudio
if (rstudioapi::isAvailable()) {
  ROOT <- dirname(dirname(rstudioapi::getSourceEditorContext()$path))
} else {
  # Insert the ROOT dir manually
  ROOT <- "../../"
}

# Load custom functions
source(file.path(ROOT, "R", "utils/plot.R"))

data_dir <- file.path(ROOT, "data")
spec_dir <- file.path(data_dir, "spectral_signatures")

specs <- readRDS(file.path(spec_dir, "preprocessed_spectral_signatures.rds"))
rois <- sf::st_read(
  file.path(spec_dir, "spec_database.gpkg"),
  layer="square_rois"
)

# Compositions of the plots
dist_plots <- species_occupation_plot(metadata_veg)
dist_plots[[1]]

# Spectral signatures by species
