# Create the square ROIs around each plot's center coordinate and link the
# vegetation distribution in it.
library(sf)

# Get the current script path if running in RStudio
if (rstudioapi::isAvailable()) {
  ROOT <- dirname(dirname(rstudioapi::getSourceEditorContext()$path))
} else {
  # Insert the ROOT dir manually
  ROOT <- "../../"
}

# Load custom functions
source(file.path(ROOT, "R", "utils/preprocess.R"))

# Folder containing all the information to run the script
spec_dir <- file.path(ROOT, "data/spectral_signatures")

# Read the geojson file
points <- sf::st_read(file.path(spec_dir, "spec_locations.geojson"))

# Create square buffers around each point
buffers <- create_square_rois(points, size=3)

# View the result
plot(st_geometry(head(buffers, 1)))

# Include spectral signatures' plot metadata
metadata <- read.csv(file.path(spec_dir, "metadata.csv"))

# Add vegetation labels
veg_labels_path <- file.path(spec_dir, "veg_codes.csv")

metadata_veg <- metadata |>
  add_veg_info("esp1", veg_labels_path) |>
  add_veg_info("esp2", veg_labels_path) |>
  add_veg_info("esp3", veg_labels_path) |>
  add_veg_info("esp4", veg_labels_path)

buffers_veg <- merge(
  metadata_veg,
  buffers,
  by = "id"
) |> dplyr::select(!"tree_shrub_plot")

# Save the buffers as a new database object
st_write(
  buffers_veg,
  file.path(spec_dir, "spec_database.gpkg"),
  index=FALSE,
  layer="square_rois"
)
