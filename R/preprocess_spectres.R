library("asdreader")

# Get the current script path if running in RStudio
if (rstudioapi::isAvailable()) {
  ROOT <- dirname(dirname(rstudioapi::getSourceEditorContext()$path))
} else {
  # Insert the ROOT dir manually
  ROOT <- getwd()
}

# Load custom functions
source(file.path(ROOT, "R", "utils/preprocess.R"))
source(file.path(ROOT, "R", "utils/plot.R"))

spec_dir <- file.path(ROOT, "data/spectral_signatures")

# Open the metadata file
metadata <- read.csv(file.path(spec_dir, "metadata.csv"))

# Add labels
labels <- read.csv(file.path(spec_dir, "veg_codes.csv"))
names(labels) <- paste("esp1_", names(labels), sep="")

metadata <- merge(
  metadata,
  labels,
  by.x="esp1",
  by.y="esp1_code",
  suffixes = c("", "esp1_"),
  sort=FALSE
)

# Load asd spectra paths
asdPaths <- list.files(
  file.path(spec_dir, "raw"),
  pattern = ".asd",
  full.names = TRUE
)

# Construct metadata (field user data plus asd metadata)
specMeta <- build_metadata(asdPaths, metadata)

# LOAD SPECTRE AS NAMED LIST --------------------------------------------------
# Load target asd data with asdreader function inside a large matrix:
# cols = spectre wavelengths, rows = spectre id
spec <- get_spectra(specMeta$path, type = "reflectance")
spec_list <- split(spec, specMeta$spec_id)
spec_wavelengths <- colnames(spec)

# PLOTTING (FIRST VISUAL INSPECTION) ------------------------------------------
# Plot the first spectre
matplot(x=spec_wavelengths, y=spec_list[[1]], ylim=c(0, 1), type='l',
        ylab="Reflectance", main="Plot spectre")
# Plot all the espectra
matplot(x = spec_wavelengths, y = t(do.call(rbind, spec_list)), type="l", col=1,
        ylim=c(0,1), ylab="Reflectance", main="All specs")

# Differences between Tree and Shrub species
plotVeg(specMeta, spec_list, spec_wavelengths, 3, 36, TRUE)
plotVeg(specMeta, spec_list, spec_wavelengths, 5, 14, TRUE)

# PREPROCESSING --------------------------------------------------------------
# Remove invalid spectra (with infinit data)
spec_list <- Filter(function(x) !any(is.infinite(x)), spec_list)

# Remove invalid regions of the spectre
invalid_regions <- list(
  c(2451, 2500),
  c(1820, 1950),
  c(990, 1010),
  c(350, 499),
  c(1350, 1550)
)

spec_list_trim <- Reduce(
  function(spectra, limits) {
    lapply(
      spectra,
      trimSpec,
      wavelengths = spec_wavelengths,
      wavlimits = limits
    )
  },
  invalid_regions,
  init = spec_list
)

# Apply Savitzky-Golay
spec_list_sg <- lapply(
  spec_list_trim,
  custom_sgolayfilt,
  w = 11,
  k = 2,
  # Any continuous segment shorter than "w" observations remains unchanged
  short_segments = "keep"
)

plotVeg(specMeta, spec_list_sg, spec_wavelengths, 3, 36, TRUE)

# EXPORT --------------------------------------------------------------
saveRDS(spec_list_sg, file.path(spec_dir, "preprocessed_spectral_signatures.rds"))
