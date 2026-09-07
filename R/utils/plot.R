#' Make a dataframe "long"
#' 
#' Reshape the original list containing the spectres in a new dataframe to plot
#' them with ggplot2. The lapply function is used to break the list into small
#' data.frames containing each spectre. Note that if data.frame function is
#' called with several arguments with more than one value but one argument with
#' only one value, the function will return a dataframe and the unique value
#' is replicated the same number of times of the largest vector.
#' 
#' Example: View(data.frame(id=1, wav = c(3,4,5), ref = c(.09,.08,.9)))
#' 
#' Important: The specs list should contain the item ids as its names.
#' 
#' @param specs Spectre signatures' list.
#' @param waves Character|Numeric vector with the wavelengths
#' @returns A dataframe object.
reshapeDf <- function(specs, waves) {
  # Create rows containing:
  # 1. The spec id
  # 2. The wavelength and
  # 3. The reflectance of the current wavelength linked with the target ID.
  rows <- lapply(seq_along(specs), function(i) {
    data.frame(
      id = as.integer(names(specs[i])), 
      wavelength = as.integer(waves),
      reflectance = specs[[i]]
    )
  })
  
  # Use the rbind function to add together each dataframe row:
  spectral_signatures_df <- do.call(rbind, rows)
  
  return(spectral_signatures_df)
}


#' Plot spectres linked with certain veg type
#' 
#' Reshape the original spectra. Show the capture time and their ids.
#' 
#' @param meta data.frame with the spectral signatures metadata.
#' @param specs Spectre signatures' list.
#' @param waves Character|Numeric vector with the wavelength
#' @param veg_code Code from esp1 in metadata
#' @param plot_id Identifier of the plot (i.e., row in metadata)
#' @param average Perform the average of all the spectre signatures
plotVeg <- function(meta, specs, waves, veg_code, plot_id, average=FALSE) {
  # Retrieve the veg name
  veg_name <- meta[meta$esp1 == veg_code,]$esp1_name[1]
  # Select WR data
  vegData <- meta[(meta$esp1 == veg_code) & (meta$plot_id == plot_id),]
  
  # Transform the specs into a long format dataframe
  specs_df <- reshapeDf(specs, waves)
  
  # Merge above two dataframes
  vegData_merged <- merge(
    specs_df,
    vegData[, c('plot_id', 'spec_id')], by.x="id", by.y="spec_id")
  
  if (average) {
    # Group by 'wavelength' and calculate the average of 'reflectance'
    vegData_merged <- vegData_merged |>
      dplyr::group_by(wavelength) |>
      dplyr::summarize(
        reflectance = mean(reflectance, na.rm = TRUE),
        # Keep the rest of the necessary columns
        id = max(id)
      )
  }
  
  # Plot
  ggplot2::ggplot(
    vegData_merged,
    ggplot2::aes(x = wavelength, y = reflectance, group = id)
  ) +
    ggplot2::geom_line() +
    ggplot2::labs(x = "Wavelength", y = "Reflectance", group = "ID") +
    ggplot2::theme_minimal() +
    ggplot2::ylim(0, 0.75) + 
    # Get the desired wavelength region
    ggplot2::xlim(400, 2400) +
    ggplot2::ggtitle(paste0("Spectral Signatures - ", veg_name))
}


species_occupation_plot <- function(metadata_df) {
  occu_df <- metadata_df |>
    dplyr::select(id, esp1_name, esp2_name, ocu1, ocu2) |>
    tidyr::pivot_longer(
      cols = c(esp1_name, esp2_name, ocu1, ocu2),
      names_to = c(".value", "rank"),
      names_pattern = "(esp|ocu)([12])"
    ) |>
    dplyr::rename(
      species = esp,
      occupation = ocu
    ) |>
    dplyr::filter(
      !is.na(species),
      !is.na(occupation)
    )
  
  # Boxplot for each species
  boxplot_obj <- ggplot2::ggplot(
    occu_df,
    ggplot2::aes(
      x = reorder(species, occupation, FUN = median),
      y = occupation
    )
  ) +
    ggplot2::geom_boxplot() +
    ggplot2::coord_flip() +
    ggplot2::labs(
      x = "Species",
      y = "Occupation (%)",
      title = "Distribution of occupation percentage by species"
    ) +
    ggplot2::theme_minimal()
  
  # Histogram for each species
  hist_obj <- ggplot2::ggplot(
    occu_df,
    ggplot2::aes(x = occupation)
  ) +
    ggplot2::geom_histogram(
      binwidth = 10,
      boundary = 0,
      color = "white"
    ) +
    ggplot2::facet_wrap(~ species) +
    ggplot2::scale_x_continuous(
      limits = c(0, 100),
      breaks = seq(0, 100, 20)
    ) +
    ggplot2::labs(
      x = "Occupation (%)",
      y = "Number of plots",
      title = "Distribution of occupation percentage by species"
    ) +
    ggplot2::theme_minimal()
  
  return(list(boxplot_obj, hist_obj))
}
