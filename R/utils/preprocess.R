#' Get .asd metadata from path
#' 
#' Read the 'path' property inside the spectra metadata. Use get_metadata to
#' retrieve a data.frame containing the ASD FieldSpectre metadata from that
#' path.
#' 
#' Then, add the spectrum custom ID to join the returned DF
#' (instrument's metadata) with the spectra's metadata from the path has been
#' retrieved.
#' 
#' Only a few desired columns are returned.
#' 
#' This function is integrated inside apply method through a dataframe
#' containing the spectre path and the spectre id.
#' 
#' @param row A character vector.
#' @returns A named character vector.
#' @examples
#' apply(paths)
writeMetadata <- function(row){
    # Retrieve original ASD metadata
    specMeta <- asdreader::get_metadata(row['path'])
    # Add the asd ID to perform a join later
    specMeta['spec_id'] <- as.integer(row[['spec_id']])
    
    # Convert dates to POSIX objects
    whenPOSIX <- as.POSIXlt(specMeta[['when']], format="%Y-%m-%d %H:%M:%S %Z")
    refPOSIX <- as.POSIXlt(specMeta[['ref_time']], format="%Y-%m-%d %H:%M:%S %Z")
    dcPOSIX <- as.POSIXlt(specMeta[['dc_time']], format="%Y-%m-%d %H:%M:%S %Z")
    
    # Update time to increase one hour (Madrid time zone)
    # POSIXct objects are a measure of seconds from an origin, add 1h's seconds
    specMeta['when'] <- format(whenPOSIX + 60*60, format="%Y-%m-%d %H:%M:%S")
    specMeta['ref_time'] <- format(refPOSIX + 60*60, format="%Y-%m-%d %H:%M:%S")
    specMeta['dc_time'] <- format(dcPOSIX + 60*60, format="%Y-%m-%d %H:%M:%S")
    
    # Unlist the result to append it later to a data frame
    return(specMeta)
}


#' Join user metadata with each spectrum's instrument metadata 
#' 
#' 1. Retrieve all the spectra's ids from their paths.
#' 2. Insert the spectra's paths inside the user metadata by ID
#' 3. Get the instrument metadata by the spectrum's path and add its user ID
#' 4. Join the user metadata with the spectra's one by user's ID
#' 
#' @param asdPaths List with string spectrum paths.
#' @param metadata Data frame with user metadata.
#' @returns A named character vector.
build_metadata <- function(asdPaths, metadata) {
    # First, retrieve the spectra's ids and plot id from their filenames
    # Construct an empty tabular data structure to store the ids
    proto <- data.frame(plot_id = integer(), spec_id = integer())
    
    # Get the ID of the spectra inside its filename path
    # Note: The above proto remains empty, the ids're going to store inside ids 
    pattern <- "(\\d+)_spec(\\d+)"
    ids <- strcapture(pattern, asdPaths, proto, perl = TRUE)
    
    # Merge ids with their paths inside the above DF with the extracted paths
    ids$path <- asdPaths
    
    # Remove paths without ID (NAs)
    ids <- na.omit(ids)
    
    # Add custom metadata
    specMeta <- merge(ids, metadata, by.x="plot_id", by.y="id")

    # Get asd metadata for the target spectra with its id
    # The result of the apply function is a list of lists
    asdMeta <- apply(specMeta, 1, writeMetadata)
    
    # Transform each list into a dataframe and append all of the lists into one DF
    asdMeta <- do.call(rbind, lapply(asdMeta, as.data.frame))
    
    # Add asd metadata inside the main dataframe (from ASD)
    specMeta <- merge(specMeta, asdMeta, by = "spec_id")
    
    return(specMeta)
}


#' Correct the original spectres fitting the good white references (WR)
#' 
#' Several of the WR field samples had been taken with errors. Compute a K
#' coefficient between the good one and the bad one of each spectre, and apply
#' it to the spectre in order to correct the sample.
#'
#' It returns only the corrected spectres (not the WR ones).
#' 
#' @param specs Named list with all the spectres.
#' @param meta Data frame with the spectre metadata.
computeK <- function(specs, meta, ref_wr_id = 67){
    # Select the WR to correct the rest (the good one)
    good_wr <- specs[[ref_wr_id]]
    
    # Get all the spectres except WR ones
    target_specs <- specs[meta[meta$label != 'WR',]$id]
    
    # Correct all the above spectres
    for (id in names(target_specs)) {
        # Select the WR with the current spectre was taken
        wr_id <- meta[meta$id == id,]$wr_id
        
        # Check if current spectre has an associate WR or it's the good WR
        if (wr_id > 0 | wr_id == 67) {
            # Get the WR to correct
            bad_wr <- specs[[wr_id]]
            # Compute K
            k <- c(good_wr / bad_wr)
            # Apply k to the original spect to correct it
            target_specs[[id]] <- k * target_specs[[id]]
        }
    }
    
    return(target_specs)
}


#' Trim wavelengths
#' 
#' Given a numeric vector with a spectre, add NAs inside the limits pointed by
#' "lowerLimit" and "upperLimit" (both included).
#' 
#' This function is apply inside lapply function to all the spectres inside a
#' list.
#' 
#' @param spec Vector of wavelength's values
#' @param lower_limit Wavelength number where starting to put NAs
#' @param upper_limit Wavelength number where stop trimming
#' @param waves Character|Numeric vector with the wavelengths
trimWaves <- function(spec, lower_limit, upper_limit, waves) {
    # Get position of the desired wavelength inside the spec
    low_lim_pos <- which(as.integer(waves) == lower_limit)
    up_lim_pos <- which(as.integer(waves) == upper_limit)
    
    # Add NA values inside the above limits
    spec[low_lim_pos:up_lim_pos] <- NA
    
    return(spec)
}


#' Apply a Savitzky-Golay filter to continuous spectral regions
#'
#' Applies the filter independently to each uninterrupted region without NAs.
#' NA regions are preserved.
#'
#' @param x Numeric vector containing spectral intensity values.
#' @param w Odd integer specifying the filter window length.
#' @param k Polynomial order.
#' @param m Derivative order.
#' @param short_segments How segments shorter than `w` should be handled:
#'   `"keep"` leaves them unchanged; `"adaptive"` uses a smaller valid window.
#'
#' @return A numeric vector of the same length as `x`.
custom_sgolayfilt <- function(
    x,
    w = 11L,
    k = 2L,
    m = 0L,
    short_segments = c("keep", "adaptive")
) {
  short_segments <- match.arg(short_segments)
  
  if (!is.numeric(x)) {
    stop("`x` must be numeric.")
  }
  
  if (length(w) != 1L || w < 1L || w %% 2L == 0L) {
    stop("`w` must be a positive odd integer.")
  }
  
  if (length(k) != 1L || k < 0L || k >= w) {
    stop("`k` must be non-negative and smaller than `w`.")
  }
  
  if (length(m) != 1L || m < 0L || m > k) {
    stop("`m` must be between 0 and `k`.")
  }
  
  valid_indices <- which(!is.na(x))
  
  # Return unchanged when all values are NA
  if (length(valid_indices) == 0L) {
    return(x)
  }
  
  # Split valid indices whenever they are not consecutive
  segment_id <- cumsum(c(
    TRUE,
    diff(valid_indices) != 1L
  ))
  
  segments <- split(valid_indices, segment_id)
  
  for (indices in segments) {
    segment_length <- length(indices)
    
    if (segment_length >= w) {
      x[indices] <- signal::sgolayfilt(
        x[indices],
        n = w,
        p = k,
        m = m
      )
      
    } else if (short_segments == "adaptive") {
      # Largest odd window no longer than the segment
      adaptive_w <- if (segment_length %% 2L == 1L) {
        segment_length
      } else {
        segment_length - 1L
      }
      
      # The window must be larger than the polynomial order
      if (adaptive_w > k) {
        x[indices] <- signal::sgolayfilt(
          x[indices],
          n = adaptive_w,
          p = k,
          m = m
        )
      }
    }
  }
  
  x
}


#' Apply Savitzky-Golay smoothing filter / Noise Removal
#'
#' Fits a local polynomial regression of order k on a series of spectral values
#' to determine the smoothed value for each wavelength, for a given filter
#' length (aka window size).
#'
#' @param spectra Loaded spectra as named list.
#' @param w Filter length or window size
#' @param k Polynomial order of the SG regression.
#' @param m Return the m-th derivative of the filter coefficients.
filterSg <- function(spectra, w, k, m=0) {
  spectra <- as.matrix(unlist(spectra))
  
  # Run filter
  sg <- aaply(spectra, 1, signal::sgolayfilt, n = w, p = k, m = m)
  
  # Arrange appropriately if a single sample
  if (nrow(spectra) == 1) {
    sg <- matrix(sg, dim(spectra))
  }
  
  # Return data frame
  sg <- as.data.frame(sg)
  colnames(sg) <- colnames(spectra)
  
  return(sg)
}


#' Apply SNV correction
#' 
#' Normalize a spectrum by itself.
#'  
#' @param x Numeric vector with spectrum wavelength values
snv <- function(x) {
    snv_spectra <- (x - mean(x, na.rm=TRUE)) / sd(x, na.rm=TRUE)
    return(snv_spectra)
}


#' @title Splice correction of a spectral matrix acquired with an ASD spectrometer
#' @description
#' Corrects steps in an input spectral matrix by linear interpolation of the
#' values of the edges of the middle sensor.
#'
#' @usage
#'
#' spliceCorrection(X, wav, splice = c(1000, 1830), interpol.bands = 10)
#'
#' @param X a numeric vector.
#' @param wav a numeric vector with band positions.
#' @param splice a numeric vector of length 1, the 1830 region is not available
#'   due to the atmospheric noise (this region has been trimmed previously).
#' @param interpol.bands the number of interpolation bands.
#' @details
#' This function modifies the \code{spliceCorrection} one from the package
#' \code{prospectr} to handle the saved spectra schema in this project.
#' 
spliceCorrection <- function(X, wav, splice = 1000, interpol.bands = 10){
    
    if (!is.integer(wav)) {
        wav <- as.integer(wav)
    }
    
    # Function to make the linear interpolation
    extrapfun <- function(x, y, xout) {
        fit <- lm(y ~ x)
        fit$coefficients[1] + fit$coefficients[2] * xout
    }
    
    # Get the index of the values to splice
    index <- which(wav %in% splice)
    
    # Select the wavelengths before and after the splice wavelength
    Xa <- X[1:index[1], drop = FALSE]
    Xb <- X[(index[1] + 1):length(X), drop = FALSE]
    # Select data to interpolate
    tmp_first <- Xb[1:interpol.bands, drop = FALSE]
    w_first <- wav[(index[1] + 1):(index[1] + interpol.bands)]
    # Interpolate bands after the splice break (return the intercept coef.)
    pred_Xa <- extrapfun(x = w_first, y = tmp_first, xout = splice[1])
    # Compute the offset
    offset_a <- Xa[length(Xa)] - pred_Xa
    
    output <- c((Xa - offset_a), Xb)
    return(output)
}


#' Spectral Trimming 
#' 
#' Remove regions within the range in the defined wavelenght limits.
#' 
#' It is intended to apply this function inside a lapply operation.
#' 
#' @param spectra Named list with the spectra
#' @param wavelenghts List with the wavelengths inside the spectra
#' @param wavlimits Vector with lower and upper limits to trim..
trimSpec <- function(spectra, wavelengths, wavlimits) {
  stopifnot(
    length(wavlimits) == 2L,
    length(spectra) == length(wavelengths)
  )
  
  lower <- min(wavlimits)
  upper <- max(wavlimits)
  
  to_trim <- wavelengths >= lower & wavelengths <= upper
  
  spectra[to_trim] <- NA
  
  spectra
}


#' Create square ROIs centered on point geometries
#'
#' Creates axis-aligned square polygons of a specified size centered exactly
#' on each input point.
#'
#' @param points An `sf` object containing POINT geometries. The CRS must be
#'   projected and use metres as its coordinate units.
#' @param size Numeric. Side length of each square in metres. Default is 5.
#'
#' @return An `sf` object with the original point attributes and square
#'   polygon geometries.
#'
#' @examples
#' squares <- create_square_rois(points, size = 5)
create_square_rois <- function(points, size = 5) {
  
  # Check that coordinates are projected
  if (sf::st_is_longlat(points)) {
    stop(
      "`points` uses a geographic CRS. Transform it to a projected CRS ",
      "with metre units before creating the squares."
    )
  }
  
  half_size <- size / 2
  
  # Get point coordinates
  coords <- sf::st_coordinates(points)
  
  # Create one square around each point
  squares <- lapply(seq_len(nrow(coords)), function(i) {
    
    x <- coords[i, "X"]
    y <- coords[i, "Y"]
    
    sf::st_polygon(list(
      matrix(
        c(
          x - half_size, y - half_size,
          x + half_size, y - half_size,
          x + half_size, y + half_size,
          x - half_size, y + half_size,
          x - half_size, y - half_size
        ),
        ncol = 2,
        byrow = TRUE
      )
    ))
  })
  
  # Replace point geometries with square geometries
  sf::st_geometry(points) <- sf::st_sfc(
    squares,
    crs = sf::st_crs(points)
  )
  
  points
}

#' Add vegetation information for a species column
#'
#' Reads the vegetation-code lookup table, adds vegetation type information,
#' and joins the resulting metadata to a species column such as `esp1` or
#' `esp2`.
#'
#' The joined fields are renamed using the species column as a prefix,
#' producing columns such as `esp1_name`, `esp1_label`, and
#' `esp1_label_code`.
#'
#' @param data A data frame containing the species code column.
#' @param species Character string giving the name of the species code column
#'   to use for the join (e.g. `"esp1"` or `"esp2"`).
#' @param veg_codes_path Path to the `veg_codes.csv` file.
#'
#' @return The input data frame with three additional columns:
#'   `<species>_name`, `<species>_label`, and `<species>_label_code`.
#'
#' @examples
#' veg_codes_path <- file.path(spec_dir, "veg_codes.csv")
#'
#' metadata_veg <- metadata |>
#'   add_veg_info("esp1", veg_codes_path) |>
#'   add_veg_info("esp2", veg_codes_path)
add_veg_info <- function(data, species, veg_codes_path) {
  
  # Read vegetation codes
  labels <- read.csv(veg_codes_path)
  
  # Define vegetation types
  veg_types <- data.frame(
    type = c("Shrub", "Tree", "Bare ground"),
    label = c(0, 1, 2)
  )
  
  # Add vegetation type information
  labels <- labels |>
    dplyr::left_join(
      veg_types,
      by = "label"
    ) |>
    dplyr::rename(
      label_code = label,
      label = type
    )
  
  # Prepare lookup table
  lookup <- labels |>
    dplyr::rename(
      name_tmp = name,
      label_tmp = label,
      label_code_tmp = label_code
    )
  
  # Join vegetation information
  data |>
    dplyr::left_join(
      lookup,
      by = stats::setNames("code", species)
    ) |>
    dplyr::rename(
      "{species}_name" := name_tmp,
      "{species}_label" := label_tmp,
      "{species}_label_code" := label_code_tmp
    )
}
