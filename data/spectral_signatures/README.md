---
author: Cristian Iranzo
date: 17/06/2024
---

# Encinacorba Fieldwork

The spectral information contained in this folder was collected at Encinacorba fire perimeter on June 17th 2024. A crew formed by a GNSS operator, an ecologist and a radiometer assistant collected a sample of spectral signatures and the vegetation formation in 37 $9m^2$ plots (equivalent to one pixel in the GEOSAT image).

The main objectives of this study are two:

1. Asses if there are differences between plots predominantly covered by shrubland and the remaining plots using GEOSAT data.
2. Use the plots covered with shrubs to delineate this kind of vegetation, to validate a CNN model to segment shrubs.

## Processing

* Each ID in the `"metadata.csv"` file corresponds to a plot.
* Several measurements are taken in each plot using the spectroradiometer FieldSpec4.
* The `"start"` and `"end"` columns in `"metadata.csv"` indicate the first and last IDs of the spectra collected in the plot (1 ID = 1 measurement).
* Each plot has associated information about its vegetation formation:

  * `esp1` and `ocu1`: species code and percentage cover of the dominant species.
  * `espX`: codes of the secondary species, ordered according to their percentage cover.
  * `commnts`: Field comments.

The `"veg_codes.csv"` file contains the species names and codes. It can be joined to the `"metadata.csv"` file using the `code == espX` columns. This file includes a `"label"` column that distinguishes vegetation types as follows:

* 0: Shrub vegetation.
* 1: Tree vegetation.
* 2: Bare soil.

### R Code

The R script `"preprocess_spectres.R"` processes the spectral signatures to remove atmospheric contamination from the measurements. It uses the functions defined in the `"spectre_utils.R"` file, which can be used to join the files containing the spectra and the methodology data. This makes it possible to plot the spectra by species.

The script `create_square_rois.R` will export the square plots with the percentage cover of the dominated species. This will be used to delineate the shrubs making the test data to validate the CNN model.

## Discriminant Analysis

Visualising the spectral signatures can provide an indication of the differences between forest vegetation and shrub vegetation in the study area. These contrasts will make it possible to model shrubland areas using, e.g., segmentation models. However, it is necessary to determine precisely **at which wavelengths** the greatest differences occur. If the identified wavelengths fall within the spectral range of the GEOSAT bands, it could be concluded that validating a model capable of distinguishing between tree and shrub vegetation cover is feasible.

TODO: The final part of the script is intended to train a model using the 738 spectral signatures. Each spectral signature is linked to the vegetation metadata of the plot in which it was collected and is assigned one of three different codes:

* 0: Shrub vegetation.
* 1: Tree vegetation.
* 2: Bare soil.

For this part of the analysis, **filter the dataset** and model only labels 0 (shrubs) and 1 (trees). The PLSR technique is used for the following reason: correctly modelling the response of a variable requires the training dataset to contain **more observations than predictor variables**, that is, the number of spectra must be greater than the number of wavelengths. A total of 738 spectra were collected, each containing measurements at 2,151 wavelengths. Therefore, PLSR can be used to reduce the number of predictor variables by grouping them through the calculation of latent components.
