# LiDAR refinement

The aim of this section is post-processing the model derived masks and refine the predictions using the canopy height model (CHM) product derived from LiDAR data (`data/rasters/encinacorba_chm.tif`).

The LiDAR information is not fed back into the CNN; instead, each RGB-derived instance is independently assessed against an external structural measurement. This makes it possible to apply exactly the same refinement procedure to outputs from different architectures. The same calibrated CHM thresholds and refinement parameters should be applied to every architecture, rather than optimizing them separately for each CNN.

## Methodological considerations

- CHM coverage should not be calculated from a filled CHM. If the function `p2r` usess `na.fill = tin()` initially empty cells are interpolated. The final CHM can therefore contain valid numerical values where there was no direct point-to-raster support. A CHM without interpolation is generated with the `R/create_dtm_derived_products` script.
- In order to improve RGB–LiDAR integration, the CHM share the same CRS, origin, and raster alignment. `lidR` allows a `SpatRaster` to be supplied as the rasterization template.
- The height signal is only trustworthy for tree vs. shrub, not shrub vs. ground. Thus the CHM is not used to reject "ground-dominated" masks, leaving that to the RGB model's confidence.

## Vegetation heigh study

The code `R/mask_height_stats.R` is used to compute the height statistics for each annotated mask and species. The result is the mean height and the mean of masks means (m) for each species and also groupped based on their growth form (tree or shrub). The mean height weights each mask by its number of valid cells, as in the species table. The mean of mask means gives every mask the same weight.

The mask with no valid CHM cells is excluded, so the totals add up to 78 masks.

Shrubs peak at 1.63 m, and tree masks have a mean-of-means of 1.71 m and reach 7.3 m. A single threshold around 1.5–2 m on `h_p90` probably does most of the work. A shrub mask overlapping a tree crown will have a high `h_p90` even though the model's instance is right. 

The 78-mask sample also has negative CHM values (-0.16 m), which confirms the noise floor is comparable to the *Lavandula stoechas* signal.

## Refinement study

For each polygon in any vector file (GeoJSON or GPKG, reprojected to the CHM CRS if needed), the code `Python/lidar_refine_masks.py` computes the following over the CHM cells whose centre falls inside the mask:

- valid cell count and coverage
- `h_p90`, `h_mean`, `h_max`
- `tall_fraction` (share of cells above `--h-tall`)

It then assigns one status, in this order:

| Status	| Condition |	Result |
| ------- | --------- | ------ |
| review	| coverage < `--min-coverage` (0.5) or fewer than `--min-cells` (5) valid cells | Kept |
| tree | `h_p90` > `--h-tall` (1.7) and `tall_fraction` ≥ `--tall-fraction` (0.5) | Rejected |
| clipped | Some tall cells but not enough to reject | Tall cells are cut out and the largest remaining part is kept. Rejected as `tree` if the remainder after clip operation is under `--min-area` (0.25). |
| keep | Otherwise	| Kept unchanged |

Masks smaller than a cell, outside the raster, or empty come back with no valid cells and get review.

The refinement script is running on annotated masks to tests the thresholds with `--stats-only`:

```bash
python Python/lidar_refine_masks.py \
   data/encinacorba_annotations.gpkg \
   --layer field_work_masks \
   --chm data/rasters/encinacorba_chm.tif \
   --output results/lidar_refined/annotations.gpkg \
   --stats-only --min-coverage 0.3 --h-tall 1.7 --tall-fraction 0.1
```

After running the script, only 5.5% of CHM cells hold a direct LiDAR measurement, so most cells contain no point. Shrub masks have a median of 30 valid cells and 22 of 60 have fewer than 20, so their p90 is noisy. Only 8 shrub masks have fewer than 5 valid cells.

The non-interpolated CHM is generated at 0.25 m to avoid a low number of CHM cells withouth a direct LiDAR measurement. The CHM layer contains a point in 69% of cases, against 5.5% at 5 cm.

The median valid cells per mask is 15 for shrubs and 97 for trees. That is still small for tiny *Lavandula* masks, but usable.

Three trees are only `keep`. Ids 44 (*Crataegus monogyna*, `h_p90` of 1.37), 65 (*Quercus ilex*, 1.35) and 60 (*Pinus pinaster*, 1.44) have a `tall_fraction` of 0. Their heights overlap *Cistus laurifolius* (up to 1.47) and *Crataegus monogyna* (1.55), so height alone cannot remove them. These could be low resprouts after the 2003 fire.

Lowering `tall_fraction` to 0.1–0.3 produced the best number of trees rejected at no cost of reject shrubs, since every shrub has `tall_fraction` below 0.05. `h_p90` $>$ 1.5 with `tall_fraction` $\geq$ 0.1 reaches 14 of 18 at no shrub loss, but only by a narrow margin: *Cistus laurifolius* and *Crataegus monogina* reach 1.47 and 1.55 m.

| `h_p90` $>$ | `tall_fraction` $\geq$ | Trees rejected | Shrubs rejected |
| ------ | ----- | ------- | ------- |
| 1.7 | 0.5 | 12 | 0 |  
| 1.7 | 0.1 to 0.3| 13 | 0 |  
| 1.5 | 0.1 to 0.5 | 12-14 | 0 |  
|1.3 or 1.5| 0.0 | 16-18 | 1-3 |  

The 1.7 threshold is kept. `h_p90` was confirmed as the right statistic. *Lavandula stoechas* Ids 78 and 79 have `h_max` about 1.5 m but `h_p90` of 0.22 and 0.33 m, probably single stray cells. These thresholds (`--min-coverage 0.3 --h-tall 1.7 --tall-fraction 0.1`) are applied unchanged to the three models in the next section.
