# LiDAR refinement

The aim of this section is post-processing the model derived masks and refine the predictions using the canopy height model (CHM) product derived from LiDAR data (`data/rasters/encinacorba_chm.tif`).

The LiDAR information is not fed back into the CNN; instead, each RGB-derived instance is independently assessed against an external structural measurement. This makes it possible to apply exactly the same refinement procedure to outputs from different architectures. The same calibrated CHM thresholds and refinement parameters should be applied to every architecture, rather than optimizing them separately for each CNN.

## Methodological considerations

- CHM coverage should not be calculated from a filled CHM. If the function `p2r` usess `na.fill = tin()` initially empty cells are interpolated. The final CHM can therefore contain valid numerical values where there was no direct point-to-raster support. A CHM without interpolation is generated with the `R/create_dtm_derived_products` script.
- In order to improve RGB–LiDAR integration, the CHM share the same CRS, origin, and raster alignment. `lidR` allows a `SpatRaster` to be supplied as the rasterization template.
- The height signal is only trustworthy for tree vs. shrub, not shrub vs. ground. Thus the CHM is not used to reject "ground-dominated" masks, leaving that to the RGB model's confidence.

## Vegetation heigh study

The code `R/mask_height_stats.R` is used to compute the height statistics for each annotated mask and species. The result is the mean height and the mean of masks means (m) for each species and also groupped based on their growth form (tree or shrub). The mean height weights each mask by its number of valid cells, as in the species table. The mean of mask means gives every mask the same weight.

Masks with no valid CHM cells are excluded. On the 0.25 m CHM every mask has at least one valid cell, so all 79 masks are used: 58 shrubs (*Lavandula stoechas*, *Cistus laurifolius*, *Erica vagans*) and 21 trees (13 *Quercus ilex*, 6 *Pinus pinaster*, 2 *Crataegus monogyna*). Growth forms follow the `label` column of `data/spectral_signatures/veg_codes.csv`, where *Crataegus monogyna* is a tree.

No CHM cell inside a shrub mask exceeds 1.63 m (*Cistus laurifolius*), while tree masks have a mean-of-means of 1.92 m (2.79 m weighted by cells) and reach 7.28 m. At mask level, `h_p90` ranges from 0 to 1.46 m in shrubs and from 1.35 to 6.75 m in trees, so a single threshold around 1.5–2 m on `h_p90` probably does most of the work. A shrub mask overlapping a tree crown will have a high `h_p90` even though the model's instance is right. 

The sample also has negative CHM values (down to -0.05 m), which confirms the noise floor is comparable to the *Lavandula stoechas* signal.

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

In a first test with a 5 cm CHM, only 5.5% of CHM cells held a direct LiDAR measurement, so most cells contained no point and the shrub `h_p90` was noisy. The non-interpolated CHM is therefore generated at 0.25 m, where 69% of cells contain a point.

The median valid cells per mask is 15 for shrubs and 96 for trees. That is still small for tiny *Lavandula* masks, but usable. With `--min-coverage 0.3` and `--min-cells 5`, 10 shrub masks (all *Lavandula stoechas*) and no tree go to `review`.

### Threshold sweep

`Python/lidar_threshold_sweep.py` runs the same refinement on the annotated masks for every combination of `--h-tall` and `--tall-fraction` and counts the status of trees and shrubs. The full table is saved to `threshold_sweep.csv`:

```bash
python Python/lidar_threshold_sweep.py --min-coverage 0.3 0.5
```

With `--min-coverage 0.3` (10 shrubs in `review` in every case):

| `h_p90` $>$ | `tall_fraction` $\geq$ | Trees rejected | Trees clipped | Trees kept | Shrubs rejected | Shrubs clipped |
| --- | --- | --- | --- | --- | --- | --- |
| 1.3 | 0.1 | 21 | 0 | 0 | 1 | 6 |
| 1.3 | 0.3 | 19 | 2 | 0 | 1 | 6 |
| 1.3 | 0.5 | 14 | 7 | 0 | 0 | 7 |
| 1.5 | 0.1 | 18 | 3 | 0 | 0 | 6 |
| 1.5 | 0.3 | 14 | 7 | 0 | 0 | 6 |
| 1.5 | 0.5 | 14 | 7 | 0 | 0 | 6 |
| **1.7** | **0.1** | **14** | **4** | **3** | **0** | **0** |
| 1.7 | 0.3 | 13 | 5 | 3 | 0 | 0 |
| 1.7 | 0.5 | 12 | 6 | 3 | 0 | 0 |
| 1.9 | 0.1 | 13 | 2 | 6 | 0 | 0 |
| 1.9 | 0.3 | 12 | 3 | 6 | 0 | 0 |
| 1.9 | 0.5 | 11 | 4 | 6 | 0 | 0 |

`tall_fraction` $\geq$ 0 gives the same counts as 0.1 at every height: `h_p90` $>$ `h_tall` already implies that about 10% of the cells are taller than `h_tall`, so the tall-fraction condition only adds something from 0.2 upwards. With `--min-coverage 0.5`, *Pinus pinaster* Ids 60 and 61 (coverage 0.48 and 0.44) go to `review`, and 1.7 / 0.1 rejects 13 trees instead of 14.

At 1.5 m four more trees are rejected and no shrub is lost, but 6 shrubs are clipped, and the threshold is only 4 cm above the tallest shrub `h_p90` (*Cistus laurifolius*, 1.46 m). At 1.3 m one shrub is rejected. The 1.7 threshold is kept because it is above every shrub cell (1.63 m), so no shrub mask is rejected or clipped in the calibration sample.

Three trees are only `keep`. Ids 44 (*Crataegus monogyna*, `h_p90` of 1.37), 65 (*Quercus ilex*, 1.35) and 60 (*Pinus pinaster*, 1.44) have no cell above 1.7 m. Their `h_p90` is below the tallest *Cistus laurifolius* (1.46), so height alone cannot remove them. These could be low resprouts after the 2003 fire, and *Crataegus monogyna* often grows as a tall shrub. Four more trees are `clipped` (Ids 38, 45, 58 and 66, `h_p90` between 1.55 and 1.69 m): their tall cells are removed and the rest of the mask is kept.

### Leave-one-ROI-out check

Because the thresholds are chosen on the same masks, `--loo` repeats the choice 26 times, each time leaving out the masks of one `square_rois` square. Each mask belongs to the ROI it overlaps most, or to the nearest one (3 masks lie up to 0.85 m outside every ROI). The thresholds are selected on the other ROIs with a fixed rule and applied to the held-out masks. The rule picks the combination that rejects most trees among those that do not reject or clip any shrub (`no-shrub-change`); ties go to the higher `h_tall`. On all masks this rule gives 1.7 / 0.1. The folds are saved to `threshold_sweep_loo.csv`:

```bash
python Python/lidar_threshold_sweep.py --loo
```

| Rule | Thresholds selected (folds) | Held-out trees rejected / clipped / kept | Held-out shrubs rejected / clipped |
| --- | --- | --- | --- |
| `no-shrub-change` | 1.7 / 0.1 (25), 1.9 / 0.1 (1) | 13 / 4 / 4 | 0 / 0 |
| `no-shrub-loss` | 1.5 / 0.1 (25), 1.3 / 0.1 (1) | 18 / 3 / 0 | 1 / 5 |

With the rule used in the paper, the held-out result loses one tree compared with the calibration sample (13 against 14 rejected): in one fold 1.9 ties with 1.7 once a tree is left out, and the tie goes to 1.9. No held-out shrub is rejected or clipped. With `no-shrub-loss` (`--criterion no-shrub-loss`), 1.5 m is chosen, but when the ROI with the tallest *Cistus laurifolius* is held out, the rule drops to 1.3 m and rejects that shrub. This shows that a threshold just above the tallest shrub `h_p90` does not generalise as well as one above every shrub cell.

`h_p90` was preferred over `h_max` because it is robust to isolated returns. *Lavandula stoechas* Ids 78 and 79 have `h_max` of 1.54 and 1.52 m but `h_p90` of 0.22 and 0.33 m, probably single stray cells. These thresholds (`--min-coverage 0.3 --h-tall 1.7 --tall-fraction 0.1`) are applied unchanged to the three models in the next section.
