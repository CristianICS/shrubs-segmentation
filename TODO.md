## Paper

I compared the Methodology section of [main_nostyle.md](paper/main_nostyle.md#L27-L174) against the code, configs, run logs and result CSVs. Most of it matches. Three discrepancies matter, there are some smaller wording mismatches, and some steps in the code aren't described in the paper.

## Discrepancies that matter

**1. The committed deployment results predate the scoring rule the paper describes.** The paper ([L135](paper/main_nostyle.md#L135)) says an unmatched prediction counts as a false positive only if at least 50% of its area lies inside a plot. That rule (`--min-plot-share`) was only added in commit 6d1e520. Both [results/evaluation/deployment_accuracy/summary.csv](results/evaluation/deployment_accuracy/summary.csv) and [results/lidar_refined/summary.csv](results/lidar_refined/summary.csv) still have the old `n_predictions_in_plot` column. They were produced by the previous rule: any prediction touching a plot was scored, and every unmatched one was a false positive. The precision and F1 figures for the study area and for the LiDAR refinement need to be re-run before the Results can match the Methodology.

## Smaller wording mismatches

- **GNSS validation ([L153](paper/main_nostyle.md#L153)):** the paper says RMSE was computed for each target and the mean RMSE was used. The code ([validate_ground_cls.R:148-161](R/validate_ground_cls.R#L148-L161)) takes one residual per target (bilinear sample of a 5 cm TIN DTM) and computes a single RMSE across the 26 targets. PMF had the lowest (0.612 m). The sentence "DTMs for each classification method of each GNSS reference point were extracted" is also garbled.
- **Alpha-band masking ([L131](paper/main_nostyle.md#L131)):** Detectron2 and SAM3 AND their masks with the alpha band. Matterport doesn't: its PNG tiles carry no alpha, so only completely empty windows are skipped ([matterport_deploy.py:182-186](Python/matterport_deploy.py#L182-L186)).
- **"Polygons from the same window were not merged" ([L133](paper/main_nostyle.md#L133)):** two polygons from the same window are never linked directly. But grouping uses union-find, so they can still end up merged if both touch a polygon from another window.
- **4-pixel minimum ([L131](paper/main_nostyle.md#L131)):** it's applied per contour polygon after mask-to-polygon conversion, not per instance ([polygons.py:36](bin/shrubs_common/polygons.py#L36)).
- **IoU true positives ([L113](paper/main_nostyle.md#L113)):** the paper says the overlap "exceeded" the threshold; the code uses ≥.

## In the code but not in the paper

- **Detectron2:**
  - WarmupCosineLR with 2,000 warm-up iterations (warm-up factor 0.1).
  - Batch size 2, seed 42, validation every 2 epochs.
  - Tiles upsampled from 448 to an 800‑px short edge for training and inference. The paper only explains SAM3's resizing.
- **Matterport:**
  - Exact stage schedules and learning rates: heads → 4+ → all, with the learning rate halved and then divided by 10.
  - Batch size 4, SGD with momentum 0.9, weight decay 1e‑4.
  - Augmentation: flips plus 90° rotations.
  - Up to 100 detections per tile at deployment.
- **SAM3:** annotations under 4 px² are removed from the train and val sets ([training.py:36](bin/shrubs_sam3/training.py#L36)).
- **LiDAR:**
  - The paper jumps from ground classification straight to "The resulting CHM" ([L155](paper/main_nostyle.md#L155)). DTM creation (TIN on the orthomosaic grid), height normalization (`knnidw`) and CHM rasterization (`p2r`, no fill) are never described.
  - The three PTD parameter sets (30°/2 m, 30°/3 m, 40°/3 m) and the MCC/PMF parameters are not given.
- **Paper text error:** [L161](paper/main_nostyle.md#L161) repeats the definition of the tall-cell fraction.

One thing to check yourself: the S‑IoU code normalizes the prediction-directed score by the area of the union of reference masks, |P ∩ ∪G| / |∪G| ([metrics.py:155-172](bin/shrubs_common/metrics.py#L155-L172)). A prediction much larger than the shrub can therefore still score 1.0. If that matches Khaldi et al.'s Eq. 2, the paper should probably give the formula, since "S‑IoU" suggests a standard IoU.

This is the S-IoU formula.