# Deployment and accuracy assessment

> Note: the `*_map_sam3.geojson` files and `tile_masks/sam3` outputs currently in this folder were produced with the deprecated `checkpoint_10.pt` (epoch 10) and must be regenerated with `model_best.pt` (epoch 19). The Detectron2 and Matterport maps were produced with the earlier thresholds (0.35 and 0.50) and must also be regenerated.

### Operating point selection

Following Khaldi et al. (2024), each model's operating point is read from its F1 vs. confidence curve on the JuniperMapper Field Work (FW) test split, the set closest to real-world data. The selected threshold is **the highest confidence threshold whose S-IoU@50% F1 is within 1 percentage point of the curve's maximum**. Among thresholds that perform almost equally well, this keeps the most confident one, which limits false-positive masks in the deployed shrub maps. The curves are the `siou_threshold_50.csv` files in `results/evaluation/model_<model>/shrubs_test_fw/`.

| Model | Maximum S-IoU@50% F1 (threshold) | Selected threshold (F1) |
| --- | --- | --- |
| Detectron2 Mask R-CNN | 0.758 (0.35) | 0.45 (0.752) |
| Matterport Mask R-CNN | 0.687 (0.00) | 0.60 (0.681) |
| SAM 3 (domain-adapted) | 0.783 (0.60) | 0.60 (0.783) |

Matterport's FW curve is almost flat (its maximum is at 0.00), so the rule is what sets its threshold. The `selected_confidence_threshold: 0.85` in `best_model_configuration.json` comes from the training-time sweep on the PI validation split and is not used for deployment. As a robustness check, applying the same rule to the IoU@50% curves (`iou_paper_threshold_50.csv`) gives 0.50, 0.75 and 0.60 respectively.

## Deployment phase: methodological considerations

All three models tile their source orthomosaic with overlap, giving a shrub sitting on a former tile seam more than one chance to be detected whole, which the merge step's cross-window grouping then reconciles into one shrub instead of two clipped halves. All three models write real per-instance mask polygons (`cv2.findContours` on the boolean instance masks), not boxes.

Shared building blocks (`bin/shrubs_common/`), used by all three models:

- `tiling.py`: `iter_windows`/`starts` (the window generator) and `read_rgb_window` (alpha-band-aware RGB read). Requires rasterio.
- `polygons.py`: `masks_to_features`/`write_window_masks` extract real per-instance polygons from a boolean mask stack and writes one `<window_stem>_masks.json` per window, with that window's own affine transform plus CRS embedded in `metadata`. No rasterio dependency, so this half works even where `tiling.py` can't be imported (see below).

The `Python/refine_detections.py` script reads the embedded transform directly. No paired GeoTIFF tile is written or needed. Prediction files without an embedded transform and CRS are skipped.

Per model:

- Detectron2 (`Python/detectron2_deploy.py`) must run in WSL2
  - tiles `--orthomosaic` internally (default 448px / 64px overlap, matching the JuniperMapper training crop size)
  - runs `ScaleMatchedPredictor` per window
  - writes `<ortho_stem>_r<row>_c<col>_masks.json` directly into `--output`
- SAM 3 (`Python/sam3_deploy.py`) must run in WSL2
  - Same loop as `detectron2_deploy.py`, with `TextPromptPredictor` (`bin/shrubs_sam3/utils.py`) as the predictor
  - Tiles `--orthomosaic` with the same 448px / 64px overlap windows as Detectron2. SAM 3 upsamples every input to 1008px, so 448px windows reproduce the scale of the validation that selected `model_best.pt` (448px JuniperMapper crops upsampled to 1008px; training used random sizes between 480 and 1008px). Native 1008px windows would show shrubs at less than half that scale, below the smallest one seen in training.
  - Reads each window with `read_rgb_window` (raw 8-bit RGB, alpha band for validity), like Detectron2 and Matterport. An earlier version applied a per-tile 2–98% percentile stretch, which shifted the radiometry away from the JuniperMapper training images and made the same shrub look different in overlapping windows.
  - Same `--min-area 4` polygon filter; optional same-window mask NMS (`--nms-iou`, off by default)
  - writes `<ortho_stem>_r<row>_c<col>_masks.json` directly into `--output` (no raster or CSV outputs)
- Matterport (`Python/matterport_deploy.py`)
  - Run `Python/tile_orthomosaic.py` (run in `mrcnn_masks`, has rasterio[^1]) to pre-tile the orthomosaic into plain `<stem>_r<row>_c<col>.png` plus `.json` transform-sidecar pairs;
  - Run `matterport_deploy.py` (run in `mrcnn_tf2`) to read those with PIL and the pure-Python `affine` package.
  - Calls `model.detect()` directly on each window array (matterport's stock inference API takes raw arrays)
  - Writes the same `*_masks.json` schema.

[^1] `mrcnn_tf2`'s old TensorFlow/Keras stack has a Windows DLL conflict with GDAL. rasterio cannot be imported there at all. 

After running the three model predictions, refine the masks to produce the final shrub maps using `Python/refine_detections.py` script.

## Evaluation

The script `Python/evaluate_deployment_accuracy.py` scores each model's refined, georeferenced shrub GeoJSON against `field_work_masks`. :

- Match GT shrub polygons to predictions by greedy one-to-one IoU matching (IoU $\geq 0.1$ to count).
- Precision = matched predictions / all in-plot predictions.
- Recall = matched GT shrubs / all GT shrubs.
- F1 = their harmonic mean.
- Tree commission rate (on `shrub == 0`): fraction of ground-truth tree polygons that have an overlapping shrub prediction.
- Match quality: mean IoU of matched shrub pairs.

The tree commission metric is a diagnostic breakdown of what a chunk of the false positives land on, not a separate test (a false positive doesn't need to land on a digitized tree to count).

> Note: The `bin/shrubs_common/metrics.py` metrics cannot be reused as-is, since those operate on same-shape raster mask stacks from COCO-style datasets, not on independent real-world polygon layers.

The `field_work_masks` layer contains every shrub inside the validation plots (`square_rois` layer). Because digitization is exhaustive, plain one-to-one-matched precision/recall/F1 is valid. Any in-plot prediction not matched to a real shrub genuinely is a false positive, whether it lands on a digitized tree or anywhere else. 

After the evaluation, a summary with the statistics per model is created with the following variables: `model`, `sam3_prompt`, `n_gt_shrub`, `n_matched_shrub`, `n_predictions_in_plot`, `true_positives`, `false_positives`, `false_negatives`, `precision`, `recall`, `f1`, `mean_matched_iou`, `n_gt_tree`, `n_matched_tree`, `tree_commission_rate`. `model` is always the base model type (`detectron2`, `matterport`, `sam3`); `sam3_prompt` disambiguates the SAM 3 rows by the text prompt used (empty for the other models). It also writes one `<run_name>_matches.csv` per run with the per-ground-truth-polygon match detail -- `<run_name>` is `detectron2`/`matterport` as-is, or `sam3_<prompt>` for SAM 3 (e.g. `sam3_shrub_matches.csv`, `sam3_brown_shrub_matches.csv`), which also gets a `sam3_prompt` column.

Scores of the unrefined shrub maps in this folder against the 58 shrub and 21 tree polygons of `field_work_masks`. `N masks` is the number of predicted masks that intersect the plot (`TP + FP`), `Tree rate` is the tree commission rate and Precision, Recall and F1 are computed as described above (IoU $\geq 0.1$). Most models detect only 1 to 3 of the 58 shrubs, so precision rests on very few predictions.

## Reproducibility steps

```bash
# 1. Detectron2 — inside WSL2, ~/venvs/detectron2
python Python/detectron2_deploy.py \
  --orthomosaic data/rasters/orthomosaic_clip.tif \
  --weights data/logs/detectron2/model_best.pth \
  --output results/deployment/tile_masks/detectron2 \
  --confidence 0.45 --tile-size 448 --overlap 64

# 2a. Matterport pre-tiling — conda env mrcnn_masks (has rasterio)
python Python/tile_orthomosaic.py \
  --orthomosaic data/rasters/orthomosaic_clip.tif \
  --output data/rasters/deployment_tiles/matterport \
  --tile-size 448 --overlap 64

# 2b. Matterport inference — conda env mrcnn_tf2 (rasterio-free)
python Python/matterport_deploy.py \
  --tiles data/rasters/deployment_tiles/matterport \
  --weights "data/logs/matterport/phase3_resnet50_seed20260803_20260816T204953/best_validation_map_checkpoint.h5" \
  --output results/deployment/tile_masks/matterport \
  --confidence 0.60

# 3. SAM 3 — conda env sam3_shrubs, same internal tiling as Detectron2
# Run the script for every tested prompt. Empty each prompt's output folder
# first: windows from an older run (other tile size or raster name) don't get
# overwritten and would be merged by refine_detections.py.
rm -rf results/deployment/tile_masks/sam3/shrub
python Python/sam3_deploy.py \
  --orthomosaic data/rasters/orthomosaic_clip.tif \
  --checkpoint data/logs/sam3/domain_adapt/checkpoints/model_best.pt \
  --output results/deployment/tile_masks/sam3/shrub \
  --prompt shrub --confidence 0.60 --tile-size 448 --overlap 64

# 4. Merge each model's per-window detections — conda env mrcnn_masks
python Python/refine_detections.py \
  --predictions results/deployment/tile_masks/detectron2 \
  --output results/deployment/shrub_map_detectron2.geojson --crs EPSG:25830

python Python/refine_detections.py \
  --predictions results/deployment/tile_masks/matterport \
  --output results/deployment/shrub_map_matterport.geojson --crs EPSG:25830

# Note that you must run the SAM 3 refine script for each tested prompt
python Python/refine_detections.py \
  --predictions results/deployment/tile_masks/sam3/shrub \
  --output results/deployment/shrub_map_sam3.geojson --crs EPSG:25830

python Python/refine_detections.py \
  --predictions results/deployment/tile_masks/sam3/bush \
  --output results/deployment/bush_map_sam3.geojson --crs EPSG:25830
# And so on...

# 5. Accuracy assessment against field_work_masks — conda env mrcnn_masks
# --predictions can point at the results/deployment folder directly: it
# auto-discovers every '<prompt>_map_<model>.geojson' file there (including
# every SAM 3 prompt variant) and infers model names and, for SAM 3, the
# text prompt used from the filename.
python Python/evaluate_deployment_accuracy.py \
  --predictions results/deployment \
  --ground-truth data/encinacorba_annotations.gpkg --layer field_work_masks \
  --rois-layer square_rois \
  --output-dir results/evaluation/deployment_accuracy

# Equivalent, listing files and names by hand (still supported):
python Python/evaluate_deployment_accuracy.py \
  --predictions results/deployment/shrub_map_detectron2.geojson \
    results/deployment/shrub_map_matterport.geojson \
    results/deployment/shrub_map_sam3.geojson \
  --model-names detectron2 matterport sam3 \
  --ground-truth data/encinacorba_annotations.gpkg --layer field_work_masks \
  --rois-layer square_rois \
  --output-dir results/evaluation/deployment_accuracy
```

> Note: `merge_distance = 0.05 m`, inside `refine_detections.py`, was lowered from 0.2 m. It barely changed anything. Refined-shrub counts rose only ~2-6%, and precision/recall/F1 moved by hundredths of a percent. At 0.2 m, merging was already conservative enough that a further 4x reduction has little left to affect — most nearby fragments that were going to stay separate already had.
