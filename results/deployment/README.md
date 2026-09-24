# Deployment and accuracy assessment

Deploy all three trained models (Detectron2 Mask R-CNN, Matterport Mask R-CNN, SAM 3) over the tilled RGB orthomosaic, using the models' chosen operating points. Then score each model's shrub map against the manually digitized `field_work` plot (`data/encinacorba_annotations.gpkg`).

| Model | Weights / checkpoint | Confidence threshold |
| --- | --- | --- |
| Detectron2 Mask R-CNN | `data/logs/detectron2/model_best.pth` | 0.35 |
| Matterport Mask R-CNN | `data/logs/matterport/phase3_resnet50_seed20260803_20260816T204953/best_validation_map_checkpoint.h5` (the winning config in `best_model_configuration.json`) | 0.50 |
| SAM 3 (domain-adapted) | `data/logs/sam3/domain_adapt/checkpoints/checkpoint_10.pt` | 0.60 |

> Note: `best_model_configuration.json` records `selected_confidence_threshold: 0.85`, the value that maximized validation F1 during training's confidence sweep. Deployment instead uses 0.50 for Matterport, a deliberately looser threshold favoring recall over the training-time F1-optimal point.

## Data

- `data/rasters/orthomosaic_clip.tif`: the RGBA orthomosaic (5cm resolution), clipped to a smaller AOI around the field_work plot for faster iteration. No pre-cropped tile files are consumed for deployment any more (see below).
- `data/encinacorba_annotations.gpkg`:
  - `field_work_masks`: Manually digitized ground truth masks (EPSG:25830, 79 polygons). It contains:
    - 58 shrub masks (`shrub == 1`, mean 3.2 square meters)
    - 21 tree masks (`shrub == 0`, mean 15.2 square meters) to assess false positives.
    - The `esp_code` and `esp_scientific_name` columns include the species.
  - `square_rois`: Regions of 9 square meters where the field campaings to digitize the masks were conducted.

Th digitization is exhaustive within the plots (every shrub inside it was digitized), which is what makes plain precision/recall valid. See [Evaluation](#evaluation).

## Deployment phase: methodological considerations

All three models tile their source orthomosaic with overlap, giving a shrub sitting on a former tile seam more than one chance to be detected whole, which the merge step's cross-window grouping then reconciles into one shrub instead of two clipped halves. All three models write real per-instance mask polygons (`cv2.findContours` on the boolean instance masks), not boxes.

Shared building blocks (`bin/shrubs_common/`), used by all three models:

- `tiling.py`: `iter_windows`/`starts` (the window generator) and `read_rgb_window` (alpha-band-aware RGB read). Requires rasterio.
- `polygons.py`: `masks_to_features`/`write_window_masks` extract real per-instance polygons from a boolean mask stack and writes one `<window_stem>_masks.json` per window, with that window's own affine transform plus CRS embedded in `metadata`. No rasterio dependency, so this half works even where `tiling.py` can't be imported (see below).

The `Python/refine_detections.py` script reads the embedded transform directly. No paired GeoTIFF tile is written or needed. It still falls back to opening `<stem>.tif` in `--images` for old-style prediction files that have no embedded transform.

Per model:

- Detectron2 (`Python/detectron2_deploy.py`) must run in WSL2
  - tiles `--orthomosaic` internally (default 448px / 64px overlap, matching the JuniperMapper training crop size)
  - runs `ScaleMatchedPredictor` per window
  - writes `<ortho_stem>_r<row>_c<col>_masks.json` directly into `--output`
- SAM 3 (`Python/sam3_deploy.py`) must run in WSL2
  - Uses `bin/shrubs_sam3/utils.py:process_image`
  - Unchanged tiling (1008px / 128px overlap by default)
  - Unchanged raster (`*_shrubs.tif`/`*_shrub_confidence.tif`) plus box CSV outputs
  - One real-polygon `<image_stem>_r<row>_c<col>_masks.json` per window
- Matterport (`Python/matterport_deploy.py`)
  - Run `Python/tile_orthomosaic.py` (run in `mrcnn_masks`, has rasterio[^1]) to pre-tile the orthomosaic into plain `<stem>_r<row>_c<col>.png` plus `.json` transform-sidecar pairs;
  - Run `matterport_deploy.py` (run in `mrcnn_tf2`) to read those with PIL and the pure-Python `affine` package.
  - Calls `model.detect()` directly on each window array (matterport's stock inference API takes raw arrays)
  - Writes the same `*_masks.json` schema.

[^1] `mrcnn_tf2`'s old TensorFlow/Keras stack has a Windows DLL conflict with GDAL. rasterio cannot be imported there at all. 

After running the three model predictions, refine the masks to produce the final shrub maps using `Python/refine_detections.py` script.

## Evaluation

The script `Python/evaluate_deployment_accuracy.py` scores each model's refined, georeferenced shrub GeoJSON against `field_work_masks`. It scores by direct polygon-to-polygon matching, clipped to the `square_rois` plot geometries with no buffer (`--plot-buffer 0`, so only masks actually inside the plot are scored):

- Match GT shrub polygons to predictions by greedy one-to-one IoU matching (IoU $\geq 0.1$ to count).
- Precision = matched predictions / all in-plot predictions.
- Recall = matched GT shrubs / all GT shrubs.
- F1 = their harmonic mean.
- Tree commission rate (on `shrub == 0`): fraction of ground-truth tree polygons that have an overlapping shrub prediction.
- Match quality: mean IoU of matched shrub pairs.

The tree commission metric is a diagnostic breakdown of what a chunk of the false positives land on, not a separate test (a false positive doesn't need to land on a digitized tree to count).

> Note: The `bin/shrubs_common/metrics.py` and `bin/shrubs_matterport/evaluate.py` cannot be reuse as-is, since those operate on same-shape raster mask stacks from COCO-style datasets, not on independent real-world polygon layers.

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
  --confidence 0.35 --tile-size 448 --overlap 64

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
  --confidence 0.50

# 3. SAM 3 — conda env sam3_shrubs, runs on the orthomosaic directly (own tiler)
# Run the script for every tested prompt
python Python/sam3_deploy.py data/rasters/orthomosaic_clip.tif \
  --output results/deployment/tile_masks/sam3/shrub \
  --prompt shrub --confidence 0.60 \
  --checkpoint data/logs/sam3/domain_adapt/checkpoints/checkpoint_10.pt \
  --tile-size 1008 --overlap 128

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
