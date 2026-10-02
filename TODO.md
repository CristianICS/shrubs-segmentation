## Consistency and methodology issues a reviewer would notice

- **Plot description conflicts:** the main README describes a "~261x278 m plot", while the results README describes `square_rois` as 9 m² regions.
- **Other scripts' docstring examples use old filenames:** `model_detectron2.geojson` in `evaluate_deployment_accuracy.py`, `results/predicted_shrub_masks.geojson` in `lidar_refine_masks.py`.
- **`matterport_convert_annotations.py`:**
  - `images_dict` keeps accumulating across splits, so `Train/masks_points.json` also contains the Test entries. This only doesn't cause errors because image IDs are unique across splits.
  - It truncates pixel coordinates with `int()`.
  - It has an `except Exception: continue` that skips errors silently.
  - It runs at import time.
- **Hardcoded values in the training notebook:** Phase C hardcodes `0.0025` instead of the Phase B winner, and cell 25 uses a timestamped run folder.

## Reproducibility and publication gaps
- **Data isn't available.** The orthomosaic, CHM, point cloud and trained weights (`model_best.pth`, the Matterport `.h5`) are all gitignored, and nothing says where to get them. Deposit them on Zenodo and link from the README, or say clearly that they're available on request.
- **`data/spectral_signatures/` is gitignored**, but the Licenses table says it's included under CC BY 4.0.
- **The README has unfinished parts:** empty `## LiDAR refinement` and `## Funding` sections, a typo (`masks_mrcnn` where it means `mrcnn_masks`), `cd /d/shrubs-segmentation`, and "RAM instalada".
- **Nothing is version-pinned** in `environment.yml` or `requirements.txt`. `tensorflow==2.0.0` with an unpinned numpy/opencv on Python 3.7 is likely to fail to resolve now. Export the working environments (`conda env export --from-history` plus versions).
- **`CITATION.cff` has no `version`, `date-released` or DOI.** Mint a Zenodo DOI from a GitHub release.
- **Notebooks** have out-of-order execution counts and leftover outputs (~95 KB of images). Clear them, or run them top to bottom before release.
- Update the SAM 3 codes by avoiding the windows platform handlers. The only workflow allowed is the WSL2 one.

## Minor
- `bin/` holds importable packages that are reached through `sys.path.append` hacks. A `src/` layout would be cleaner, but this is acceptable for research code.
- [shrubs_detectron/test.py](bin/shrubs_detectron/test.py) isn't a test, it's dataset validation. Rename it to something like `validation.py`.
- In `evaluate_deployment_accuracy.py`, F1 is reported as `None` instead of `0` when precision is 0.

## Readmes

- Move the main LiDAR refinement text inside the main README to one README file inside R folder.
- Remove the deployment README. Put its relevant information inside the paper.
- Put the contents inside results\lidar_refined\README.md inside the main README or in the paper.

## Paper

- Export the paper in md and try to fix the style and contents from introduction to results (do not go further the end of the Methodology section).
- Check the final tested and fixed manuscript sections with the repository to ensure both match.
- Go for the Results, making again the deployment and accuracy processeses.