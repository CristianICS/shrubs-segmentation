# `mrcnn`

Modified from Matterport Mask R-CNN via [Ahmed Gad's Mask-RCNN-TF2 port](https://github.com/ahmedfgad/Mask-RCNN-TF2), which itself ports [Matterport's Mask_RCNN](https://github.com/matterport/Mask_RCNN) (Copyright (c) 2017 Matterport, Inc.) to TensorFlow 2. Both are released under the MIT License, and the per-file copyright headers have been kept unchanged.

The changes below were listed by comparing these files against the upstream Mask-RCNN-TF2 `master` branch. `__init__.py`, `config.py`, `parallel_model.py` and `utils.py` are identical to upstream.

## Changes

### `model.py`

- `MaskRCNN.train()` accepts two new optional arguments:
  - `validation_freq` (default `1`): run validation every N epochs. It is forwarded to `keras_model.fit`.
  - `save_default_checkpoints` (default `True`): when `False`, the default `ModelCheckpoint` callback, which saves the weights after every epoch, is not added. This lets the caller manage checkpoints itself, for example to keep only the best one.
- `MaskRCNN.set_log_dir()` accepts a new optional `model_dir` argument, and `MaskRCNN.__init__()` passes it through. When given, the log directory is `<model_dir>/<config.NAME.lower()>`, so each experiment writes into its own folder without a timestamped subfolder.
- The hardcoded upstream log directory `"//logdir//train"` was removed from `set_log_dir()`.
- The `TensorBoard` callback now uses `write_graph=False` and `profile_batch=0` to avoid a TensorFlow warning.

### `visualize.py`

- `display_images()` and `display_top_masks()` accept an optional `figsize` argument.
- The default figure size in `display_images()` now uses `14 * rows / cols` (float division) instead of `14 * rows // cols` (integer division), so figures are no longer squashed for wide grids.

## Where it is used

The project code (`bin/shrubs_matterport/`) uses these additions, in particular `validation_freq`, `save_default_checkpoints` and `model_dir`, to run the training experiments in `Notebooks/shrubs_train_experiments.ipynb`.
