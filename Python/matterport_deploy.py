"""
Run the winning Matterport Mask R-CNN checkpoint on pre-tiled orthomosaic
windows.

This script is deliberately rasterio-free: Matterport's `mrcnn_tf2` conda
env has a Windows DLL conflict between its old TensorFlow/Keras stack and
GDAL (rasterio fails to import there even after a clean reinstall), which is
exactly why this repo has always kept the TF/Keras and geopandas/rasterio
stacks in separate conda envs (see README). So instead of tiling the
orthomosaic itself, this script consumes the plain `<stem>_r<row>_c<col>.png`
+ `.json` transform sidecar pairs `tile_orthomosaic.py` writes (run that
first, in an environment where rasterio works, e.g. `mrcnn_masks`) -- PIL and
the pure-Python `affine` package have no GDAL dependency, so they work fine
here.

Loads the current best checkpoint (`data/logs/matterport/best_model_configuration.json`,
selected by `select_best_experiment()` during training) through
`bin/shrubs_matterport`, and calls `model.detect()` directly on each window's
raw RGB array (matterport's stock inference API -- no Dataset wrapper
needed). Writes one `<window_stem>_masks.json` per window with real
per-instance mask polygons and that window's transform + CRS (copied from
its sidecar) embedded in `metadata` (see `bin/shrubs_common/polygons.py`),
so `refine_detections.py` can merge them with no paired GeoTIFF needed.

Example usage:

python tile_orthomosaic.py \
  --orthomosaic data/rasters/orthomosaic_clip.tif \
  --output data/rasters/deployment_tiles/matterport \
  --tile-size 448 --overlap 64
# (in mrcnn_masks, or any environment with a working rasterio)

python matterport_deploy.py \
  --tiles data/rasters/deployment_tiles/matterport \
  --output results/deployment/tile_masks/matterport \
  --confidence 0.50
# (in mrcnn_tf2)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from affine import Affine
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parent.parent / "bin"))

import mrcnn.model as modellib  # noqa: E402
from shrubs_common import write_window_masks  # noqa: E402
from shrubs_matterport import BestModelShrubInferenceConfig  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_METADATA = ROOT / "data" / "logs" / "matterport" / "best_model_configuration.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Segment shrubs in pre-tiled orthomosaic windows with the winning Matterport Mask R-CNN checkpoint."
    )
    parser.add_argument(
        "--tiles", type=Path, required=True,
        help="Directory of '<stem>.png' + '<stem>.json' pairs written by tile_orthomosaic.py.",
    )
    parser.add_argument(
        "--metadata", type=Path, default=DEFAULT_METADATA,
        help=f"best_model_configuration.json produced by training (default: {DEFAULT_METADATA}).",
    )
    parser.add_argument(
        "--weights", type=Path, default=None,
        help=(
            "Override the checkpoint path stored in --metadata. By default that path is "
            "resolved relative to the metadata file's folder."
        ),
    )
    parser.add_argument("--output", type=Path, required=True, help="Output directory for '*_masks.json'.")
    parser.add_argument("--confidence", type=float, default=0.5, help="Score threshold to keep a prediction.")
    parser.add_argument("--min-area", type=float, default=4.0, help="Drop polygons smaller than this many pixels.")
    parser.add_argument("--model-dir", type=Path, default=ROOT / "data" / "logs", help="Scratch dir for Keras logs.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def find_windows(tiles_dir: Path) -> list[Path]:
    windows = sorted(
        path for path in tiles_dir.glob("*.png")
        if path.with_suffix(".json").exists()
    )
    if not windows:
        raise FileNotFoundError(
            f"No '<stem>.png' + '<stem>.json' pairs found in {tiles_dir}. "
            "Run tile_orthomosaic.py first, in an environment with a working rasterio."
        )
    return windows


def main() -> None:
    args = parse_args()

    # Raises FileNotFoundError, suggesting --weights, if the checkpoint is missing.
    config = BestModelShrubInferenceConfig(args.metadata, weights_path=args.weights)
    weights_path = Path(config.best_model_path)

    windows = find_windows(args.tiles)
    args.output.mkdir(parents=True, exist_ok=True)

    model = modellib.MaskRCNN(mode="inference", config=config, model_dir=str(args.model_dir))
    print(f"Loading weights from {weights_path}")
    model.load_weights(str(weights_path), by_name=True)

    total_instances = total_skipped = total_empty = 0
    for number, png_path in enumerate(windows, start=1):
        if number % 50 == 0 or number == len(windows):
            print(f"Window {number}/{len(windows)}")

        output_path = args.output / f"{png_path.stem}_masks.json"
        if not args.overwrite and output_path.exists():
            total_skipped += 1
            continue

        sidecar = json.loads(png_path.with_suffix(".json").read_text(encoding="utf-8"))
        image = np.array(Image.open(png_path).convert("RGB"))

        result = model.detect([image], verbose=0)[0]
        keep = result["scores"] >= args.confidence
        masks = result["masks"][..., keep]
        scores = result["scores"][keep]
        class_ids = result["class_ids"][keep]

        written = write_window_masks(
            output_path, masks, scores, class_ids,
            Affine(*sidecar["transform"]), sidecar["crs"], args.min_area,
        )
        if written is None:
            total_empty += 1
        else:
            total_instances += int(masks.shape[-1])

    print(f"Finished: {total_instances} shrub instance(s); {total_empty} empty window(s); {total_skipped} skipped.")
    print(f"Results: {args.output}")


if __name__ == "__main__":
    main()
