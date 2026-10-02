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

python Python/tile_orthomosaic.py
# (in mrcnn_masks, or any environment with a working rasterio)

python Python/matterport_deploy.py
# (in mrcnn_tf2)

Every path and option has a project default (tiles from
`data/rasters/deployment_tiles/matterport`, the default output of
`tile_orthomosaic.py`; results in `results/deployment/tile_masks/matterport`;
confidence 0.60, ...); run with `--help` to list them.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent / "bin"))

# Catch a wrong environment
try:
    from affine import Affine
    from PIL import Image
    import mrcnn.model as modellib  # noqa: E402
    from shrubs_matterport import BestModelShrubInferenceConfig  # noqa: E402
except ImportError as e:
    raise ImportError(
        "Matterport Mask R-CNN dependencies are not installed in the active Python "
        "environment. Activate the 'mrcnn_tf2' environment before running this script."
    ) from e

from shrubs_common import write_window_masks  # noqa: E402

# Store the project base directory
ROOT = Path(__file__).resolve().parent.parent

DEFAULT_TILES = ROOT / "data" / "rasters" / "deployment_tiles" / "matterport"
DEFAULT_METADATA = ROOT / "data" / "logs" / "matterport" / "best_model_configuration.json"
DEFAULT_OUTPUT = ROOT / "results" / "deployment" / "tile_masks" / "matterport"

# Matterport operating point: highest confidence whose S-IoU@50 F1 on the
# JuniperMapper FW test split is within 1 point of its maximum (Khaldi et al. 2024)
DEFAULT_CONFIDENCE = 0.60


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Segment shrubs in pre-tiled orthomosaic windows with the winning Matterport Mask R-CNN checkpoint."
    )
    parser.add_argument(
        "--tiles", type=Path, default=DEFAULT_TILES,
        help="Directory of '<stem>.png' + '<stem>.json' pairs written by "
             "tile_orthomosaic.py (default: %(default)s).",
    )
    parser.add_argument(
        "--metadata", type=Path, default=DEFAULT_METADATA,
        help="best_model_configuration.json produced by training (default: %(default)s).",
    )
    parser.add_argument(
        "--weights", type=Path, default=None,
        help=(
            "Override the checkpoint path stored in --metadata. By default that path is "
            "resolved relative to the metadata file's folder."
        ),
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT,
        help="Output directory for '*_masks.json' (default: %(default)s).",
    )
    parser.add_argument(
        "--confidence", type=float, default=DEFAULT_CONFIDENCE,
        help="Score threshold to keep a prediction (default: %(default)s).",
    )
    parser.add_argument(
        "--min-area", type=float, default=4.0,
        help="Drop polygons smaller than this many pixels (default: %(default)s).",
    )
    parser.add_argument(
        "--model-dir", type=Path, default=ROOT / "data" / "logs",
        help="Scratch dir for Keras logs (default: %(default)s).",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Re-run windows whose '*_masks.json' already exists.",
    )
    return parser.parse_args()


def check_inputs(args: argparse.Namespace) -> None:
    """Fail early, before the model is built, on missing inputs or invalid options."""
    if not args.metadata.is_file():
        raise FileNotFoundError(
            f"Model metadata not found at '{args.metadata}'. Train the Matterport "
            "model first, or pass its 'best_model_configuration.json' with --metadata."
        )

    if args.weights is not None and not args.weights.is_file():
        raise FileNotFoundError(f"Checkpoint passed with --weights not found: '{args.weights}'.")

    if not args.tiles.is_dir():
        raise FileNotFoundError(
            f"Tiles directory not found at '{args.tiles}'. Run tile_orthomosaic.py "
            "first (in the 'mrcnn_masks' environment), or pass the folder with --tiles."
        )

    if not 0.0 <= args.confidence <= 1.0:
        raise ValueError(f"--confidence must be in [0, 1], got {args.confidence}.")
    if args.min_area < 0:
        raise ValueError(f"--min-area must be non-negative, got {args.min_area}.")


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
    check_inputs(args)

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
