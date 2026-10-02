"""
Run a trained Detectron2 Mask R-CNN checkpoint directly on a GeoTIFF
orthomosaic, tiling it internally into (optionally overlapping) windows --
the same approach `sam3_deploy.py` already uses -- instead of consuming a
folder of pre-cropped, non-overlapping tile files. Overlap gives a shrub
sitting on a former tile seam more than one chance to be detected whole,
which `refine_detections.py`'s cross-window grouping can then reconcile.

Writes one `<ortho_stem>_r<row>_c<col>_masks.json` per window with real
per-instance mask polygons (via `cv2.findContours`, not bounding boxes) and
that window's own affine transform + CRS embedded in `metadata`, so
`refine_detections.py` can merge them with no paired GeoTIFF tile needed on
disk (see `bin/shrubs_common/polygons.py`).

Must run where Detectron2 is installed (WSL2 -- see README's Detectron2
install section).

Example usage (inside the Detectron2 environment):

python Python/detectron2_deploy.py

Every path and option has a project default (orthomosaic
`data/rasters/orthomosaic_clip.tif`, weights
`data/logs/detectron2/model_best.pth`, results in
`results/deployment/tile_masks/detectron2`, confidence 0.45, 448px windows
with 64px overlap, ...); run with `--help` to list them.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent / "bin"))

# Catch a wrong environment
try:
    import rasterio
    from rasterio.windows import transform as window_transform
    from shrubs_detectron import ScaleMatchedPredictor, build_inference_cfg  # noqa: E402
except ImportError as e:
    raise ImportError(
        "Detectron2 or rasterio is not installed in the active Python environment. "
        "Activate the Detectron2 environment (WSL2) before running this script."
    ) from e

from shrubs_common import check_tiling_inputs, iter_windows, read_rgb_window, write_window_masks  # noqa: E402

# Store the project base directory
ROOT = Path(__file__).resolve().parent.parent

DEFAULT_ORTHOMOSAIC = ROOT / "data" / "rasters" / "orthomosaic_clip.tif"
DEFAULT_WEIGHTS = ROOT / "data" / "logs" / "detectron2" / "model_best.pth"
DEFAULT_OUTPUT = ROOT / "results" / "deployment" / "tile_masks" / "detectron2"

# Same architecture used in detectron2_shrubs_train.py
DEFAULT_CONFIG = "COCO-InstanceSegmentation/mask_rcnn_R_101_C4_3x.yaml"

# Detectron2 operating point: highest confidence whose S-IoU@50 F1 on the
# JuniperMapper FW test split is within 1 point of its maximum (Khaldi et al. 2024)
DEFAULT_CONFIDENCE = 0.45


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Segment shrubs in a GeoTIFF orthomosaic with a trained Detectron2 Mask R-CNN checkpoint."
    )
    parser.add_argument(
        "--orthomosaic", type=Path, default=DEFAULT_ORTHOMOSAIC,
        help="Georeferenced 8-bit RGB(A) GeoTIFF (default: %(default)s).",
    )
    parser.add_argument(
        "--weights", type=Path, default=DEFAULT_WEIGHTS,
        help="Trained Detectron2 .pth checkpoint (default: %(default)s).",
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT,
        help="Output directory for '*_masks.json' (default: %(default)s).",
    )
    parser.add_argument(
        "--config", default=DEFAULT_CONFIG,
        help="Detectron2 model-zoo config name or a local YAML path; must match "
             "the weights (default: %(default)s).",
    )
    parser.add_argument(
        "--num-classes", type=int, default=1,
        help="Number of classes the weights were trained with (default: %(default)s).",
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
        "--tile-size", type=int, default=448,
        help="Window size in pixels; matches the JuniperMapper training crop "
             "size (default: %(default)s).",
    )
    parser.add_argument(
        "--overlap", type=int, default=64,
        help="Overlap in pixels between adjacent windows (default: %(default)s).",
    )
    parser.add_argument(
        "--rgb-bands", type=int, nargs=3, default=(1, 2, 3), metavar=("R", "G", "B"),
        help="1-based band indices read as RGB (default: 1 2 3).",
    )
    parser.add_argument(
        "--alpha-band", type=int, default=4,
        help="Band used to detect empty/nodata pixels, 0 to disable (default: %(default)s).",
    )
    parser.add_argument(
        "--device", choices=("cuda", "cpu"), default=None,
        help="Inference device (default: CUDA if available, otherwise CPU).",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Re-run windows whose '*_masks.json' already exists.",
    )
    return parser.parse_args()


def check_inputs(args: argparse.Namespace) -> None:
    """Fail early, before the model is built, on missing inputs or invalid options."""
    if not args.weights.is_file():
        raise FileNotFoundError(
            f"Detectron2 weights not found at '{args.weights}'. Run "
            "'detectron2_shrubs_train.py' first, or pass the checkpoint with --weights."
        )

    if not 0.0 <= args.confidence <= 1.0:
        raise ValueError(f"--confidence must be in [0, 1], got {args.confidence}.")
    if args.min_area < 0:
        raise ValueError(f"--min-area must be non-negative, got {args.min_area}.")

    check_tiling_inputs(args.orthomosaic, args.tile_size, args.overlap, args.rgb_bands, args.alpha_band)


def main() -> None:
    args = parse_args()
    check_inputs(args)

    args.output.mkdir(parents=True, exist_ok=True)
    alpha_band = args.alpha_band or None

    predictor = ScaleMatchedPredictor(
        build_inference_cfg(
            args.weights, config=args.config, num_classes=args.num_classes, device=args.device,
        )
    )

    with rasterio.open(args.orthomosaic) as source:
        windows = list(iter_windows(source.width, source.height, args.tile_size, args.overlap))
        stem = args.orthomosaic.stem

        total_instances = total_skipped = total_empty = 0
        for number, window in enumerate(windows, start=1):
            if number % 50 == 0 or number == len(windows):
                print(f"Window {number}/{len(windows)}")

            output_path = args.output / f"{stem}_r{int(window.row_off)}_c{int(window.col_off)}_masks.json"
            if not args.overwrite and output_path.exists():
                total_skipped += 1
                continue

            image, valid = read_rgb_window(source, window, tuple(args.rgb_bands), alpha_band)
            if not valid.any():
                total_empty += 1
                continue

            # ScaleMatchedPredictor expects BGR (cfg.INPUT.FORMAT), matching what
            # detectron2_evaluate.py gets from detectron2's own read_image().
            result = predictor(image[..., ::-1] if predictor.input_format == "BGR" else image)
            keep = result["scores"] >= args.confidence
            masks = result["masks"][..., keep] & valid[..., None]
            scores = result["scores"][keep]
            class_ids = result["class_ids"][keep]

            written = write_window_masks(
                output_path, masks, scores, class_ids,
                window_transform(window, source.transform), source.crs, args.min_area,
            )
            if written is None:
                total_empty += 1
            else:
                total_instances += int(masks.shape[-1])

    print(f"Finished: {total_instances} shrub instance(s); {total_empty} empty window(s); {total_skipped} skipped.")
    print(f"Results: {args.output}")


if __name__ == "__main__":
    main()
