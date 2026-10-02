"""
Run SAM 3 (zero-shot or a domain-adapted checkpoint) with a text prompt
directly on a GeoTIFF orthomosaic, tiling it internally into overlapping
windows -- the same pipeline as `detectron2_deploy.py`, with only the
predictor swapped. Overlap gives a shrub sitting on a tile seam more than one
chance to be detected whole, which `refine_detections.py`'s cross-window
grouping can then reconcile.

Writes one `<ortho_stem>_r<row>_c<col>_masks.json` per window with real
per-instance mask polygons (via `cv2.findContours`, not bounding boxes) and
that window's own affine transform + CRS embedded in `metadata`, so
`refine_detections.py` can merge them with no paired GeoTIFF tile needed on
disk (see `bin/shrubs_common/polygons.py`).

Must run where SAM 3 is installed (WSL2, conda env `sam3_shrubs` -- see
README's SAM 3 install section).

Example usage:

python Python/sam3_deploy.py

Every path and option has a project default (orthomosaic
`data/rasters/orthomosaic_clip.tif`, the domain-adapted checkpoint
`data/logs/sam3/domain_adapt/checkpoints/model_best.pt`, prompt `shrub`,
results in `results/deployment/tile_masks/sam3/<prompt>`, confidence 0.60,
448px windows with 64px overlap, ...); run with `--help` to list them. Pass
`--zero-shot` to deploy the official release model downloaded from Hugging
Face instead of the domain-adapted checkpoint.
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
    from shrubs_sam3 import TextPromptPredictor, build_processor, sam3_autocast  # noqa: E402
except ImportError as e:
    raise ImportError(
        "SAM 3 or rasterio is not installed in the active Python environment. "
        "Activate the 'sam3_shrubs' environment (WSL2) before running this script."
    ) from e

from shrubs_common import (  # noqa: E402
    check_tiling_inputs, greedy_nms, iter_windows, read_rgb_window, write_window_masks,
)

# Store the project base directory
ROOT = Path(__file__).resolve().parent.parent

DEFAULT_ORTHOMOSAIC = ROOT / "data" / "rasters" / "orthomosaic_clip.tif"
DEFAULT_CHECKPOINT = ROOT / "data" / "logs" / "sam3" / "domain_adapt" / "checkpoints" / "model_best.pt"
# One subfolder per prompt, e.g. 'brown shrub' -> sam3/brown_shrub
DEFAULT_OUTPUT_ROOT = ROOT / "results" / "deployment" / "tile_masks" / "sam3"

# SAM 3 operating point: highest confidence whose S-IoU@50 F1 on the
# JuniperMapper FW test split is within 1 point of its maximum (Khaldi et al. 2024)
DEFAULT_CONFIDENCE = 0.60


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Segment shrubs in a GeoTIFF orthomosaic with a text-prompted SAM 3 model."
    )
    parser.add_argument(
        "--orthomosaic", type=Path, default=DEFAULT_ORTHOMOSAIC,
        help="Georeferenced 8-bit RGB(A) GeoTIFF (default: %(default)s).",
    )
    parser.add_argument(
        "--checkpoint", type=Path, default=DEFAULT_CHECKPOINT,
        help="Fine-tuned or official release SAM 3 checkpoint (default: %(default)s).",
    )
    parser.add_argument(
        "--zero-shot", action="store_true",
        help="Ignore --checkpoint and download the official release checkpoint "
             "from Hugging Face (requires 'hf auth login').",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output directory for '*_masks.json' (default: "
             f"{DEFAULT_OUTPUT_ROOT}/<prompt>, spaces replaced by underscores).",
    )
    parser.add_argument("--prompt", default="shrub", help="SAM 3 text concept (default: %(default)s).")
    parser.add_argument(
        "--confidence", type=float, default=DEFAULT_CONFIDENCE,
        help="Score threshold to keep a prediction (default: %(default)s).",
    )
    parser.add_argument(
        "--min-area", type=float, default=4.0,
        help="Drop polygons smaller than this many pixels (default: %(default)s).",
    )
    parser.add_argument(
        "--nms-iou", type=float, default=1.0,
        help="Suppress same-window masks overlapping a higher-score one by at least this IoU "
             "(default: 1.0, i.e. no suppression).",
    )
    parser.add_argument(
        "--tile-size", type=int, default=448,
        help="Window size in pixels (default matches the 448px JuniperMapper crops, which SAM 3 "
             "upsamples to its 1008px input exactly as in the validation that selected the checkpoint).",
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
        "--overwrite", action="store_true",
        help="Re-run windows whose '*_masks.json' already exists.",
    )
    args = parser.parse_args()

    args.prompt = args.prompt.strip()
    if args.output is None:
        args.output = DEFAULT_OUTPUT_ROOT / args.prompt.replace(" ", "_")
    if args.zero_shot:
        args.checkpoint = None
    return args


def check_inputs(args: argparse.Namespace) -> None:
    """Fail early, before SAM 3 is loaded, on missing inputs or invalid options."""
    if args.checkpoint is not None and not args.checkpoint.is_file():
        raise FileNotFoundError(
            f"SAM 3 checkpoint not found at '{args.checkpoint}'. Run "
            "'sam3_domain_adapt.py' first, pass a checkpoint with --checkpoint, "
            "or use --zero-shot to download the official release model."
        )

    if not args.prompt:
        raise ValueError("The text prompt cannot be empty.")
    if not 0 < args.confidence <= 1 or not 0 < args.nms_iou <= 1:
        raise ValueError("--confidence and --nms-iou must be in (0, 1].")
    if args.min_area < 0:
        raise ValueError(f"--min-area must be non-negative, got {args.min_area}.")

    check_tiling_inputs(args.orthomosaic, args.tile_size, args.overlap, args.rgb_bands, args.alpha_band)

    # No contrast stretch is applied, so SAM 3 must get the same 8-bit
    # radiometry as the JuniperMapper images it was adapted on.
    with rasterio.open(args.orthomosaic) as source:
        dtypes = [source.dtypes[band - 1] for band in args.rgb_bands]
    if any(dtype != "uint8" for dtype in dtypes):
        raise ValueError(
            f"{args.orthomosaic.name}: RGB bands must be 8-bit (uint8), got {dtypes}. "
            "Convert the orthomosaic to 8-bit before deploying SAM 3."
        )


def main() -> None:
    args = parse_args()
    check_inputs(args)

    args.output.mkdir(parents=True, exist_ok=True)
    alpha_band = args.alpha_band or None

    if args.checkpoint is None:
        print("Loading the zero-shot SAM 3 release model (the first run downloads the gated checkpoint)...")
    else:
        print(f"Loading SAM 3 from {args.checkpoint}...")
    processor, device = build_processor(args.confidence, args.checkpoint)
    predictor = TextPromptPredictor(processor, args.prompt)
    print(f"Using {device.upper()} for inference.")

    with rasterio.open(args.orthomosaic) as source, sam3_autocast(device):
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

            # Sam3Processor already drops predictions below args.confidence.
            result = predictor(image)
            keep = np.ones(len(result["scores"]), dtype=bool)
            if args.nms_iou < 1.0 and keep.any():
                keep = greedy_nms(result["masks"], result["scores"], args.nms_iou)
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
