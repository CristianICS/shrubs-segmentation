"""
Pre-tile a GeoTIFF orthomosaic into overlapping RGB windows for deployment
scripts that cannot use rasterio directly.

Detectron2 and SAM 3 tile the orthomosaic internally at inference time (see
`detectron2_deploy.py`, `sam3_deploy.py`), since their environments
can use rasterio directly. Matterport's `mrcnn_tf2` conda env cannot: it has
a Windows DLL conflict between its old TensorFlow/Keras stack and GDAL.
So for Matterport only, this script does the windowed
reading up front, in an environment where rasterio works, and writes each
window as a plain `<stem>_r<row>_c<col>.png` (ordinary RGB pixel data, no
GDAL needed to read it) plus a `<stem>_r<row>_c<col>.json` sidecar carrying
that window's affine transform and CRS -- `matterport_deploy.py` then reads
these with PIL and the pure-Python `affine` package instead.

Example usage (inside the `mrcnn_masks` environment):

python Python/tile_orthomosaic.py

Every path and option has a project default (orthomosaic
`data/rasters/orthomosaic_clip.tif`, output
`data/rasters/deployment_tiles/matterport`, 448px windows with 64px overlap,
...); run with `--help` to list them, or pass `--orthomosaic` to tile another
GeoTIFF.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Catch a wrong environment
try:
    import rasterio
    from PIL import Image
    from rasterio.windows import transform as window_transform
except ImportError as e:
    raise ImportError(
        "rasterio/Pillow could not be imported in the active Python environment. "
        "Activate the 'mrcnn_masks' environment before running this script "
        "(rasterio does not work in Matterport's 'mrcnn_tf2' environment)."
    ) from e

sys.path.append(str(Path(__file__).resolve().parent.parent / "bin"))
from shrubs_common import check_tiling_inputs, iter_windows, read_rgb_window  # noqa: E402

# Store the project base directory
ROOT = Path(__file__).resolve().parent.parent

DEFAULT_ORTHOMOSAIC = ROOT / "data" / "rasters" / "orthomosaic_clip.tif"
DEFAULT_OUTPUT = ROOT / "data" / "rasters" / "deployment_tiles" / "matterport"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pre-tile a GeoTIFF orthomosaic into overlapping windows (plain PNG + transform sidecar)."
    )
    parser.add_argument(
        "--orthomosaic", type=Path, default=DEFAULT_ORTHOMOSAIC,
        help="Georeferenced 8-bit RGB(A) GeoTIFF (default: %(default)s).",
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT,
        help="Output directory for '*.png'/'*.json' pairs (default: %(default)s).",
    )
    parser.add_argument(
        "--tile-size", type=int, default=448,
        help="Window size in pixels; must match the deployment models (default: %(default)s).",
    )
    parser.add_argument(
        "--overlap", type=int, default=64,
        help="Overlap between neighbouring windows in pixels (default: %(default)s).",
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
        help="Rewrite windows whose '.png'/'.json' pair already exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    check_tiling_inputs(args.orthomosaic, args.tile_size, args.overlap, args.rgb_bands, args.alpha_band)

    args.output.mkdir(parents=True, exist_ok=True)
    alpha_band = args.alpha_band or None

    with rasterio.open(args.orthomosaic) as source:
        windows = list(iter_windows(source.width, source.height, args.tile_size, args.overlap))
        stem = args.orthomosaic.stem

        written = skipped = empty = 0
        for number, window in enumerate(windows, start=1):
            if number % 100 == 0 or number == len(windows):
                print(f"Window {number}/{len(windows)}")

            base = f"{stem}_r{int(window.row_off)}_c{int(window.col_off)}"
            png_path = args.output / f"{base}.png"
            json_path = args.output / f"{base}.json"
            if not args.overwrite and png_path.exists() and json_path.exists():
                skipped += 1
                continue

            image, valid = read_rgb_window(source, window, tuple(args.rgb_bands), alpha_band)
            if not valid.any():
                empty += 1
                continue

            Image.fromarray(image).save(png_path)
            transform = window_transform(window, source.transform)
            json_path.write_text(json.dumps({
                "transform": [transform.a, transform.b, transform.c, transform.d, transform.e, transform.f],
                "crs": str(source.crs),
            }), encoding="utf-8")
            written += 1

    print(f"Finished: {written} window(s) written; {empty} empty; {skipped} skipped.")
    print(f"Results: {args.output}")


if __name__ == "__main__":
    main()
