"""Run zero-shot SAM 3 shrub segmentation on GeoTIFFs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BIN_DIR = PROJECT_ROOT / "bin"
if str(BIN_DIR) not in sys.path:
    sys.path.insert(0, str(BIN_DIR))

from shrubs_sam3 import parse_rgb_bands, process_input


def probability(value: str) -> float:
    number = float(value)
    if not 0 < number <= 1:
        raise argparse.ArgumentTypeError("must be greater than 0 and at most 1")
    return number


def existing_path(value: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.exists():
        raise argparse.ArgumentTypeError(f"path does not exist: {path}")
    return path


def existing_file(value: str) -> Path:
    path = existing_path(value)
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"file does not exist: {path}")
    return path


def rgb_bands(value: str) -> tuple[int, int, int]:
    try:
        return parse_rgb_bands(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Zero-shot SAM 3 shrub inference on GeoTIFFs (no fine-tuning)."
    )
    parser.add_argument("input", type=existing_path, help="GeoTIFF or folder of GeoTIFFs.")
    parser.add_argument("--output", type=Path, required=True, help="Output directory.")
    parser.add_argument(
        "--prompt", default="shrub",
        help="SAM 3 text concept (default: %(default)s).",
    )
    parser.add_argument("--confidence", type=probability, default=0.70)
    parser.add_argument("--rgb-bands", type=rgb_bands, default=(1, 2, 3), metavar="R,G,B")
    parser.add_argument("--tile-size", type=int, default=1008)
    parser.add_argument("--overlap", type=int, default=128)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--checkpoint", type=existing_file,
        help="Optional local official SAM 3 release checkpoint.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.prompt.strip():
        raise SystemExit("The text prompt cannot be empty.")
    if args.tile_size < 256:
        raise SystemExit("Tile size must be at least 256 pixels.")
    if args.overlap < 0 or args.overlap >= args.tile_size:
        raise SystemExit("Overlap must be non-negative and smaller than tile size.")

    output = args.output.expanduser().resolve()
    results = process_input(
        input_path=args.input,
        output_folder=output,
        prompt=args.prompt.strip(),
        confidence=args.confidence,
        rgb_bands=args.rgb_bands,
        tile_size=args.tile_size,
        overlap=args.overlap,
        checkpoint=args.checkpoint,
        overwrite=args.overwrite,
    )
    detected = sum(int(result["instances"]) for result in results)
    skipped = sum(int(result["skipped"]) for result in results)
    print(f"Finished: {detected} tile-level shrub candidates; {skipped} image(s) skipped.")
    print(f"Results: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
