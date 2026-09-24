"""
Run the whole LiDAR refinement workflow with one command.

1. Refine the masks of every model (Detectron2, Matterport, SAM 3) with the
   same CHM thresholds (`lidar_refine_masks.py`).
2. Score the refined masks against the ground truth
   (`evaluate_deployment_accuracy.py`).

It works from any folder and needs no shell syntax, so it runs the same in a
Miniforge Prompt (cmd), PowerShell or bash. Inside the `mrcnn_masks`
environment:

python Python/run_lidar_refinement.py

All paths and thresholds default to the values used in the paper; the options
below only exist to override them.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS = ["detectron2", "matterport", "sam3"]


def run(command: list) -> None:
    """Run one step and stop with a readable message if it fails."""
    command = [str(part) for part in command]
    print("\n>", " ".join(command), flush=True)
    if subprocess.run(command, cwd=ROOT).returncode != 0:
        sys.exit("\nA step failed, see the error above. Nothing else was run.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--models", nargs="+", default=MODELS, choices=MODELS,
                        help="Models to process (default: all).")
    parser.add_argument("--predictions-dir", type=Path, default=ROOT / "results" / "deployment",
                        help="Folder with shrub_map_<model>.geojson (default: %(default)s).")
    parser.add_argument("--chm", type=Path, default=ROOT / "data" / "rasters" / "encinacorba_chm.tif")
    parser.add_argument("--ground-truth", type=Path, default=ROOT / "data" / "encinacorba_annotations.gpkg")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "lidar_refined")
    parser.add_argument("--min-coverage", type=float, default=0.3)
    parser.add_argument("--h-tall", type=float, default=1.7)
    parser.add_argument("--tall-fraction", type=float, default=0.1)
    args = parser.parse_args()

    inputs = {m: args.predictions_dir / f"shrub_map_{m}.geojson" for m in args.models}
    required = [args.chm, args.ground_truth, *inputs.values()]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        sys.exit("These input files were not found:\n  " + "\n  ".join(missing))

    try:
        import geopandas, rasterio  # noqa: F401
    except ImportError:
        sys.exit("geopandas/rasterio are missing. Activate the environment first:\n"
                 "  conda activate mrcnn_masks")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    py = sys.executable

    for model, source in inputs.items():
        run([py, ROOT / "Python" / "lidar_refine_masks.py", source,
             "--chm", args.chm,
             "--output", args.output_dir / f"model_{model}.gpkg",
             "--min-coverage", args.min_coverage,
             "--h-tall", args.h_tall,
             "--tall-fraction", args.tall_fraction])

    run([py, ROOT / "Python" / "evaluate_deployment_accuracy.py",
         "--predictions", *[args.output_dir / f"model_{m}.gpkg" for m in args.models],
         "--model-names", *args.models,
         "--ground-truth", args.ground_truth, "--layer", "field_work_masks",
         "--rois-layer", "square_rois",
         "--output-dir", args.output_dir])

    print(f"\nDone. Open {args.output_dir / 'summary.csv'} for the scores.")


if __name__ == "__main__":
    main()
