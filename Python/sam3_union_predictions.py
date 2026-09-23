"""
Union SAM 3's per-prompt shrub maps into a single combined prediction, then
score it against `field_work_masks` like any other model.

`results/deployment/` holds one refined shrub map per SAM 3 text prompt
tested (`shrub_map_sam3.geojson`, `brown_shrub_map_sam3.geojson`, ...,
discovered the same way `evaluate_deployment_accuracy.py` does). Different
prompts often re-detect the same real shrub, so this script dissolves
polygons that overlap or touch across prompts into one shape -- the same
connected-component + `unary_union` approach `refine_detections.py` uses to
reconcile a shrub split across a tile seam -- so a shrub caught by more than
one prompt counts once in the union.

The union is written to `results/deployment/sam3_union.geojson`, then scored
with `evaluate_deployment_accuracy.evaluate_model()` and folded into the
existing `summary.csv` as a `model=sam3` row (`sam3_prompt` marks it as the
union, distinguishing it from the per-prompt `sam3` rows already there), plus
its own `sam3_union_matches.csv` detail file.

Example usage:

python Python/sam3_union_predictions.py \
  --deployment-dir results/deployment \
  --ground-truth data/encinacorba_annotations.gpkg --layer field_work_masks \
  --rois-layer square_rois \
  --output-dir results/evaluation/deployment_accuracy
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import geopandas as gpd
import pandas as pd

from evaluate_deployment_accuracy import discover_predictions, evaluate_model
from refine_detections import group_related_detections, refine_detection_groups

RUN_NAME = "sam3_union"
MODEL = "sam3"
PROMPT_LABEL = "union (all prompts)"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Union SAM 3's per-prompt shrub maps and score the result against field_work_masks."
    )
    parser.add_argument(
        "--deployment-dir", type=Path, default=Path("results/deployment"),
        help="Folder to auto-discover the per-prompt '<prompt>_map_sam3.geojson' files from (default: %(default)s).",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output path for the unioned GeoJSON (default: <deployment-dir>/sam3_union.geojson).",
    )
    parser.add_argument(
        "--merge-distance", type=float, default=0.05,
        help="Metres within which polygons from different prompts are dissolved together (default matches refine_detections.py's %(default)s).",
    )
    parser.add_argument("--crs", default="EPSG:25830", help="Projected CRS used for the dissolve and the output.")
    parser.add_argument("--ground-truth", type=Path, required=True, help="GeoPackage with the field_work and square_rois layers.")
    parser.add_argument("--layer", default="field_work_masks", help="Ground-truth shrub/tree polygon layer.")
    parser.add_argument(
        "--rois-layer", default="square_rois",
        help="Layer of survey-plot squares that defines what counts as 'in the plot' (default: %(default)s).",
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True,
        help="Same evaluation output dir evaluate_deployment_accuracy.py wrote to; its summary.csv gets the sam3_union row.",
    )
    parser.add_argument(
        "--plot-buffer", type=float, default=0.0,
        help="Metres to buffer the square_rois geometries by when clipping predictions (default: %(default)s, i.e. no buffer).",
    )
    parser.add_argument(
        "--iou-threshold", type=float, default=0.1,
        help="Minimum IoU between a ground-truth polygon and its best-overlapping prediction to count as a match.",
    )
    return parser.parse_args()


def union_sam3_predictions(deployment_dir: Path, crs: str, merge_distance: float) -> gpd.GeoDataFrame:
    """Dissolve every SAM 3 prompt variant's shrub map in `deployment_dir` into one."""
    predictions_paths, run_names, sam3_prompts = discover_predictions(deployment_dir)
    sam3_entries = [
        (path, sam3_prompts[run_name])
        for path, run_name in zip(predictions_paths, run_names)
        if run_name in sam3_prompts
    ]
    if not sam3_entries:
        raise SystemExit(f"No SAM 3 prediction files found in {deployment_dir}.")

    frames = []
    for path, prompt in sam3_entries:
        gdf = gpd.read_file(path).to_crs(crs)
        frames.append(gpd.GeoDataFrame({
            "geometry": gdf.geometry,
            "score": gdf["score_max"].astype(float),
            "class_id": gdf["class_id"],
            "image_name": prompt,
        }, geometry="geometry", crs=crs))
    detections = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), geometry="geometry", crs=crs)

    groups = group_related_detections(
        detections, merge_distance=merge_distance, minimum_overlap_ratio=0.0, merge_same_tile=True,
    )
    unioned = refine_detection_groups(detections, groups, close_gaps=True, merge_distance=merge_distance)
    unioned = unioned.rename(columns={
        "shrub_id": "union_shrub_id",
        "source_tiles": "source_prompts",
        "source_tile_count": "source_prompt_count",
    })

    print(f"SAM 3 prompts unioned: {len(sam3_entries)}")
    print(f"Individual detections: {len(detections)}")
    print(f"Unioned shrubs: {len(unioned)}")
    return unioned


def save_union(unioned: gpd.GeoDataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    to_save = unioned.copy()
    to_save["source_prompts"] = to_save["source_prompts"].apply(json.dumps)
    to_save["original_scores"] = to_save["original_scores"].apply(json.dumps)
    to_save.to_file(output_path, driver="GeoJSON", index=False)
    print(f"Saved union: {output_path}")


def update_summary(output_dir: Path, new_row: dict) -> None:
    """Replace/append `new_row` in the existing summary.csv, keeping every other row."""
    summary_path = output_dir / "summary.csv"
    rows = []
    if summary_path.exists():
        with summary_path.open("r", newline="", encoding="utf-8") as file:
            rows = list(csv.DictReader(file))
    rows = [row for row in rows if not (row.get("model") == new_row["model"] and row.get("sam3_prompt") == new_row["sam3_prompt"])]
    rows.append(new_row)

    with summary_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(new_row.keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Updated summary: {summary_path}")


def main() -> None:
    args = parse_args()
    output_path = args.output or (args.deployment_dir / "sam3_union.geojson")

    unioned = union_sam3_predictions(args.deployment_dir, args.crs, args.merge_distance)
    save_union(unioned, output_path)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    gt = gpd.read_file(args.ground_truth, layer=args.layer)
    gt_shrub = gt[gt["shrub"] == 1].reset_index(drop=True)
    gt_tree = gt[gt["shrub"] == 0].reset_index(drop=True)
    rois = gpd.read_file(args.ground_truth, layer=args.rois_layer).to_crs(gt.crs)
    plot_geometry = rois.geometry.union_all().buffer(args.plot_buffer)

    row = evaluate_model(
        RUN_NAME, MODEL, output_path, gt_shrub, gt_tree, plot_geometry,
        args.iou_threshold, args.output_dir, sam3_prompt=PROMPT_LABEL,
    )
    update_summary(args.output_dir, row)

    print(
        f"\n{MODEL} ({PROMPT_LABEL}): precision={row['precision']}, recall={row['recall']}, f1={row['f1']}, "
        f"tree_commission_rate={row['tree_commission_rate']}, mean_matched_iou={row['mean_matched_iou']}"
    )


if __name__ == "__main__":
    main()
