"""
Score deployed, georeferenced shrub predictions against the manually
digitized `field_work` ground truth.

`field_work` (`data/encinacorba_annotations.gpkg`) exhaustively digitizes
every shrub (`shrub == 1`) inside its plot, plus the trees (`shrub == 0`)
there, as the false-positive-on-trees check. Because digitization is
exhaustive, a prediction inside the plot with no matching ground-truth shrub
genuinely is a false positive (it doesn't even need to land on a digitized
tree to count as one -- "not a digitized shrub" is enough), so plain
one-to-one-matched precision/recall/F1 is valid here:

- recall: fraction of ground-truth shrub polygons matched by a prediction.
- precision: fraction of in-plot predictions matched to a ground-truth shrub.
- tree_commission_rate: fraction of ground-truth tree polygons matched by a
  prediction -- a diagnostic breakdown of what a chunk of the false positives
  land on, not a separate test.
- "in the plot" means intersecting the union of the `square_rois` square
  geometries, with no buffer (`--plot-buffer 0`). This -- not the
  ground-truth polygons themselves -- is what "in the plot" is measured
  against, since the survey plots also contain bare ground where a
  prediction would be a genuine false positive despite not touching any
  digitized polygon.

Example usage (explicit files and names):

python evaluate_deployment_accuracy.py \
  --predictions results/deployment/model_detectron2.geojson \
    results/deployment/model_matterport.geojson \
    results/deployment/model_sam3.geojson \
  --model-names detectron2 matterport sam3 \
  --ground-truth data/encinacorba_annotations.gpkg --layer field_work_masks \
  --rois-layer square_rois \
  --output-dir results/evaluation/deployment_accuracy

Or, point --predictions at a folder instead to auto-discover every
'<prompt>_map_<model>.geojson' file in it (e.g. results/deployment) and infer
model names and, for SAM 3, the text prompt used, without listing files or
--model-names by hand:

python evaluate_deployment_accuracy.py \
  --predictions results/deployment \
  --ground-truth data/encinacorba_annotations.gpkg --layer field_work_masks \
  --rois-layer square_rois \
  --output-dir results/evaluation/deployment_accuracy
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import mean

import geopandas as gpd
from shapely.geometry.base import BaseGeometry

PREDICTIONS_FILENAME_SEP = "_map_"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score deployed shrub predictions against manually digitized field_work polygons."
    )
    parser.add_argument(
        "--predictions", type=Path, nargs="+", required=True,
        help=(
            "Either one refined GeoJSON per model (paired with --model-names), or a single folder "
            "containing '<prompt>_map_<model>.geojson' files to auto-discover models and names from."
        ),
    )
    parser.add_argument(
        "--model-names", nargs="+",
        help="Name for each --predictions entry, same order. Omit when --predictions is a folder.",
    )
    parser.add_argument("--ground-truth", type=Path, required=True, help="GeoPackage with the field_work and square_rois layers.")
    parser.add_argument("--layer", default="field_work_masks", help="Ground-truth shrub/tree polygon layer.")
    parser.add_argument(
        "--rois-layer", default="square_rois",
        help="Layer of survey-plot squares that defines what counts as 'in the plot' (default: %(default)s).",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--plot-buffer", type=float, default=0.0,
        help="Metres to buffer the square_rois geometries by when clipping predictions (default: %(default)s, i.e. no buffer).",
    )
    parser.add_argument(
        "--iou-threshold", type=float, default=0.1,
        help="Minimum IoU between a ground-truth polygon and its best-overlapping prediction to count as a match.",
    )
    args = parser.parse_args()

    args.predictions_dir = args.predictions[0] if len(args.predictions) == 1 and args.predictions[0].is_dir() else None
    if args.predictions_dir is not None:
        if args.model_names:
            parser.error("--model-names cannot be combined with a --predictions folder; names are inferred automatically.")
    else:
        if not args.model_names:
            parser.error("--model-names is required when --predictions lists individual files.")
        if len(args.predictions) != len(args.model_names):
            parser.error("--predictions and --model-names must have the same length.")
    return args


def model_label(run_name: str) -> str:
    """The base model type ('sam3', 'detectron2', 'matterport', ...) a run belongs to."""
    return "sam3" if run_name == "sam3" or run_name.startswith("sam3_") else run_name


def discover_predictions(directory: Path) -> tuple[list[Path], list[str], dict[str, str]]:
    """
    Auto-discover '<prompt>_map_<model>.geojson' files in `directory` (e.g.
    results/deployment). Returns predictions paths, their inferred *run*
    names, and a {run_name: prompt_text} map for the SAM 3 entries (whose
    filenames encode the text prompt used, e.g. 'brown_shrub_map_sam3.geojson'
    -> run name 'sam3_brown_shrub', prompt 'brown shrub'). Non-SAM-3 models
    (detectron2, matterport, ...) aren't prompted, so their run name is just
    the model name. A run name is a unique, file-safe identifier (used for
    each run's '<run_name>_matches.csv'); model_label() reduces it back to
    the base model type for reporting.
    """
    predictions_paths: list[Path] = []
    run_names: list[str] = []
    sam3_prompts: dict[str, str] = {}
    seen_names: set[str] = set()

    for path in sorted(directory.glob("*.geojson")):
        stem = path.stem
        if PREDICTIONS_FILENAME_SEP not in stem:
            print(f"Skipping {path.name}: doesn't match '<prompt>_map_<model>.geojson'.")
            continue
        prompt_slug, model = stem.rsplit(PREDICTIONS_FILENAME_SEP, 1)
        if model == "sam3":
            run_name = f"sam3_{prompt_slug}"
            sam3_prompts[run_name] = prompt_slug.replace("_", " ")
        else:
            run_name = model

        if run_name in seen_names:
            raise SystemExit(
                f"Duplicate inferred run name '{run_name}' from {path.name}; "
                "rename the files or fall back to --predictions/--model-names."
            )
        seen_names.add(run_name)
        predictions_paths.append(path)
        run_names.append(run_name)

    if not predictions_paths:
        raise SystemExit(f"No '<prompt>_map_<model>.geojson' files found in {directory}.")
    return predictions_paths, run_names, sam3_prompts


def iou(a: BaseGeometry, b: BaseGeometry) -> float:
    if a.is_empty or b.is_empty or not a.intersects(b):
        return 0.0
    intersection = a.intersection(b).area
    union = a.union(b).area
    return intersection / union if union else 0.0


def match_ground_truth(
    gt: gpd.GeoDataFrame,
    predictions: gpd.GeoDataFrame,
    iou_threshold: float,
) -> tuple[list[dict], set[int]]:
    """
    Greedily match each ground-truth polygon to its best-overlapping,
    still-unused prediction. Returns per-GT match records and the set of
    prediction row-positions used by any match (of either class).
    """
    candidates = []
    for gt_pos, gt_row in enumerate(gt.itertuples()):
        for pred_pos, pred_row in enumerate(predictions.itertuples()):
            score = iou(gt_row.geometry, pred_row.geometry)
            if score > 0:
                candidates.append((score, gt_pos, pred_pos))
    candidates.sort(key=lambda item: item[0], reverse=True)

    matched_gt: dict[int, tuple[int, float]] = {}
    used_predictions: set[int] = set()
    for score, gt_pos, pred_pos in candidates:
        if gt_pos in matched_gt or pred_pos in used_predictions:
            continue
        matched_gt[gt_pos] = (pred_pos, score)
        used_predictions.add(pred_pos)

    records = []
    matched_predictions: set[int] = set()
    for gt_pos, gt_row in enumerate(gt.itertuples()):
        pred_pos, score = matched_gt.get(gt_pos, (None, 0.0))
        is_match = pred_pos is not None and score >= iou_threshold
        if is_match:
            matched_predictions.add(pred_pos)
        records.append({
            "gt_id": gt_row.Id,
            "esp_code": gt_row.esp_code,
            "matched": is_match,
            "best_iou": round(score, 4),
        })
    return records, matched_predictions


def evaluate_model(
    run_name: str,
    model: str,
    predictions_path: Path,
    gt_shrub: gpd.GeoDataFrame,
    gt_tree: gpd.GeoDataFrame,
    plot_geometry: BaseGeometry,
    iou_threshold: float,
    output_dir: Path,
    sam3_prompt: str | None = None,
) -> dict:
    predictions = gpd.read_file(predictions_path).to_crs(gt_shrub.crs)
    predictions = predictions[predictions.intersects(plot_geometry)].reset_index(drop=True)

    shrub_records, matched_by_shrub = match_ground_truth(gt_shrub, predictions, iou_threshold)
    tree_records, matched_by_tree = match_ground_truth(gt_tree, predictions, iou_threshold)

    n_matched_shrub = sum(record["matched"] for record in shrub_records)
    n_matched_tree = sum(record["matched"] for record in tree_records)
    matched_ious = [record["best_iou"] for record in shrub_records if record["matched"]]

    # Exhaustive shrub digitization means every in-plot prediction not matched
    # to a real shrub is a genuine false positive, whether or not it happens
    # to land on a digitized tree.
    true_positives = len(matched_by_shrub)
    false_positives = len(predictions) - true_positives
    false_negatives = len(gt_shrub) - n_matched_shrub
    precision = true_positives / len(predictions) if len(predictions) else None
    recall = n_matched_shrub / len(gt_shrub) if len(gt_shrub) else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision and recall and (precision + recall) > 0
        else None
    )

    detail_fieldnames = ["gt_id", "esp_code", "class", "matched", "best_iou"]
    if sam3_prompt is not None:
        detail_fieldnames.append("sam3_prompt")
    detail_path = output_dir / f"{run_name}_matches.csv"
    with detail_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=detail_fieldnames)
        writer.writeheader()
        for record in shrub_records:
            row = {**record, "class": "shrub"}
            if sam3_prompt is not None:
                row["sam3_prompt"] = sam3_prompt
            writer.writerow(row)
        for record in tree_records:
            row = {**record, "class": "tree"}
            if sam3_prompt is not None:
                row["sam3_prompt"] = sam3_prompt
            writer.writerow(row)

    return {
        "model": model,
        "sam3_prompt": sam3_prompt,
        "n_gt_shrub": len(gt_shrub),
        "n_matched_shrub": n_matched_shrub,
        "n_predictions_in_plot": len(predictions),
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": round(precision, 4) if precision is not None else None,
        "recall": round(recall, 4) if recall is not None else None,
        "f1": round(f1, 4) if f1 is not None else None,
        "mean_matched_iou": round(mean(matched_ious), 4) if matched_ious else None,
        "n_gt_tree": len(gt_tree),
        "n_matched_tree": n_matched_tree,
        "tree_commission_rate": round(n_matched_tree / len(gt_tree), 4) if len(gt_tree) else None,
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    gt = gpd.read_file(args.ground_truth, layer=args.layer)
    gt_shrub = gt[gt["shrub"] == 1].reset_index(drop=True)
    gt_tree = gt[gt["shrub"] == 0].reset_index(drop=True)

    rois = gpd.read_file(args.ground_truth, layer=args.rois_layer).to_crs(gt.crs)
    plot_geometry = rois.geometry.union_all().buffer(args.plot_buffer)

    if args.predictions_dir is not None:
        predictions_paths, run_names, sam3_prompts = discover_predictions(args.predictions_dir)
    else:
        predictions_paths, run_names, sam3_prompts = args.predictions, args.model_names, {}

    summary = []
    for run_name, predictions_path in zip(run_names, predictions_paths):
        print(f"Evaluating {run_name}: {predictions_path}")
        summary.append(evaluate_model(
            run_name, model_label(run_name), predictions_path, gt_shrub, gt_tree, plot_geometry,
            args.iou_threshold, args.output_dir, sam3_prompt=sam3_prompts.get(run_name),
        ))

    summary_path = args.output_dir / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)

    print(f"\nGround truth: {len(gt_shrub)} shrub / {len(gt_tree)} tree polygons in the field_work plot.")
    for row in summary:
        print(
            f"  {row['model']}: precision={row['precision']}, recall={row['recall']}, f1={row['f1']}, "
            f"tree_commission_rate={row['tree_commission_rate']}, "
            f"mean_matched_iou={row['mean_matched_iou']}"
        )
    print(f"\nSummary: {summary_path}")


if __name__ == "__main__":
    main()
