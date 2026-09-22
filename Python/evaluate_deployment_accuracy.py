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
- "in the plot" means intersecting the union of the field_work polygons with
  no buffer (`--plot-buffer 0`), not their overall bounding box.

Example usage:

python evaluate_deployment_accuracy.py \
  --predictions results/deployment/model_detectron2.geojson \
    results/deployment/model_matterport.geojson \
    results/deployment/model_sam3.geojson \
  --model-names detectron2 matterport sam3 \
  --ground-truth data/encinacorba_annotations.gpkg --layer field_work \
  --output-dir results/evaluation/deployment_accuracy
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import mean

import geopandas as gpd
from shapely.geometry.base import BaseGeometry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score deployed shrub predictions against manually digitized field_work polygons."
    )
    parser.add_argument("--predictions", type=Path, nargs="+", required=True, help="One refined GeoJSON per model.")
    parser.add_argument("--model-names", nargs="+", required=True, help="Name for each --predictions entry, same order.")
    parser.add_argument("--ground-truth", type=Path, required=True, help="GeoPackage with the field_work layer.")
    parser.add_argument("--layer", default="field_work")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--plot-buffer", type=float, default=0.0,
        help="Metres to buffer the field_work polygons by when clipping predictions (default: %(default)s, i.e. no buffer).",
    )
    parser.add_argument(
        "--iou-threshold", type=float, default=0.1,
        help="Minimum IoU between a ground-truth polygon and its best-overlapping prediction to count as a match.",
    )
    args = parser.parse_args()
    if len(args.predictions) != len(args.model_names):
        parser.error("--predictions and --model-names must have the same length.")
    return args


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
    model_name: str,
    predictions_path: Path,
    gt_shrub: gpd.GeoDataFrame,
    gt_tree: gpd.GeoDataFrame,
    plot_geometry: BaseGeometry,
    iou_threshold: float,
    output_dir: Path,
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

    detail_path = output_dir / f"{model_name}_matches.csv"
    with detail_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["gt_id", "esp_code", "class", "matched", "best_iou"])
        writer.writeheader()
        for record in shrub_records:
            writer.writerow({**record, "class": "shrub"})
        for record in tree_records:
            writer.writerow({**record, "class": "tree"})

    return {
        "model": model_name,
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

    plot_geometry = gt.geometry.union_all().buffer(args.plot_buffer)

    summary = []
    for model_name, predictions_path in zip(args.model_names, args.predictions):
        print(f"Evaluating {model_name}: {predictions_path}")
        summary.append(evaluate_model(
            model_name, predictions_path, gt_shrub, gt_tree, plot_geometry, args.iou_threshold, args.output_dir,
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
