"""
Evaluate a Detectron2 Mask R-CNN model on COCO-format shrub datasets.

Images are resized using Detectron2's configured test transformation.
Predicted masks and boxes are returned in the original image coordinate
system before comparison with the ground-truth annotations.

The script computes:

* micro-averaged pixel precision, recall, and F1;
* one-to-one mask-IoU instance precision, recall, and F1;
* directional S-IoU metrics;
* confidence-threshold sweeps saved as CSV files; and
* F1-versus-confidence plots.

Inference is performed at each image's native resolution. Detectron2 may pad
images internally to satisfy backbone size divisibility, but it does not resize
or geometrically augment them in this script.


Example usage:

python evaluate_detectron2.py ^
  --weights model_final.pth ^
  --dataset shrubs_test_pi ^
    ./JuniperMapper/Photo_Interpretation_Data/Test/Annotations/Test.json ^
    ./JuniperMapper/Photo_Interpretation_Data/Test/Images ^
  --dataset shrubs_test_fw ^
    ./JuniperMapper/Field_Work_Data/External_Val_Data/Annotations/Val.json ^
    ./JuniperMapper/Field_Work_Data/External_Val_Data/Images ^
  --output-dir ./project/results/evaluation/model_detectron2

python "/mnt/c/IRANZO/shrubs-segmentation/Python/detectron2_evaluate.py" --weights "/mnt/c/IRANZO/shrubs-segmentation/data/logs/shrubs_detectron2/model_best.pth" --dataset shrubs_test_pi "/mnt/c/IRANZO/shrubs-segmentation/data/JuniperMapper/Photo_Interpretation_Data/Test/Annotations/Test.json" "/mnt/c/IRANZO/shrubs-segmentation/data/JuniperMapper/Photo_Interpretation_Data/Test/Images" --dataset shrubs_test_fw "/mnt/c/IRANZO/shrubs-segmentation/data/JuniperMapper/Field_Work_Data/External_Val_Data/Annotations/Val.json" "/mnt/c/IRANZO/shrubs-segmentation/data/JuniperMapper/Field_Work_Data/External_Val_Data/Images" --output-dir "/mnt/c/IRANZO/shrubs-segmentation/results/evaluation/model_detectron2"
"""
from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from matplotlib import pyplot as plt

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.detection_utils import annotations_to_instances, read_image

sys.path.append(str(Path(Path(__file__).absolute().parent.parent, 'bin')))
from shrubs_detectron import ScaleMatchedPredictor
from shrubs_detectron import build_inference_cfg
from shrubs_detectron import inspect_dataset
from shrubs_detectron import register_juniper_dataset


Prediction = Mapping[str, np.ndarray]
PredictionCache = Mapping[int, Prediction]


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    json_file: Path
    image_root: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a Detectron2 Mask R-CNN model using the "
            "same test resizing configured during training."
        )
    )

    parser.add_argument(
        "--weights",
        type=Path,
        required=True,
        help="Path to the trained Detectron2 .pth checkpoint.",
    )
    parser.add_argument(
        "--config",
        default="COCO-InstanceSegmentation/mask_rcnn_R_101_C4_3x.yaml",
        help="Detectron2 model-zoo config name or a local YAML path.",
    )
    parser.add_argument("--num-classes", type=int, default=1)
    parser.add_argument(
        "--dataset",
        action="append",
        nargs=3,
        metavar=("NAME", "COCO_JSON", "IMAGE_ROOT"),
        required=True,
        help=(
            "Dataset definition. Repeat for multiple test sets, for example: "
            "--dataset shrubs_test_pi annotations.json Images"
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/evaluation"))
    parser.add_argument(
        "--class-id",
        type=int,
        default=0,
        help="Detectron2 contiguous class ID. One-class COCO datasets normally use 0.",
    )
    parser.add_argument(
        "--confidence-thresholds",
        type=float,
        nargs="*",
        default=None,
        help="Threshold sweep. Default: 0.00, 0.05, ..., 1.00.",
    )
    parser.add_argument(
        "--evaluation-thresholds",
        type=float,
        nargs="+",
        default=(0.50, 0.75),
        help="IoU and S-IoU thresholds.",
    )
    parser.add_argument(
        "--pixel-confidence",
        type=float,
        default=0.50,
        help="Confidence threshold used for the standalone pixel evaluation.",
    )
    parser.add_argument("--minimum-intersection-pixels", type=int, default=1)
    parser.add_argument("--device", choices=("cuda", "cpu"), default=None)
    parser.add_argument("--show-plots", action="store_true")
    return parser.parse_args()


def validate_probability(value: float, name: str, *, allow_zero: bool = True) -> float:
    value = float(value)
    lower_valid = value >= 0.0 if allow_zero else value > 0.0
    if not np.isfinite(value) or not lower_valid or value > 1.0:
        interval = "[0, 1]" if allow_zero else "(0, 1]"
        raise ValueError(f"{name} must be within {interval}; received {value}.")
    return value


def harmonic_mean(precision: float, recall: float) -> float:
    denominator = precision + recall
    return 2.0 * precision * recall / denominator if denominator > 0 else 0.0


def precision_recall_f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    return precision, recall, harmonic_mean(precision, recall)


def register_dataset(spec: DatasetSpec) -> None:
    if spec.name in DatasetCatalog.list():
        DatasetCatalog.remove(spec.name)
        MetadataCatalog.remove(spec.name)

    register_juniper_dataset(
        dataset_name=spec.name,
        json_file=spec.json_file,
        image_root=spec.image_root
    )


def load_ground_truth(record: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return boxes, contiguous class IDs, and masks for one COCO record."""
    height = int(record["height"])
    width = int(record["width"])
    annotations = [ann for ann in record.get("annotations", []) if ann.get("iscrowd", 0) == 0]

    if not annotations:
        return (
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
            np.zeros((height, width, 0), dtype=bool),
        )

    instances = annotations_to_instances(annotations, (height, width), mask_format="bitmask")
    boxes = instances.gt_boxes.tensor.numpy().astype(np.float32, copy=False)
    class_ids = instances.gt_classes.numpy().astype(np.int64, copy=False)
    masks = np.moveaxis(instances.gt_masks.tensor.numpy().astype(bool, copy=False), 0, -1)
    return boxes, class_ids, masks


def predict_dataset(
    predictor: ScaleMatchedPredictor,
    records: Sequence[Mapping[str, Any]],
    *,
    verbose: bool = True,
) -> dict[int, Prediction]:
    predictions: dict[int, Prediction] = {}
    for index, record in enumerate(records, start=1):
        image = read_image(record["file_name"], format=predictor.input_format)
        expected_shape = (int(record["height"]), int(record["width"]))
        if image.shape[:2] != expected_shape:
            raise ValueError(
                f"Image dimensions differ from COCO metadata for {record['file_name']}: "
                f"image={image.shape[:2]}, metadata={expected_shape}."
            )
        result = predictor(image)
        if result["masks"].shape[:2] != image.shape[:2]:
            raise RuntimeError("Predicted masks were not returned at the original image size.")
        predictions[int(record["image_id"])] = result
        if verbose:
            print(f"Predicted image {index}/{len(records)}: {record['file_name']}")
    return predictions


def filter_instances(
    boxes: np.ndarray,
    class_ids: np.ndarray,
    masks: np.ndarray,
    target_class_id: int,
    scores: np.ndarray | None = None,
    confidence_threshold: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    keep = class_ids == target_class_id
    if scores is not None:
        keep &= scores >= confidence_threshold
    return boxes[keep], class_ids[keep], masks[..., keep], None if scores is None else scores[keep]


def mask_iou_matrix(pred_masks: np.ndarray, gt_masks: np.ndarray) -> np.ndarray:
    num_pred = pred_masks.shape[-1]
    num_gt = gt_masks.shape[-1]
    if num_pred == 0 or num_gt == 0:
        return np.zeros((num_pred, num_gt), dtype=np.float32)
    pred_flat = pred_masks.reshape(-1, num_pred).astype(np.uint8)
    gt_flat = gt_masks.reshape(-1, num_gt).astype(np.uint8)
    intersections = pred_flat.T.astype(np.int64) @ gt_flat.astype(np.int64)
    pred_areas = pred_flat.sum(axis=0, dtype=np.int64)[:, None]
    gt_areas = gt_flat.sum(axis=0, dtype=np.int64)[None, :]
    unions = pred_areas + gt_areas - intersections
    return np.divide(
        intersections,
        unions,
        out=np.zeros_like(intersections, dtype=np.float32),
        where=unions > 0,
    )


def one_to_one_matches(
    pred_masks: np.ndarray,
    pred_scores: np.ndarray,
    gt_masks: np.ndarray,
    iou_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Greedy score-ordered matching equivalent to standard detection evaluation."""
    order = np.argsort(pred_scores, kind="stable")[::-1]
    ious = mask_iou_matrix(pred_masks[..., order], gt_masks)
    pred_match_sorted = np.full(len(order), -1, dtype=np.int32)
    gt_match = np.full(gt_masks.shape[-1], -1, dtype=np.int32)

    for sorted_pred_index in range(len(order)):
        candidates = np.argsort(ious[sorted_pred_index])[::-1]
        for gt_index in candidates:
            if ious[sorted_pred_index, gt_index] < iou_threshold:
                break
            if gt_match[gt_index] != -1:
                continue
            pred_match_sorted[sorted_pred_index] = gt_index
            gt_match[gt_index] = sorted_pred_index
            break

    pred_match = np.full(len(order), -1, dtype=np.int32)
    pred_match[order] = pred_match_sorted
    return pred_match, gt_match


def compute_s_iou_matches(
    gt_class_ids: np.ndarray,
    gt_masks: np.ndarray,
    pred_class_ids: np.ndarray,
    pred_scores: np.ndarray,
    pred_masks: np.ndarray,
    *,
    s_iou_threshold: float,
    prediction_score_threshold: float,
    minimum_intersection_pixels: int,
) -> dict[str, np.ndarray]:
    score_keep = pred_scores >= prediction_score_threshold
    pred_class_ids = pred_class_ids[score_keep]
    pred_scores = pred_scores[score_keep]
    pred_masks = pred_masks[..., score_keep]

    order = np.argsort(pred_scores, kind="stable")[::-1]
    pred_class_ids = pred_class_ids[order]
    pred_scores = pred_scores[order]
    pred_masks = pred_masks[..., order]

    num_pred = pred_masks.shape[-1]
    num_gt = gt_masks.shape[-1]
    pred_s_iou = np.zeros(num_pred, dtype=np.float32)
    gt_s_iou = np.zeros(num_gt, dtype=np.float32)

    if num_pred > 0 and num_gt > 0:
        pred_pixels = pred_masks.reshape(-1, num_pred).astype(np.int64)
        gt_pixels = gt_masks.reshape(-1, num_gt).astype(np.int64)
        intersections = pred_pixels.T @ gt_pixels
        candidates = (
            pred_class_ids[:, None] == gt_class_ids[None, :]
        ) & (intersections >= minimum_intersection_pixels)

        for pred_index in range(num_pred):
            matching_gt = np.flatnonzero(candidates[pred_index])
            if matching_gt.size:
                gt_union = np.any(gt_masks[..., matching_gt], axis=-1)
                union_area = int(np.count_nonzero(gt_union))
                if union_area:
                    pred_s_iou[pred_index] = np.count_nonzero(
                        pred_masks[..., pred_index] & gt_union
                    ) / union_area

        for gt_index in range(num_gt):
            matching_pred = np.flatnonzero(candidates[:, gt_index])
            gt_area = int(np.count_nonzero(gt_masks[..., gt_index]))
            if matching_pred.size and gt_area:
                pred_union = np.any(pred_masks[..., matching_pred], axis=-1)
                gt_s_iou[gt_index] = np.count_nonzero(
                    gt_masks[..., gt_index] & pred_union
                ) / gt_area

    return {
        "pred_is_tp": pred_s_iou >= s_iou_threshold,
        "pred_is_fp": pred_s_iou < s_iou_threshold,
        "gt_is_detected": gt_s_iou >= s_iou_threshold,
        "gt_is_fn": gt_s_iou < s_iou_threshold,
        "pred_scores": pred_scores,
    }


def evaluate_pixel_metrics(
    records: Sequence[Mapping[str, Any]],
    predictions: PredictionCache,
    *,
    class_id: int,
    confidence_threshold: float,
) -> dict[str, float | int]:
    total_tp = total_fp = total_fn = 0
    for record in records:
        _, gt_classes, gt_masks = load_ground_truth(record)
        _, _, gt_masks, _ = filter_instances(
            np.zeros((len(gt_classes), 4)), gt_classes, gt_masks, class_id
        )
        result = predictions[int(record["image_id"])]
        _, _, pred_masks, _ = filter_instances(
            result["boxes"], result["class_ids"], result["masks"], class_id,
            result["scores"], confidence_threshold,
        )
        shape = (int(record["height"]), int(record["width"]))
        gt_binary = np.any(gt_masks, axis=-1) if gt_masks.shape[-1] else np.zeros(shape, bool)
        pred_binary = np.any(pred_masks, axis=-1) if pred_masks.shape[-1] else np.zeros(shape, bool)
        total_tp += int(np.count_nonzero(pred_binary & gt_binary))
        total_fp += int(np.count_nonzero(pred_binary & ~gt_binary))
        total_fn += int(np.count_nonzero(~pred_binary & gt_binary))

    precision, recall, f1 = precision_recall_f1(total_tp, total_fp, total_fn)
    return {
        "precision": precision, "recall": recall, "f1": f1,
        "tp_pixels": total_tp, "fp_pixels": total_fp, "fn_pixels": total_fn,
    }


def evaluate_instance_iou(
    records: Sequence[Mapping[str, Any]],
    predictions: PredictionCache,
    *,
    class_id: int,
    confidence_threshold: float,
    iou_threshold: float,
) -> dict[str, float | int]:
    total_tp = total_fp = total_fn = total_gt = 0
    for record in records:
        gt_boxes, gt_classes, gt_masks = load_ground_truth(record)
        _, gt_classes, gt_masks, _ = filter_instances(gt_boxes, gt_classes, gt_masks, class_id)
        result = predictions[int(record["image_id"])]
        _, _, pred_masks, pred_scores = filter_instances(
            result["boxes"], result["class_ids"], result["masks"], class_id,
            result["scores"], confidence_threshold,
        )
        assert pred_scores is not None
        pred_match, gt_match = one_to_one_matches(pred_masks, pred_scores, gt_masks, iou_threshold)
        tp = int(np.count_nonzero(pred_match >= 0))
        fp = int(np.count_nonzero(pred_match < 0))
        fn = int(np.count_nonzero(gt_match < 0))
        total_tp += tp; total_fp += fp; total_fn += fn; total_gt += gt_masks.shape[-1]

    precision, recall, f1 = precision_recall_f1(total_tp, total_fp, total_fn)
    return {
        "precision": precision, "recall": recall, "f1": f1,
        "true_positive_predictions": total_tp,
        "false_positive_predictions": total_fp,
        "false_negative_ground_truths": total_fn,
        "num_predictions": total_tp + total_fp,
        "num_ground_truths": total_gt,
    }


def evaluate_instance_siou(
    records: Sequence[Mapping[str, Any]],
    predictions: PredictionCache,
    *,
    class_id: int,
    confidence_threshold: float,
    siou_threshold: float,
    minimum_intersection_pixels: int,
) -> dict[str, float | int]:
    total_tp = total_fp = total_detected = total_fn = total_gt = 0
    for record in records:
        gt_boxes, gt_classes, gt_masks = load_ground_truth(record)
        _, gt_classes, gt_masks, _ = filter_instances(gt_boxes, gt_classes, gt_masks, class_id)
        result = predictions[int(record["image_id"])]
        class_keep = result["class_ids"] == class_id
        values = compute_s_iou_matches(
            gt_class_ids=gt_classes,
            gt_masks=gt_masks,
            pred_class_ids=result["class_ids"][class_keep],
            pred_scores=result["scores"][class_keep],
            pred_masks=result["masks"][..., class_keep],
            s_iou_threshold=siou_threshold,
            prediction_score_threshold=confidence_threshold,
            minimum_intersection_pixels=minimum_intersection_pixels,
        )
        total_tp += int(np.count_nonzero(values["pred_is_tp"]))
        total_fp += int(np.count_nonzero(values["pred_is_fp"]))
        total_detected += int(np.count_nonzero(values["gt_is_detected"]))
        total_fn += int(np.count_nonzero(values["gt_is_fn"]))
        total_gt += gt_masks.shape[-1]

    num_predictions = total_tp + total_fp
    precision = total_tp / num_predictions if num_predictions else 0.0
    recall = total_detected / total_gt if total_gt else 0.0
    return {
        "precision": precision, "recall": recall,
        "f1": harmonic_mean(precision, recall),
        "true_positive_predictions": total_tp,
        "false_positive_predictions": total_fp,
        "detected_ground_truths": total_detected,
        "false_negative_ground_truths": total_fn,
        "num_predictions": num_predictions,
        "num_ground_truths": total_gt,
    }


def confidence_thresholds(values: Sequence[float] | None) -> list[float]:
    if values is None or len(values) == 0:
        values = np.linspace(0.0, 1.0, 21)
    return [validate_probability(value, "confidence threshold") for value in values]


def write_sweep(
    output_file: Path,
    *,
    metric_name: str,
    evaluation_threshold: float,
    thresholds: Sequence[float],
    evaluator,
) -> list[dict[str, float | int]]:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    threshold_column = f"{metric_name.lower()}_threshold"
    fieldnames = [
        threshold_column, "confidence_threshold", "precision", "recall", "f1",
        "num_predictions", "num_ground_truths",
    ]
    with output_file.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for confidence in thresholds:
            result = evaluator(confidence)
            row = {
                threshold_column: evaluation_threshold,
                "confidence_threshold": confidence,
                "precision": result["precision"],
                "recall": result["recall"],
                "f1": result["f1"],
                "num_predictions": result["num_predictions"],
                "num_ground_truths": result["num_ground_truths"],
            }
            rows.append(row); writer.writerow(row)
            print(
                f"{metric_name}={evaluation_threshold:.2f}, confidence={confidence:.2f}, "
                f"predictions={result['num_predictions']}, P={result['precision']:.4f}, "
                f"R={result['recall']:.4f}, F1={result['f1']:.4f}"
            )
    print(f"Metrics saved to: {output_file}")
    return rows


def plot_performance(csv_files: Mapping[float, Mapping[str, Path]], out_dir: Path, show: bool) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for evaluation_threshold, metric_files in csv_files.items():
        fig, ax = plt.subplots(figsize=(7, 5))
        for label, csv_path in metric_files.items():
            thresholds_pct, f1_pct = [], []
            with csv_path.open(newline="", encoding="utf-8") as stream:
                for row in csv.DictReader(stream):
                    thresholds_pct.append(float(row["confidence_threshold"]) * 100)
                    f1_pct.append(float(row["f1"]) * 100)
            ax.plot(thresholds_pct, f1_pct, marker="o", markersize=4, linewidth=1.5, label=label)
        percent = round(evaluation_threshold * 100)
        ax.set_title(f"Evaluation threshold = {percent}%")
        ax.set_xlabel("Prediction-confidence threshold (%)")
        ax.set_ylabel("F1-score (%)")
        ax.set_xlim(0, 100); ax.set_ylim(0, 100)
        ax.set_xticks(range(0, 101, 20)); ax.set_yticks(range(0, 101, 20))
        ax.grid(alpha=0.25); ax.legend(title="Evaluation metric"); fig.tight_layout()
        output = out_dir / f"evaluation_threshold_{percent}.png"
        fig.savefig(output, dpi=300, bbox_inches="tight")
        if show: plt.show()
        plt.close(fig)
        print(f"Figure saved to: {output}")


def evaluate_dataset(
    predictor: ScaleMatchedPredictor,
    spec: DatasetSpec,
    args: argparse.Namespace,
) -> None:
    records = DatasetCatalog.get(spec.name)
    output_dir = args.output_dir / spec.name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nDataset: {spec.name} ({len(records)} images)")
    predictions = predict_dataset(predictor, records, verbose=True)

    pixel = evaluate_pixel_metrics(
        records, predictions, class_id=args.class_id,
        confidence_threshold=args.pixel_confidence,
    )
    print(
        f"Pixel metrics at confidence {args.pixel_confidence:.2f}: "
        f"P={pixel['precision']:.4f}, R={pixel['recall']:.4f}, F1={pixel['f1']:.4f}, "
        f"TP={pixel['tp_pixels']}, FP={pixel['fp_pixels']}, FN={pixel['fn_pixels']}"
    )

    thresholds = confidence_thresholds(args.confidence_thresholds)
    plot_inputs: dict[float, dict[str, Path]] = {}
    for evaluation_threshold in args.evaluation_thresholds:
        evaluation_threshold = validate_probability(
            evaluation_threshold, "evaluation threshold", allow_zero=False
        )
        percent = round(evaluation_threshold * 100)
        iou_csv = output_dir / f"iou_threshold_{percent}.csv"
        siou_csv = output_dir / f"siou_threshold_{percent}.csv"

        write_sweep(
            iou_csv, metric_name="IoU", evaluation_threshold=evaluation_threshold,
            thresholds=thresholds,
            evaluator=lambda confidence, t=evaluation_threshold: evaluate_instance_iou(
                records, predictions, class_id=args.class_id,
                confidence_threshold=confidence, iou_threshold=t,
            ),
        )
        write_sweep(
            siou_csv, metric_name="S-IoU", evaluation_threshold=evaluation_threshold,
            thresholds=thresholds,
            evaluator=lambda confidence, t=evaluation_threshold: evaluate_instance_siou(
                records, predictions, class_id=args.class_id,
                confidence_threshold=confidence, siou_threshold=t,
                minimum_intersection_pixels=args.minimum_intersection_pixels,
            ),
        )
        plot_inputs[evaluation_threshold] = {"IoU": iou_csv, "S-IoU": siou_csv}

    plot_performance(plot_inputs, output_dir, args.show_plots)


def main() -> None:
    args = parse_args()
    if not args.weights.is_file():
        raise FileNotFoundError(f"Weights file not found: {args.weights}")

    specs = [
        DatasetSpec(name=name, json_file=Path(json_file), image_root=Path(image_root))
        for name, json_file, image_root in args.dataset
    ]
    for spec in specs:
        register_dataset(spec)

    for spec in specs:
        inspect_dataset(spec.name)

    predictor = ScaleMatchedPredictor(
        build_inference_cfg(
            args.weights, config=args.config, num_classes=args.num_classes, device=args.device,
        )
    )
    for spec in specs:
        evaluate_dataset(predictor, spec, args)


if __name__ == "__main__":
    main()
