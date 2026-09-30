"""
Model-agnostic detection/segmentation metrics.

Every function here operates on plain `(boxes, class_ids, scores, masks)` arrays
(a `Prediction`) rather than any framework's dataset or output objects, so the
same code scores Detectron2 and SAM 3 predictions identically. Masks are always
HxWxN (one channel per instance); boxes are XYXY.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np

Prediction = Mapping[str, np.ndarray]
PredictionCache = Mapping[int, Prediction]


def validate_probability(value: float, name: str, *, allow_zero: bool = True) -> float:
    value = float(value)
    lower_valid = value >= 0.0 if allow_zero else value > 0.0
    if not np.isfinite(value) or not lower_valid or value > 1.0:
        interval = "[0, 1]" if allow_zero else "(0, 1]"
        raise ValueError(f"{name} must be within {interval}; received {value}.")
    return value


def confidence_thresholds(values: Sequence[float] | None) -> list[float]:
    if values is None or len(values) == 0:
        values = np.linspace(0.0, 1.0, 21)
    return [validate_probability(value, "confidence threshold") for value in values]


def harmonic_mean(precision: float, recall: float) -> float:
    denominator = precision + recall
    return 2.0 * precision * recall / denominator if denominator > 0 else 0.0


def precision_recall_f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    return precision, recall, harmonic_mean(precision, recall)


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


def greedy_nms(masks: np.ndarray, scores: np.ndarray, iou_threshold: float) -> np.ndarray:
    """Return a boolean keep-mask after greedy, score-ordered mask NMS."""
    order = np.argsort(scores, kind="stable")[::-1]
    ious = mask_iou_matrix(masks[..., order], masks[..., order])
    keep_sorted = np.ones(len(order), dtype=bool)
    for i in range(len(order)):
        if keep_sorted[i]:
            keep_sorted[i + 1:][ious[i, i + 1:] >= iou_threshold] = False
    keep = np.zeros(len(order), dtype=bool)
    keep[order] = keep_sorted
    return keep


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
        "gt_is_fn": gt_s_iou < s_iou_threshold,
        "pred_scores": pred_scores,
    }


def evaluate_pixel_metrics(
    ground_truths: PredictionCache,
    predictions: PredictionCache,
    *,
    class_id: int,
    confidence_threshold: float,
) -> dict[str, float | int]:
    total_tp = total_fp = total_fn = 0
    for image_id, gt in ground_truths.items():
        _, _, gt_masks, _ = filter_instances(gt["boxes"], gt["class_ids"], gt["masks"], class_id)
        result = predictions[image_id]
        _, _, pred_masks, _ = filter_instances(
            result["boxes"], result["class_ids"], result["masks"], class_id,
            result["scores"], confidence_threshold,
        )
        shape = gt["masks"].shape[:2]
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


METRIC_DEFINITIONS = ("standard", "paper")


def best_iou_matches(
    pred_masks: np.ndarray,
    gt_masks: np.ndarray,
    iou_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Independent best-match classification used by Khaldi et al. (2024).

    A prediction is a TP when its best IoU with any label reaches the threshold;
    a label is an FN when its best IoU with any prediction does not. Labels are
    not consumed, so several predictions can be TPs against the same label.
    """
    ious = mask_iou_matrix(pred_masks, gt_masks)
    pred_is_tp = (
        ious.max(axis=1) >= iou_threshold if ious.shape[1] else np.zeros(ious.shape[0], dtype=bool)
    )
    gt_is_fn = (
        ious.max(axis=0) < iou_threshold if ious.shape[0] else np.ones(ious.shape[1], dtype=bool)
    )
    return pred_is_tp, gt_is_fn


def evaluate_instance_iou(
    ground_truths: PredictionCache,
    predictions: PredictionCache,
    *,
    class_id: int,
    confidence_threshold: float,
    iou_threshold: float,
    metric_definition: str = "standard",
) -> dict[str, float | int]:
    """Mask-IoU instance precision, recall, and F1.

    `metric_definition="standard"` uses greedy, score-ordered one-to-one matching
    (COCO-style), so TP + FN always equals the number of labels.
    `metric_definition="paper"` follows Khaldi et al. (2024): predictions and
    labels are classified independently by their best IoU (`best_iou_matches`)
    and recall is TP / (TP + FN), with TP counting predictions and FN labels.
    """
    if metric_definition not in METRIC_DEFINITIONS:
        raise ValueError(
            f"metric_definition must be one of {METRIC_DEFINITIONS}; received {metric_definition!r}."
        )
    total_tp = total_fp = total_fn = total_gt = 0
    for image_id, gt in ground_truths.items():
        _, _, gt_masks, _ = filter_instances(gt["boxes"], gt["class_ids"], gt["masks"], class_id)
        result = predictions[image_id]
        _, _, pred_masks, pred_scores = filter_instances(
            result["boxes"], result["class_ids"], result["masks"], class_id,
            result["scores"], confidence_threshold,
        )
        assert pred_scores is not None
        if metric_definition == "paper":
            pred_is_tp, gt_is_fn = best_iou_matches(pred_masks, gt_masks, iou_threshold)
        else:
            pred_match, gt_match = one_to_one_matches(pred_masks, pred_scores, gt_masks, iou_threshold)
            pred_is_tp, gt_is_fn = pred_match >= 0, gt_match < 0
        total_tp += int(np.count_nonzero(pred_is_tp))
        total_fp += int(np.count_nonzero(~pred_is_tp))
        total_fn += int(np.count_nonzero(gt_is_fn))
        total_gt += gt_masks.shape[-1]

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
    ground_truths: PredictionCache,
    predictions: PredictionCache,
    *,
    class_id: int,
    confidence_threshold: float,
    siou_threshold: float,
    minimum_intersection_pixels: int,
) -> dict[str, float | int]:
    """Directional S-IoU precision, recall, and F1 as defined by Khaldi et al. (2024).

    TPs and FPs come from the prediction-directed S-IoU (paper Eq. 2) and FNs from
    the label-directed S-IoU (Eq. 3). Recall is TP / (TP + FN) (Eq. 5), so TP + FN
    need not equal the number of labels.
    """
    total_tp = total_fp = total_fn = total_gt = 0
    for image_id, gt in ground_truths.items():
        _, gt_classes, gt_masks, _ = filter_instances(gt["boxes"], gt["class_ids"], gt["masks"], class_id)
        result = predictions[image_id]
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
        total_fn += int(np.count_nonzero(values["gt_is_fn"]))
        total_gt += gt_masks.shape[-1]

    precision, recall, f1 = precision_recall_f1(total_tp, total_fp, total_fn)
    return {
        "precision": precision, "recall": recall, "f1": f1,
        "true_positive_predictions": total_tp,
        "false_positive_predictions": total_fp,
        "false_negative_ground_truths": total_fn,
        "num_predictions": total_tp + total_fp,
        "num_ground_truths": total_gt,
    }


def write_sweep(
    output_file: Path,
    *,
    metric_name: str,
    evaluation_threshold: float,
    thresholds: Sequence[float],
    evaluator: Callable[[float], dict[str, float | int]],
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


def run_predictor(
    predictor: Callable[[np.ndarray], Prediction],
    records: Sequence[Mapping[str, object]],
    image_loader: Callable[[Mapping[str, object]], np.ndarray],
    *,
    verbose: bool = True,
) -> PredictionCache:
    """Run `predictor` over every record's image and collect the results by image ID."""
    predictions: dict[int, Prediction] = {}
    for index, record in enumerate(records, start=1):
        image = image_loader(record)
        result = predictor(image)
        if result["masks"].shape[:2] != image.shape[:2]:
            raise RuntimeError("Predicted masks were not returned at the original image size.")
        predictions[int(record["image_id"])] = result
        if verbose:
            print(f"Predicted image {index}/{len(records)}: {record['file_name']}")
    return predictions
