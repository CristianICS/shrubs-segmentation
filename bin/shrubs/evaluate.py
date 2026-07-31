"""
Evaluation utilities for Mask R-CNN shrub segmentation.

The module provides:

* pixel-level precision, recall, and F1;
* one-to-one mask-IoU instance metrics;
* directional S-IoU instance metrics;
* confidence-threshold sweeps saved as CSV files; and
* plots of F1 against the prediction-confidence threshold.

Model inference can be cached with `predict_dataset` and reused by all
metric functions. This avoids rerunning Mask R-CNN for every confidence
threshold.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Mapping, Sequence

from matplotlib import pyplot as plt
import numpy as np

from mrcnn import utils


Prediction = Mapping[str, np.ndarray]
PredictionCache = Mapping[Any, Prediction]


def _validate_probability(
    value: float,
    name: str,
    *,
    allow_zero: bool = True,
) -> float:
    """Validate and return a finite probability-like value."""
    value = float(value)
    lower_is_valid = value >= 0.0 if allow_zero else value > 0.0

    if not np.isfinite(value) or not lower_is_valid or value > 1.0:
        interval = "[0, 1]" if allow_zero else "(0, 1]"
        raise ValueError(f"{name} must be within {interval}; received {value}.")

    return value


def _validate_mask_stack(masks: np.ndarray, name: str) -> np.ndarray:
    """Convert a mask stack to Boolean and validate its dimensions."""
    masks = np.asarray(masks, dtype=bool)
    if masks.ndim != 3:
        raise ValueError(
            f"{name} must have shape [height, width, instances], "
            f"but received {masks.shape}."
        )
    return masks


def _validate_instance_arrays(
    boxes: np.ndarray,
    class_ids: np.ndarray,
    masks: np.ndarray,
    *,
    scores: np.ndarray | None = None,
    prefix: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """Validate arrays describing a collection of object instances."""
    boxes = np.asarray(boxes)
    class_ids = np.asarray(class_ids)
    masks = _validate_mask_stack(masks, f"{prefix}_masks")
    scores = None if scores is None else np.asarray(scores)

    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError(
            f"{prefix}_boxes must have shape [instances, 4], "
            f"but received {boxes.shape}."
        )
    if class_ids.ndim != 1:
        raise ValueError(
            f"{prefix}_class_ids must be one-dimensional, "
            f"but received {class_ids.shape}."
        )
    if scores is not None and scores.ndim != 1:
        raise ValueError(
            f"{prefix}_scores must be one-dimensional, "
            f"but received {scores.shape}."
        )

    counts = {
        "boxes": boxes.shape[0],
        "class_ids": class_ids.shape[0],
        "masks": masks.shape[-1],
    }
    if scores is not None:
        counts["scores"] = scores.shape[0]

    if len(set(counts.values())) != 1:
        raise ValueError(f"Inconsistent {prefix} instance counts: {counts}.")

    return boxes, class_ids, masks, scores


def _remove_zero_box_instances(
    boxes: np.ndarray,
    class_ids: np.ndarray,
    masks: np.ndarray,
    scores: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray]:
    """Remove zero-padded instances without assuming that padding is trailing."""
    keep = ~np.all(boxes == 0, axis=1)
    filtered_scores = None if scores is None else scores[keep]
    original_indices = np.flatnonzero(keep).astype(np.int32)
    return (
        boxes[keep],
        class_ids[keep],
        masks[..., keep],
        filtered_scores,
        original_indices,
    )


def _harmonic_mean(precision: float, recall: float) -> float:
    """Return the harmonic mean of precision and recall."""
    denominator = precision + recall
    return 2.0 * precision * recall / denominator if denominator > 0 else 0.0


def _precision_recall_f1(
    true_positives: int,
    false_positives: int,
    false_negatives: int,
) -> tuple[float, float, float]:
    """Calculate conventional precision, recall, and F1."""
    precision_denominator = true_positives + false_positives
    recall_denominator = true_positives + false_negatives

    precision = (
        true_positives / precision_denominator
        if precision_denominator > 0
        else 0.0
    )
    recall = (
        true_positives / recall_denominator
        if recall_denominator > 0
        else 0.0
    )
    return precision, recall, _harmonic_mean(precision, recall)


def _validate_prediction(result: Prediction, image_shape: tuple[int, int]) -> None:
    """Validate the fields required from ``MaskRCNN.detect``."""
    required = {"rois", "class_ids", "scores", "masks"}
    missing = required.difference(result)
    if missing:
        raise KeyError(f"Prediction is missing fields: {sorted(missing)}.")

    boxes, class_ids, masks, scores = _validate_instance_arrays(
        result["rois"],
        result["class_ids"],
        result["masks"],
        scores=result["scores"],
        prefix="prediction",
    )

    if masks.shape[:2] != image_shape:
        raise ValueError(
            "Predicted masks do not match the original image dimensions: "
            f"prediction={masks.shape[:2]}, image={image_shape}."
        )

    # Keep references alive and make static analyzers aware that validation ran.
    _ = boxes, class_ids, scores


def predict_dataset(model, dataset, *, verbose: bool = False) -> dict[Any, Prediction]:
    """
    Run inference once for every image and return a prediction cache.

    Parameters
    ----------
    model
        Mask R-CNN model created with ``mode='inference'``.
    dataset
        Prepared Mask R-CNN dataset.
    verbose
        Print progress while running inference.
    """
    predictions: dict[Any, Prediction] = {}
    image_ids = list(dataset.image_ids)

    for index, image_id in enumerate(image_ids, start=1):
        image = dataset.load_image(image_id)
        result = model.detect([image], verbose=0)[0]
        _validate_prediction(result, image.shape[:2])
        predictions[image_id] = result

        if verbose:
            print(f"Predicted image {index}/{len(image_ids)}: {image_id}")

    return predictions


def _get_prediction(
    model,
    image: np.ndarray,
    image_id: Any,
    predictions: PredictionCache | None,
) -> Prediction:
    """Retrieve a cached prediction or run inference for one image."""
    if predictions is None:
        if model is None:
            raise ValueError("A model or a prediction cache must be provided.")
        result = model.detect([image], verbose=0)[0]
    else:
        try:
            result = predictions[image_id]
        except KeyError as error:
            raise KeyError(
                f"No cached prediction was found for image ID {image_id!r}."
            ) from error

    _validate_prediction(result, image.shape[:2])
    return result


def _filter_class_predictions(
    result: Prediction,
    class_id: int,
    confidence_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Select predictions of one class above a confidence threshold."""
    keep = (
        (np.asarray(result["class_ids"]) == class_id)
        & (np.asarray(result["scores"]) >= confidence_threshold)
    )
    return (
        np.asarray(result["rois"])[keep],
        np.asarray(result["class_ids"])[keep],
        np.asarray(result["scores"])[keep],
        np.asarray(result["masks"], dtype=bool)[..., keep],
    )


def compute_s_iou_matches(
    gt_boxes,
    gt_class_ids,
    gt_masks,
    pred_boxes,
    pred_class_ids,
    pred_scores,
    pred_masks,
    s_iou_threshold: float = 0.5,
    prediction_score_threshold: float = 0.0,
    minimum_intersection_pixels: int = 1,
):
    r"""Compute directional S-IoU instance metrics.

    Matching sets contain every same-class instance whose masks overlap by at
    least ``minimum_intersection_pixels``.

    Prediction-directed S-IoU classifies predictions as TP or FP::

        area(prediction intersect union(matching GT masks))
        ---------------------------------------------------
                 area(union(matching GT masks))

    Ground-truth-directed S-IoU classifies ground-truth instances as detected
    or false negative::

        area(GT mask intersect union(matching prediction masks))
        --------------------------------------------------------
                           area(GT mask)

    These classifications are independent. Consequently, the number of
    true-positive predictions need not equal the number of detected
    ground-truth instances.
    """
    s_iou_threshold = _validate_probability(
        s_iou_threshold, "s_iou_threshold", allow_zero=False
    )
    prediction_score_threshold = _validate_probability(
        prediction_score_threshold, "prediction_score_threshold"
    )

    if not isinstance(minimum_intersection_pixels, (int, np.integer)):
        raise TypeError("minimum_intersection_pixels must be an integer.")
    if minimum_intersection_pixels < 1:
        raise ValueError("minimum_intersection_pixels must be at least 1.")

    gt_boxes, gt_class_ids, gt_masks, _ = _validate_instance_arrays(
        gt_boxes,
        gt_class_ids,
        gt_masks,
        prefix="gt",
    )
    pred_boxes, pred_class_ids, pred_masks, pred_scores = (
        _validate_instance_arrays(
            pred_boxes,
            pred_class_ids,
            pred_masks,
            scores=pred_scores,
            prefix="prediction",
        )
    )
    assert pred_scores is not None

    if gt_masks.shape[:2] != pred_masks.shape[:2]:
        raise ValueError(
            "Ground-truth and prediction masks must have the same height and "
            f"width; received {gt_masks.shape[:2]} and {pred_masks.shape[:2]}."
        )

    gt_boxes, gt_class_ids, gt_masks, _, _ = _remove_zero_box_instances(
        gt_boxes, gt_class_ids, gt_masks
    )
    (
        pred_boxes,
        pred_class_ids,
        pred_masks,
        pred_scores,
        original_pred_indices,
    ) = _remove_zero_box_instances(
        pred_boxes, pred_class_ids, pred_masks, pred_scores
    )
    assert pred_scores is not None

    score_keep = pred_scores >= prediction_score_threshold
    pred_boxes = pred_boxes[score_keep]
    pred_class_ids = pred_class_ids[score_keep]
    pred_scores = pred_scores[score_keep]
    pred_masks = pred_masks[..., score_keep]
    original_pred_indices = original_pred_indices[score_keep]

    score_order = np.argsort(pred_scores, kind="stable")[::-1]
    pred_boxes = pred_boxes[score_order]
    pred_class_ids = pred_class_ids[score_order]
    pred_scores = pred_scores[score_order]
    pred_masks = pred_masks[..., score_order]
    original_pred_indices = original_pred_indices[score_order]

    num_gt = gt_masks.shape[-1]
    num_pred = pred_masks.shape[-1]
    pred_s_iou = np.zeros(num_pred, dtype=np.float32)
    gt_s_iou = np.zeros(num_gt, dtype=np.float32)

    # Empty collections require explicit classifications. Comparing zero-valued
    # S-IoU arrays with a threshold is not sufficient for threshold zero.
    if num_pred == 0 or num_gt == 0:
        return {
            "pred_s_iou": pred_s_iou,
            "gt_s_iou": gt_s_iou,
            "pred_is_tp": np.zeros(num_pred, dtype=bool),
            "pred_is_fp": np.ones(num_pred, dtype=bool),
            "gt_is_detected": np.zeros(num_gt, dtype=bool),
            "gt_is_fn": np.ones(num_gt, dtype=bool),
            "pred_matching_gt": [
                np.empty(0, dtype=np.int32) for _ in range(num_pred)
            ],
            "gt_matching_pred": [
                np.empty(0, dtype=np.int32) for _ in range(num_gt)
            ],
            "original_prediction_indices": original_pred_indices,
            "pred_scores": pred_scores,
        }

    # int32 is sufficient while the number of image pixels fits in int32.
    num_pixels = int(np.prod(gt_masks.shape[:2]))
    integer_dtype = (
        np.int32 if num_pixels <= np.iinfo(np.int32).max else np.int64
    )
    pred_pixels = pred_masks.reshape(-1, num_pred).astype(integer_dtype)
    gt_pixels = gt_masks.reshape(-1, num_gt).astype(integer_dtype)
    intersections = pred_pixels.T @ gt_pixels

    same_class = (
        pred_class_ids[:, np.newaxis] == gt_class_ids[np.newaxis, :]
    )
    candidate_matches = same_class & (
        intersections >= minimum_intersection_pixels
    )

    gt_areas = gt_pixels.sum(axis=0, dtype=np.int64)
    pred_matching_gt: list[np.ndarray] = []

    for pred_index in range(num_pred):
        matching_gt_indices = np.flatnonzero(
            candidate_matches[pred_index]
        ).astype(np.int32)
        pred_matching_gt.append(matching_gt_indices)

        if matching_gt_indices.size == 0:
            continue

        matching_gt_union = np.any(
            gt_masks[..., matching_gt_indices], axis=-1
        )
        union_area = np.count_nonzero(matching_gt_union)
        if union_area == 0:
            continue

        intersection_area = np.count_nonzero(
            pred_masks[..., pred_index] & matching_gt_union
        )
        pred_s_iou[pred_index] = intersection_area / union_area

    gt_matching_pred: list[np.ndarray] = []

    for gt_index in range(num_gt):
        matching_pred_indices = np.flatnonzero(
            candidate_matches[:, gt_index]
        ).astype(np.int32)
        gt_matching_pred.append(matching_pred_indices)

        gt_area = int(gt_areas[gt_index])
        if matching_pred_indices.size == 0 or gt_area == 0:
            continue

        matching_pred_union = np.any(
            pred_masks[..., matching_pred_indices], axis=-1
        )
        intersection_area = np.count_nonzero(
            gt_masks[..., gt_index] & matching_pred_union
        )
        gt_s_iou[gt_index] = intersection_area / gt_area

    pred_is_tp = pred_s_iou >= s_iou_threshold
    pred_is_fp = ~pred_is_tp
    gt_is_detected = gt_s_iou >= s_iou_threshold
    gt_is_fn = ~gt_is_detected

    if not np.all((0.0 <= pred_s_iou) & (pred_s_iou <= 1.0)):
        raise RuntimeError(
            "Prediction-directed S-IoU contains values outside [0, 1]."
        )
    if not np.all((0.0 <= gt_s_iou) & (gt_s_iou <= 1.0)):
        raise RuntimeError(
            "Ground-truth-directed S-IoU contains values outside [0, 1]."
        )

    return {
        "pred_s_iou": pred_s_iou,
        "gt_s_iou": gt_s_iou,
        "pred_is_tp": pred_is_tp,
        "pred_is_fp": pred_is_fp,
        "gt_is_detected": gt_is_detected,
        "gt_is_fn": gt_is_fn,
        "pred_matching_gt": pred_matching_gt,
        "gt_matching_pred": gt_matching_pred,
        "original_prediction_indices": original_pred_indices,
        "pred_scores": pred_scores,
    }


def evaluate_pixel_metrics(
    model,
    dataset_test,
    shrub_class_id: int = 1,
    confidence_threshold: float = 0.5,
    *,
    predictions: PredictionCache | None = None,
):
    """Compute micro-averaged pixel precision, recall, and F1."""
    confidence_threshold = _validate_probability(
        confidence_threshold, "confidence_threshold"
    )

    total_tp = total_fp = total_fn = 0
    per_image_results = []

    for image_id in dataset_test.image_ids:
        image = dataset_test.load_image(image_id)
        gt_masks, gt_class_ids = dataset_test.load_mask(image_id)
        gt_masks = _validate_mask_stack(gt_masks, "gt_masks")
        gt_class_ids = np.asarray(gt_class_ids)

        if gt_class_ids.shape != (gt_masks.shape[-1],):
            raise ValueError(
                f"Inconsistent ground-truth counts for image {image_id!r}: "
                f"class_ids={gt_class_ids.shape[0]}, masks={gt_masks.shape[-1]}."
            )
        if gt_masks.shape[:2] != image.shape[:2]:
            raise ValueError(
                f"Ground-truth masks do not match image {image_id!r}: "
                f"masks={gt_masks.shape[:2]}, image={image.shape[:2]}."
            )

        gt_shrub_masks = gt_masks[..., gt_class_ids == shrub_class_id]
        gt_binary = (
            np.any(gt_shrub_masks, axis=-1)
            if gt_shrub_masks.shape[-1] > 0
            else np.zeros(image.shape[:2], dtype=bool)
        )

        result = _get_prediction(model, image, image_id, predictions)
        _, _, _, pred_shrub_masks = _filter_class_predictions(
            result, shrub_class_id, confidence_threshold
        )
        pred_binary = (
            np.any(pred_shrub_masks, axis=-1)
            if pred_shrub_masks.shape[-1] > 0
            else np.zeros(image.shape[:2], dtype=bool)
        )

        if pred_binary.shape != gt_binary.shape:
            raise ValueError(
                f"Mask shape mismatch for image {image_id!r}: "
                f"ground truth={gt_binary.shape}, prediction={pred_binary.shape}."
            )

        tp = int(np.count_nonzero(pred_binary & gt_binary))
        fp = int(np.count_nonzero(pred_binary & ~gt_binary))
        fn = int(np.count_nonzero(~pred_binary & gt_binary))
        total_tp += tp
        total_fp += fp
        total_fn += fn

        precision, recall, f1 = _precision_recall_f1(tp, fp, fn)
        per_image_results.append(
            {
                "image_id": image_id,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "tp_pixels": tp,
                "fp_pixels": fp,
                "fn_pixels": fn,
            }
        )

    precision, recall, f1 = _precision_recall_f1(
        total_tp, total_fp, total_fn
    )
    global_results = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp_pixels": total_tp,
        "fp_pixels": total_fp,
        "fn_pixels": total_fn,
        "confidence_threshold": confidence_threshold,
    }
    return global_results, per_image_results


def evaluate_instance_metrics(
    model,
    dataset_test,
    shrub_class_id: int = 1,
    confidence_threshold: float = 0.5,
    iou_threshold: float = 0.5,
    *,
    predictions: PredictionCache | None = None,
):
    """Compute one-to-one mask-IoU instance precision, recall, and F1."""
    confidence_threshold = _validate_probability(
        confidence_threshold, "confidence_threshold"
    )
    iou_threshold = _validate_probability(
        iou_threshold, "iou_threshold", allow_zero=False
    )

    total_tp = total_fp = total_fn = 0
    total_ground_truths = 0
    per_image_results = []

    for image_id in dataset_test.image_ids:
        image = dataset_test.load_image(image_id)
        gt_masks, gt_class_ids = dataset_test.load_mask(image_id)
        gt_masks = _validate_mask_stack(gt_masks, "gt_masks")
        gt_class_ids = np.asarray(gt_class_ids)

        if gt_class_ids.shape != (gt_masks.shape[-1],):
            raise ValueError(
                f"Inconsistent ground-truth counts for image {image_id!r}."
            )
        if gt_masks.shape[:2] != image.shape[:2]:
            raise ValueError(
                f"Ground-truth masks do not match image {image_id!r}."
            )

        gt_keep = gt_class_ids == shrub_class_id
        gt_masks = gt_masks[..., gt_keep]
        gt_class_ids = gt_class_ids[gt_keep]
        gt_boxes = utils.extract_bboxes(gt_masks)

        result = _get_prediction(model, image, image_id, predictions)
        pred_boxes, pred_class_ids, pred_scores, pred_masks = (
            _filter_class_predictions(
                result, shrub_class_id, confidence_threshold
            )
        )

        gt_match, pred_match, _ = utils.compute_matches(
            gt_boxes=gt_boxes,
            gt_class_ids=gt_class_ids,
            gt_masks=gt_masks,
            pred_boxes=pred_boxes,
            pred_class_ids=pred_class_ids,
            pred_scores=pred_scores,
            pred_masks=pred_masks,
            iou_threshold=iou_threshold,
        )

        tp = int(np.count_nonzero(pred_match > -1))
        fp = int(np.count_nonzero(pred_match == -1))
        fn = int(np.count_nonzero(gt_match == -1))
        num_gt = int(gt_masks.shape[-1])
        num_pred = int(pred_masks.shape[-1])

        if tp + fp != num_pred:
            raise RuntimeError(
                f"Prediction count mismatch for image {image_id!r}."
            )
        if tp + fn != num_gt:
            raise RuntimeError(
                f"Ground-truth count mismatch for image {image_id!r}."
            )

        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_ground_truths += num_gt
        precision, recall, f1 = _precision_recall_f1(tp, fp, fn)

        per_image_results.append(
            {
                "image_id": image_id,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "true_positive_predictions": tp,
                "false_positive_predictions": fp,
                "false_negative_ground_truths": fn,
                "num_predictions": num_pred,
                "num_ground_truths": num_gt,
            }
        )

    precision, recall, f1 = _precision_recall_f1(
        total_tp, total_fp, total_fn
    )
    global_results = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positive_predictions": total_tp,
        "false_positive_predictions": total_fp,
        "false_negative_ground_truths": total_fn,
        "num_predictions": total_tp + total_fp,
        "num_ground_truths": total_ground_truths,
        "confidence_threshold": confidence_threshold,
        "iou_threshold": iou_threshold,
    }
    return global_results, per_image_results


def evaluate_instance_metrics_s_iou(
    model,
    dataset_test,
    shrub_class_id: int = 1,
    confidence_threshold: float = 0.5,
    siou_threshold: float = 0.5,
    *,
    minimum_intersection_pixels: int = 1,
    predictions: PredictionCache | None = None,
):
    """
    Evaluate instance segmentation with directional S-IoU.

    Precision is prediction-directed::

        true-positive predictions / all retained predictions

    Recall is ground-truth-directed::

        detected ground-truth instances / all ground-truth instances

    Because matching is not one-to-one, true-positive predictions and detected
    ground-truth instances are counted independently.
    """
    confidence_threshold = _validate_probability(
        confidence_threshold, "confidence_threshold"
    )
    siou_threshold = _validate_probability(
        siou_threshold, "siou_threshold", allow_zero=False
    )

    total_tp = total_fp = total_fn = 0
    total_predictions = total_ground_truths = total_detected_gt = 0
    per_image_results = []

    for image_id in dataset_test.image_ids:
        image = dataset_test.load_image(image_id)
        gt_masks, gt_class_ids = dataset_test.load_mask(image_id)
        gt_masks = _validate_mask_stack(gt_masks, "gt_masks")
        gt_class_ids = np.asarray(gt_class_ids)

        if gt_class_ids.shape != (gt_masks.shape[-1],):
            raise ValueError(
                f"Inconsistent ground-truth counts for image {image_id!r}."
            )
        if gt_masks.shape[:2] != image.shape[:2]:
            raise ValueError(
                f"Ground-truth masks do not match image {image_id!r}."
            )

        gt_keep = gt_class_ids == shrub_class_id
        gt_masks = gt_masks[..., gt_keep]
        gt_class_ids = gt_class_ids[gt_keep]
        gt_boxes = utils.extract_bboxes(gt_masks)

        result = _get_prediction(model, image, image_id, predictions)

        # Confidence filtering is performed inside compute_s_iou_matches so
        # original prediction indices remain traceable.
        class_keep = np.asarray(result["class_ids"]) == shrub_class_id
        image_results = compute_s_iou_matches(
            gt_boxes=gt_boxes,
            gt_class_ids=gt_class_ids,
            gt_masks=gt_masks,
            pred_boxes=np.asarray(result["rois"])[class_keep],
            pred_class_ids=np.asarray(result["class_ids"])[class_keep],
            pred_scores=np.asarray(result["scores"])[class_keep],
            pred_masks=np.asarray(result["masks"], dtype=bool)[..., class_keep],
            s_iou_threshold=siou_threshold,
            prediction_score_threshold=confidence_threshold,
            minimum_intersection_pixels=minimum_intersection_pixels,
        )

        image_tp = int(np.count_nonzero(image_results["pred_is_tp"]))
        image_fp = int(np.count_nonzero(image_results["pred_is_fp"]))
        image_detected_gt = int(
            np.count_nonzero(image_results["gt_is_detected"])
        )
        image_fn = int(np.count_nonzero(image_results["gt_is_fn"]))
        image_num_predictions = image_tp + image_fp
        image_num_gt = int(gt_masks.shape[-1])

        if image_num_predictions != len(image_results["pred_scores"]):
            raise RuntimeError(
                f"Prediction count mismatch for image {image_id!r}."
            )
        if image_detected_gt + image_fn != image_num_gt:
            raise RuntimeError(
                f"Ground-truth count mismatch for image {image_id!r}."
            )

        image_precision = (
            image_tp / image_num_predictions
            if image_num_predictions > 0
            else 0.0
        )
        image_recall = (
            image_detected_gt / image_num_gt if image_num_gt > 0 else 0.0
        )
        image_f1 = _harmonic_mean(image_precision, image_recall)

        total_tp += image_tp
        total_fp += image_fp
        total_detected_gt += image_detected_gt
        total_fn += image_fn
        total_predictions += image_num_predictions
        total_ground_truths += image_num_gt

        per_image_results.append(
            {
                "image_id": image_id,
                "precision": image_precision,
                "recall": image_recall,
                "f1": image_f1,
                "true_positive_predictions": image_tp,
                "false_positive_predictions": image_fp,
                "detected_ground_truths": image_detected_gt,
                "false_negative_ground_truths": image_fn,
                "num_predictions": image_num_predictions,
                "num_ground_truths": image_num_gt,
            }
        )

    if total_tp + total_fp != total_predictions:
        raise RuntimeError("Global prediction count mismatch.")
    if total_detected_gt + total_fn != total_ground_truths:
        raise RuntimeError("Global ground-truth count mismatch.")

    precision = total_tp / total_predictions if total_predictions > 0 else 0.0
    recall = (
        total_detected_gt / total_ground_truths
        if total_ground_truths > 0
        else 0.0
    )
    f1 = _harmonic_mean(precision, recall)

    global_results = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positive_predictions": total_tp,
        "false_positive_predictions": total_fp,
        "detected_ground_truths": total_detected_gt,
        "false_negative_ground_truths": total_fn,
        "num_predictions": total_predictions,
        "num_ground_truths": total_ground_truths,
        "confidence_threshold": confidence_threshold,
        "siou_threshold": siou_threshold,
        "minimum_intersection_pixels": minimum_intersection_pixels,
    }
    return global_results, per_image_results


def _confidence_thresholds(
    thresholds: Sequence[float] | None,
) -> list[float]:
    """Validate a confidence-threshold sequence."""
    if thresholds is None:
        thresholds = np.linspace(0.0, 1.0, 21)

    validated = [
        _validate_probability(value, "confidence threshold")
        for value in thresholds
    ]
    if not validated:
        raise ValueError("At least one confidence threshold is required.")
    return validated


def compute_iou_range(
    model,
    dataset,
    out_csv_file,
    iou_threshold: float = 0.5,
    shrub_class_id: int = 1,
    verbose: bool = True,
    *,
    predictions: PredictionCache | None = None,
    confidence_thresholds: Sequence[float] | None = None,
):
    """Evaluate one-to-one IoU metrics over confidence thresholds."""
    iou_threshold = _validate_probability(
        iou_threshold, "iou_threshold", allow_zero=False
    )
    thresholds = _confidence_thresholds(confidence_thresholds)

    if predictions is None:
        predictions = predict_dataset(model, dataset, verbose=verbose)

    out_csv_file = Path(out_csv_file)
    out_csv_file.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "iou_threshold",
        "confidence_threshold",
        "precision",
        "recall",
        "f1",
        "num_predictions",
        "num_ground_truths",
    ]
    rows = []

    with out_csv_file.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for confidence_threshold in thresholds:
            results, _ = evaluate_instance_metrics(
                model=None,
                dataset_test=dataset,
                shrub_class_id=shrub_class_id,
                confidence_threshold=confidence_threshold,
                iou_threshold=iou_threshold,
                predictions=predictions,
            )
            row = {
                "iou_threshold": iou_threshold,
                "confidence_threshold": confidence_threshold,
                "precision": results["precision"],
                "recall": results["recall"],
                "f1": results["f1"],
                "num_predictions": results["num_predictions"],
                "num_ground_truths": results["num_ground_truths"],
            }
            rows.append(row)
            writer.writerow(row)

            if verbose:
                print(
                    f"IoU={iou_threshold:.2f}, "
                    f"confidence={confidence_threshold:.2f}, "
                    f"predictions={results['num_predictions']}, "
                    f"P={results['precision']:.4f}, "
                    f"R={results['recall']:.4f}, "
                    f"F1={results['f1']:.4f}"
                )

    if verbose:
        print(f"Metrics saved to: {out_csv_file}")
    return rows


def compute_siou_range(
    model,
    dataset,
    out_csv_file,
    siou_threshold: float = 0.5,
    shrub_class_id: int = 1,
    verbose: bool = True,
    *,
    minimum_intersection_pixels: int = 1,
    predictions: PredictionCache | None = None,
    confidence_thresholds: Sequence[float] | None = None,
):
    """Evaluate directional S-IoU metrics over confidence thresholds."""
    siou_threshold = _validate_probability(
        siou_threshold, "siou_threshold", allow_zero=False
    )
    thresholds = _confidence_thresholds(confidence_thresholds)

    if predictions is None:
        predictions = predict_dataset(model, dataset, verbose=verbose)

    out_csv_file = Path(out_csv_file)
    out_csv_file.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "siou_threshold",
        "confidence_threshold",
        "precision",
        "recall",
        "f1",
        "num_predictions",
        "num_ground_truths",
    ]
    rows = []

    with out_csv_file.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for confidence_threshold in thresholds:
            results, _ = evaluate_instance_metrics_s_iou(
                model=None,
                dataset_test=dataset,
                shrub_class_id=shrub_class_id,
                confidence_threshold=confidence_threshold,
                siou_threshold=siou_threshold,
                minimum_intersection_pixels=minimum_intersection_pixels,
                predictions=predictions,
            )
            row = {
                "siou_threshold": siou_threshold,
                "confidence_threshold": confidence_threshold,
                "precision": results["precision"],
                "recall": results["recall"],
                "f1": results["f1"],
                "num_predictions": results["num_predictions"],
                "num_ground_truths": results["num_ground_truths"],
            }
            rows.append(row)
            writer.writerow(row)

            if verbose:
                print(
                    f"S-IoU={siou_threshold:.2f}, "
                    f"confidence={confidence_threshold:.2f}, "
                    f"predictions={results['num_predictions']}, "
                    f"P={results['precision']:.4f}, "
                    f"R={results['recall']:.4f}, "
                    f"F1={results['f1']:.4f}"
                )

    if verbose:
        print(f"Metrics saved to: {out_csv_file}")
    return rows


def load_metric_data(csv_path):
    """Load confidence thresholds and F1 scores from an evaluation CSV."""
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    thresholds: list[float] = []
    f1_scores: list[float] = []

    with csv_path.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None:
            raise ValueError(f"No header found in {csv_path}.")

        required_columns = {"confidence_threshold", "f1"}
        missing_columns = required_columns.difference(reader.fieldnames)
        if missing_columns:
            raise ValueError(
                f"Missing columns in {csv_path}: {sorted(missing_columns)}. "
                f"Available columns: {reader.fieldnames}."
            )

        for row_number, row in enumerate(reader, start=2):
            try:
                confidence_threshold = float(row["confidence_threshold"])
                f1 = float(row["f1"])
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Invalid numeric value in {csv_path}, row {row_number}."
                ) from error

            if not np.isfinite(confidence_threshold) or not np.isfinite(f1):
                raise ValueError(
                    f"Non-finite value in {csv_path}, row {row_number}."
                )
            _validate_probability(
                confidence_threshold,
                f"confidence_threshold in row {row_number}",
            )
            _validate_probability(f1, f"f1 in row {row_number}")
            thresholds.append(confidence_threshold * 100.0)
            f1_scores.append(f1 * 100.0)

    if not thresholds:
        raise ValueError(f"No data rows found in {csv_path}.")

    sorted_pairs = sorted(zip(thresholds, f1_scores))
    sorted_thresholds, sorted_f1 = zip(*sorted_pairs)
    return list(sorted_thresholds), list(sorted_f1)


def plot_performance(csv_files, out_dir, *, show: bool = False):
    """
    Plot F1 against confidence threshold for one or more metric files.

    `csv_files` must map an evaluation threshold to a mapping whose keys are
    legend labels and whose values are CSV paths.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_files = []

    for evaluation_threshold, metric_files in csv_files.items():
        evaluation_threshold = _validate_probability(
            evaluation_threshold,
            "evaluation threshold",
            allow_zero=False,
        )
        fig, ax = plt.subplots(figsize=(7, 5))

        for metric_name, csv_path in metric_files.items():
            thresholds, f1_scores = load_metric_data(csv_path)
            ax.plot(
                thresholds,
                f1_scores,
                marker="o",
                markersize=4,
                linewidth=1.5,
                label=metric_name,
            )

        evaluation_percent = round(evaluation_threshold * 100)
        ax.set_title(f"Evaluation threshold = {evaluation_percent}%")
        ax.set_xlabel("Prediction-confidence threshold (%)")
        ax.set_ylabel("F1-score (%)")
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.set_xticks(range(0, 101, 20))
        ax.set_yticks(range(0, 101, 20))
        ax.grid(alpha=0.25)
        ax.legend(title="Evaluation metric")
        fig.tight_layout()

        output_file = out_dir / f"evaluation_threshold_{evaluation_percent}.png"
        fig.savefig(output_file, dpi=300, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)
        output_files.append(output_file)
        print(f"Figure saved to: {output_file}")

    return output_files
