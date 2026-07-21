from pathlib import Path
import csv

from matplotlib import pyplot as plt
import numpy as np

from mrcnn import utils
from mrcnn.utils import trim_zeros


def compute_s_iou_matches(
        gt_boxes,
        gt_class_ids,
        gt_masks,
        pred_boxes,
        pred_class_ids,
        pred_scores,
        pred_masks,
        s_iou_threshold=0.5,
        prediction_score_threshold=0.0):
    """Compute directional S-IoU instance metrics.

    Matching sets contain every spatially intersecting instance of the
    same class.

    Prediction-directed S-IoU is used to classify predictions as TP or FP:

        area(prediction \cap union(matching GT masks))
        ------------------------------------------------
                  area(union(matching GT masks))

    Ground-truth-directed S-IoU is used to classify ground-truth instances
    as detected or false negatives:

        area(GT mask \cap union(matching prediction masks))
        ------------------------------------------------
                         area(GT mask)

    This function is different from the original `compute_matches` due to
    `pred_is_tp` and `gt_is_detected` are calculated independently.
    There is no longer a one-to-one correspondence between a prediction-level
    TP and a detected ground-truth instance.

    Parameters
    ----------
    gt_boxes : np.ndarray
        Ground-truth boxes with shape [num_gt, 4].
    gt_class_ids : np.ndarray
        Ground-truth class IDs with shape [num_gt].
    gt_masks : np.ndarray
        Ground-truth masks with shape [height, width, num_gt].
    pred_boxes : np.ndarray
        Predicted boxes with shape [num_pred, 4].
    pred_class_ids : np.ndarray
        Predicted class IDs with shape [num_pred].
    pred_scores : np.ndarray
        Prediction confidence scores with shape [num_pred].
    pred_masks : np.ndarray
        Predicted masks with shape [height, width, num_pred].
    s_iou_threshold : float
        S-IoU threshold used to classify predictions and labels.
    prediction_score_threshold : float
        Minimum prediction confidence.

    Returns
    -------
    dict
        Dictionary containing directional S-IoU scores, TP/FP and
        detected/FN classifications, matching sets, prediction scores,
        and prediction ordering.
    """
    # Convert inputs to NumPy arrays.
    gt_boxes = np.asarray(gt_boxes)
    gt_class_ids = np.asarray(gt_class_ids)
    gt_masks = np.asarray(gt_masks, dtype=bool)

    pred_boxes = np.asarray(pred_boxes)
    pred_class_ids = np.asarray(pred_class_ids)
    pred_scores = np.asarray(pred_scores)
    pred_masks = np.asarray(pred_masks, dtype=bool)

    # Validate mask dimensions.
    if gt_masks.ndim != 3:
        raise ValueError(
            f"gt_masks must have shape [height, width, instances], "
            f"but received {gt_masks.shape}."
        )

    if pred_masks.ndim != 3:
        raise ValueError(
            f"pred_masks must have shape [height, width, instances], "
            f"but received {pred_masks.shape}."
        )

    if gt_masks.shape[:2] != pred_masks.shape[:2]:
        raise ValueError(
            "Ground-truth and prediction masks must have the same "
            f"height and width. Received {gt_masks.shape[:2]} and "
            f"{pred_masks.shape[:2]}."
        )


    # Remove zero-padded instances using mrcnn.utils.trim_zeros.
    #
    # Mask R-CNN stores valid instances before zero-padded instances,
    # so the corresponding class IDs and masks can be sliced using the
    # number of boxes remaining after trimming.
    gt_boxes = trim_zeros(gt_boxes)
    num_gt = gt_boxes.shape[0]

    gt_class_ids = gt_class_ids[:num_gt]
    gt_masks = gt_masks[..., :num_gt]

    pred_boxes = trim_zeros(pred_boxes)
    num_pred_before_filtering = pred_boxes.shape[0]

    pred_class_ids = pred_class_ids[:num_pred_before_filtering]
    pred_scores = pred_scores[:num_pred_before_filtering]
    pred_masks = pred_masks[..., :num_pred_before_filtering]

    # Original prediction indices before confidence filtering and sorting.
    original_pred_indices = np.arange(
        num_pred_before_filtering,
        dtype=np.int32,
    )

    # Remove predictions below the model-confidence threshold
    score_keep = pred_scores >= prediction_score_threshold

    pred_boxes = pred_boxes[score_keep]
    pred_class_ids = pred_class_ids[score_keep]
    pred_scores = pred_scores[score_keep]
    pred_masks = pred_masks[..., score_keep]

    # Sort retained predictions from highest to lowest confidence.
    score_order = np.argsort(pred_scores)[::-1]

    pred_boxes = pred_boxes[score_order]
    pred_class_ids = pred_class_ids[score_order]
    pred_scores = pred_scores[score_order]
    pred_masks = pred_masks[..., score_order]

    # Map sorted predictions back to their original indices.
    prediction_order = original_pred_indices[score_order]

    num_pred = pred_masks.shape[-1]

    # Initialize output scores.
    pred_s_iou = np.zeros(num_pred, dtype=np.float32)
    gt_s_iou = np.zeros(num_gt, dtype=np.float32)

    # Handle images without predictions or ground-truth instances.
    if num_pred == 0 or num_gt == 0:
        pred_is_tp = pred_s_iou >= s_iou_threshold
        pred_is_fp = ~pred_is_tp

        gt_is_detected = gt_s_iou >= s_iou_threshold
        gt_is_fn = ~gt_is_detected

        return {
            "pred_s_iou": pred_s_iou,
            "gt_s_iou": gt_s_iou,
            "pred_is_tp": pred_is_tp,
            "pred_is_fp": pred_is_fp,
            "gt_is_detected": gt_is_detected,
            "gt_is_fn": gt_is_fn,
            "pred_matching_gt": [
                np.empty(0, dtype=np.int32)
                for _ in range(num_pred)
            ],
            "gt_matching_pred": [
                np.empty(0, dtype=np.int32)
                for _ in range(num_gt)
            ],
            "prediction_order": prediction_order,
            "pred_scores": pred_scores,
        }

    # Flatten masks to [pixels, instances].
    # Shape after flattening:
    #     pred_pixels: [pixels, num_pred]
    #     gt_pixels:   [pixels, num_gt]
    pred_pixels = pred_masks.reshape(-1, num_pred).astype(np.uint8)
    gt_pixels = gt_masks.reshape(-1, num_gt).astype(np.uint8)

    # Pairwise intersection areas with shape [num_pred, num_gt].
    intersections = (
        pred_pixels.T.astype(np.int64)
        @ gt_pixels.astype(np.int64)
    )

    # The evaluation has one shrub class, but retaining the class
    # restriction makes the function usable with multiple classes.
    # Restrict matches to instances of the same class.
    same_class = (
        pred_class_ids[:, np.newaxis]
        == gt_class_ids[np.newaxis, :]
    )

    # Khaldi et al. 2024 paper-compatible definition:
    # every spatially intersecting same-class instance is a match.
    candidate_matches = same_class & (intersections > 0)

    # Ground-truth areas are reused in the label-directed calculation.
    gt_areas = gt_pixels.sum(axis=0, dtype=np.int64)

    # Prediction-directed S-IoU
    # Used to classify predictions as TP or FP.
    pred_matching_gt = []

    for pred_index in range(num_pred):
        matching_gt_indices = np.flatnonzero(
            candidate_matches[pred_index]
        ).astype(np.int32)

        pred_matching_gt.append(matching_gt_indices)

        # No intersecting labels means S-IoU remains zero.
        if matching_gt_indices.size == 0:
            continue

        matching_gt_union = np.any(
            gt_masks[..., matching_gt_indices],
            axis=-1,
        )
        
        matching_gt_union_area = np.count_nonzero(
            matching_gt_union
        )

        if matching_gt_union_area == 0:
            continue

        intersection_area = np.count_nonzero(
            pred_masks[..., pred_index]
            & matching_gt_union
        )

        pred_s_iou[pred_index] = (
            intersection_area / matching_gt_union_area
        )

    # Ground-truth-directed S-IoU
    # Used to identify false-negative labels
    gt_matching_pred = []

    for gt_index in range(num_gt):
        matching_pred_indices = np.flatnonzero(
            candidate_matches[:, gt_index]
        ).astype(np.int32)

        gt_matching_pred.append(matching_pred_indices)

        gt_area = gt_areas[gt_index]

        # No intersecting predictions means S-IoU remains zero.
        if matching_pred_indices.size == 0 or gt_area == 0:
            continue
        
        matching_pred_union = np.any(
            pred_masks[..., matching_pred_indices],
            axis=-1,
        )
        
        intersection_area = np.count_nonzero(
            gt_masks[..., gt_index]
            & matching_pred_union
        )

        gt_s_iou[gt_index] = intersection_area / gt_area

    # Convert directional S-IoU scores into instance classifications
    pred_is_tp = pred_s_iou >= s_iou_threshold
    pred_is_fp = ~pred_is_tp

    gt_is_detected = gt_s_iou >= s_iou_threshold
    gt_is_fn = ~gt_is_detected

    # S-IoU values must remain within [0, 1].
    if not np.all((pred_s_iou >= 0.0) & (pred_s_iou <= 1.0)):
        raise RuntimeError(
            "Prediction-directed S-IoU contains values outside [0, 1]."
        )

    if not np.all((gt_s_iou >= 0.0) & (gt_s_iou <= 1.0)):
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
        "prediction_order": prediction_order,
        "pred_scores": pred_scores,
    }


def evaluate_pixel_metrics(
    model,
    dataset_test,
    shrub_class_id=1,
    confidence_threshold=0.5
):
    """
    Computes micro-averaged pixel Precision, Recall, and F1
    over the complete test dataset.

    Parameters
    ----------
    model:
        Mask R-CNN model created with mode="inference".
    dataset_test:
        Prepared Mask R-CNN test Dataset object.
    shrub_class_id:
        Internal class ID for the shrub class.
        For a single-class dataset, this is normally 1.
    confidence_threshold:
        Predictions below this confidence are ignored.
    """

    total_tp = 0
    total_fp = 0
    total_fn = 0

    per_image_results = []

    for image_id in dataset_test.image_ids:
        # Load test image.
        image = dataset_test.load_image(image_id)

        # Ground-truth masks: [height, width, number_of_instances]
        gt_masks, gt_class_ids = dataset_test.load_mask(image_id)

        # Retain only shrub ground-truth instances.
        gt_indices = np.where(gt_class_ids == shrub_class_id)[0]
        gt_shrub_masks = gt_masks[:, :, gt_indices]

        # Convert multiple instance masks into one binary semantic mask.
        if gt_shrub_masks.shape[-1] > 0:
            gt_binary = np.any(gt_shrub_masks, axis=-1)
        else:
            gt_binary = np.zeros(image.shape[:2], dtype=bool)

        # Run inference.
        result = model.detect([image], verbose=0)[0]

        # Retain shrub predictions satisfying the confidence threshold.
        pred_indices = np.where(
            (result["class_ids"] == shrub_class_id) &
            (result["scores"] >= confidence_threshold)
        )[0]

        pred_shrub_masks = result["masks"][:, :, pred_indices]

        if pred_shrub_masks.shape[-1] > 0:
            pred_binary = np.any(pred_shrub_masks, axis=-1)
        else:
            pred_binary = np.zeros(image.shape[:2], dtype=bool)

        # Pixel-level confusion counts.
        tp = np.logical_and(pred_binary, gt_binary).sum()
        fp = np.logical_and(pred_binary, np.logical_not(gt_binary)).sum()
        fn = np.logical_and(np.logical_not(pred_binary), gt_binary).sum()

        total_tp += int(tp)
        total_fp += int(fp)
        total_fn += int(fn)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        per_image_results.append({
            "image_id": image_id,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp_pixels": int(tp),
            "fp_pixels": int(fp),
            "fn_pixels": int(fn),
        })

    # Micro-average: calculate metrics from totals over all images.
    precision = (
        total_tp / (total_tp + total_fp)
        if (total_tp + total_fp) > 0
        else 0.0
    )

    recall = (
        total_tp / (total_tp + total_fn)
        if (total_tp + total_fn) > 0
        else 0.0
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    global_results = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp_pixels": total_tp,
        "fp_pixels": total_fp,
        "fn_pixels": total_fn,
    }

    return global_results, per_image_results


def evaluate_instance_metrics(
    model,
    dataset_test,
    shrub_class_id=1,
    confidence_threshold=0.5,
    iou_threshold=0.5
):
    """
    Computes micro-averaged instance Precision, Recall, and F1
    over the complete test dataset.
    """

    total_tp = 0
    total_fp = 0
    total_fn = 0

    per_image_results = []

    for image_id in dataset_test.image_ids:
        image = dataset_test.load_image(image_id)

        gt_masks, gt_class_ids = dataset_test.load_mask(image_id)

        # Keep shrub annotations only.
        gt_indices = np.where(gt_class_ids == shrub_class_id)[0]
        gt_masks = gt_masks[:, :, gt_indices]
        gt_class_ids = gt_class_ids[gt_indices]

        # Generate boxes from ground-truth masks.
        gt_boxes = utils.extract_bboxes(gt_masks)

        # Run prediction.
        result = model.detect([image], verbose=0)[0]

        # Keep shrub predictions above confidence threshold.
        pred_indices = np.where(
            (result["class_ids"] == shrub_class_id) &
            (result["scores"] >= confidence_threshold)
        )[0]

        pred_boxes = result["rois"][pred_indices]
        pred_class_ids = result["class_ids"][pred_indices]
        pred_scores = result["scores"][pred_indices]
        pred_masks = result["masks"][:, :, pred_indices]

        # One-to-one matching based on mask IoU.
        gt_match, pred_match, overlaps = utils.compute_matches(
            gt_boxes=gt_boxes,
            gt_class_ids=gt_class_ids,
            gt_masks=gt_masks,
            pred_boxes=pred_boxes,
            pred_class_ids=pred_class_ids,
            pred_scores=pred_scores,
            pred_masks=pred_masks,
            iou_threshold=iou_threshold
        )

        tp = int(np.sum(pred_match > -1))
        fp = int(np.sum(pred_match == -1))
        fn = int(np.sum(gt_match == -1))

        total_tp += tp
        total_fp += fp
        total_fn += fn

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        per_image_results.append({
            "image_id": image_id,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
        })

    precision = (
        total_tp / (total_tp + total_fp)
        if (total_tp + total_fp) > 0
        else 0.0
    )

    recall = (
        total_tp / (total_tp + total_fn)
        if (total_tp + total_fn) > 0
        else 0.0
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    global_results = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positive_predictions": total_tp,
        "false_positive_predictions": total_fp,
        "false_negative_ground_truths": total_fn,
        "confidence_threshold": confidence_threshold,
        "iou_threshold": iou_threshold,
    }

    return global_results, per_image_results


def evaluate_instance_metrics_s_iou(
        model,
        dataset_test,
        shrub_class_id=1,
        confidence_threshold=0.5,
        siou_threshold=0.5,
        matching_iou_threshold=0.0
):
    """Evaluate instance-level shrub segmentation using S-IoU.

    To reproduce the evaluation approach described by Khaldi et al. (2024):

    - TP and FP are counted from prediction-directed S-IoU.
    - FN is counted from ground-truth-directed S-IoU.
    - Recall is calculated as TP / (TP + FN).
    """
    total_tp = 0
    total_fp = 0
    total_fn = 0

    total_predictions = 0
    total_ground_truths = 0

    for image_id in dataset_test.image_ids:
        image = dataset_test.load_image(image_id)

        gt_masks, gt_class_ids = dataset_test.load_mask(image_id)

        # Keep shrub ground-truth instances only
        gt_indices = np.where(
            gt_class_ids == shrub_class_id
        )[0]

        gt_masks = gt_masks[:, :, gt_indices]
        gt_class_ids = gt_class_ids[gt_indices]

        # Generate boxes from the retained GT masks.
        gt_boxes = utils.extract_bboxes(gt_masks)

        # Run model inference
        result = model.detect([image], verbose=0)[0]

        # Keep shrub predictions above the confidence threshold
        pred_indices = np.where(
            (result["class_ids"] == shrub_class_id)
            & (result["scores"] >= confidence_threshold)
        )[0]

        pred_boxes = result["rois"][pred_indices]
        pred_class_ids = result["class_ids"][pred_indices]
        pred_scores = result["scores"][pred_indices]
        pred_masks = result["masks"][:, :, pred_indices]

        image_results = compute_s_iou_matches(
            gt_boxes=gt_boxes,
            gt_class_ids=gt_class_ids,
            gt_masks=gt_masks,
            pred_boxes=pred_boxes,
            pred_class_ids=pred_class_ids,
            pred_scores=pred_scores,
            pred_masks=pred_masks,
            s_iou_threshold=siou_threshold,
            # Predictions were already filtered above
            prediction_score_threshold=0.0
        )

        image_tp = np.count_nonzero(image_results["pred_is_tp"])
        image_fp = np.count_nonzero(image_results["pred_is_fp"])
        image_fn = np.count_nonzero(image_results["gt_is_fn"])

        total_tp += image_tp
        total_fp += image_fp
        total_fn += image_fn

        total_predictions += pred_masks.shape[-1]
        total_ground_truths += gt_masks.shape[-1]

    # Prediction-directed precision
    precision_denominator = total_tp + total_fp

    precision = (
        total_tp / precision_denominator
        if precision_denominator > 0
        else 0.0
    )

    # Paper-compatible recall
    recall_denominator = total_tp + total_fn

    recall = (
        total_tp / recall_denominator
        if recall_denominator > 0
        else 0.0
    )

    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    # Each retained prediction must be classified as either TP or FP.
    if total_tp + total_fp != total_predictions:
        raise RuntimeError(
            "Prediction count mismatch: "
            f"TP + FP = {total_tp + total_fp}, "
            f"but {total_predictions} predictions were evaluated."
        )

    # FN cannot exceed the number of ground-truth instances.
    if total_fn > total_ground_truths:
        raise RuntimeError(
            "False-negative count exceeds the number of "
            "ground-truth instances."
        )
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positive_predictions": total_tp,
        "false_positive_predictions": total_fp,
        "false_negative_ground_truths": total_fn,
        "num_predictions": total_predictions,
        "num_ground_truths": total_ground_truths,
        "confidence_threshold": confidence_threshold,
        "siou_threshold": siou_threshold
    }


def compute_iou_range(
    model,
    dataset,
    out_csv_file,
    iou_threshold = 0.5,
    shrub_class_id = 1,
    verbose = True
):
    """Use csv module and write one row per IoU threshold"""
    fieldnames = [
        "threshold",
        "confidence_threshold",
        "precision",
        "recall",
        "f1",
    ]

    with open(out_csv_file, mode="w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for confidence_score_threshold in range(0, 105, 5):
            cst = confidence_score_threshold / 100

            results, _ = evaluate_instance_metrics(
                model,
                dataset,
                shrub_class_id=shrub_class_id,
                confidence_threshold=cst,
                iou_threshold=iou_threshold,
            )

            row = {
                "threshold": iou_threshold,
                "confidence_threshold": cst,
                "precision": results["precision"],
                "recall": results["recall"],
                "f1": results["f1"],
            }

            writer.writerow(row)

            if verbose:
                print(
                    f"IoU={iou_threshold:.2f}: "
                    f"Confidence score threshold={cst}",
                    f"P={results['precision']:.4f}, "
                    f"R={results['recall']:.4f}, "
                    f"F1={results['f1']:.4f}"
                )

    print(f"Metrics saved to: {out_csv_file}")


def compute_siou_range(
    model,
    dataset,
    out_csv_file,
    siou_threshold = 0.5,
    shrub_class_id = 1,
    verbose = True
):
    """Use csv module and write one row per S-IoU threshold"""
    fieldnames = [
        "threshold",
        "confidence_threshold",
        "precision",
        "recall",
        "f1",
    ]

    with open(out_csv_file, mode="w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for confidence_score_threshold in range(0, 105, 5):
            cst = confidence_score_threshold / 100

            results =  evaluate_instance_metrics_s_iou(
                model=model,
                dataset_test=dataset,
                shrub_class_id=shrub_class_id,
                confidence_threshold=cst,
                siou_threshold=siou_threshold
            )

            row = {
                "threshold": siou_threshold,
                "confidence_threshold": cst,
                "precision": results["precision"],
                "recall": results["recall"],
                "f1": results["f1"],
            }

            writer.writerow(row)

            if verbose:
                print(
                    f"S-IoU={siou_threshold:.2f}: "
                    f"Confidence score threshold={cst}",
                    f"P={results['precision']:.4f}, "
                    f"R={results['recall']:.4f}, "
                    f"F1={results['f1']:.4f}"
                )

    print(f"Metrics saved to: {out_csv_file}")


def load_metric_data(csv_path):
    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    confidence_score_thresholds = []
    f1_scores = []

    with open(csv_path, mode="r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)

        if reader.fieldnames is None:
            raise ValueError(f"No header found in {csv_path}")


        threshold_column = "confidence_threshold"

        if "f1" not in reader.fieldnames:
            raise ValueError(
                f"No 'f1' column found in {csv_path}. "
                f"Available columns: {reader.fieldnames}"
            )

        for row_number, row in enumerate(reader, start=2):
            try:
                confidence_threshold = float(row[threshold_column])
                f1 = float(row["f1"])
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Invalid numeric value in {csv_path}, "
                    f"row {row_number}"
                ) from error

            confidence_score_thresholds.append(confidence_threshold)
            f1_scores.append(f1)

    if not confidence_score_thresholds:
        raise ValueError(f"No data rows found in {csv_path}")

    # Convert values from 0–1 to percentages when needed.
    if max(confidence_score_thresholds) <= 1:
        confidence_score_thresholds = [
            value * 100
            for value in confidence_score_thresholds
        ]

    if max(f1_scores) <= 1:
        f1_scores = [value * 100 for value in f1_scores]

    # Sort both lists by threshold.
    sorted_pairs = sorted(zip(confidence_score_thresholds, f1_scores))
    thresholds, f1_scores = zip(*sorted_pairs)

    return list(thresholds), list(f1_scores)


def plot_performance(csv_files):
    
    for eval_metric_threshold, metric_files in csv_files.items():
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

        confidence_percent = int(eval_metric_threshold * 100)

        ax.set_title(
            f"Evaluation Metric threshold = {confidence_percent}%"
        )
        ax.set_xlabel("Evaluation metric thresholds")
        ax.set_ylabel("F1-score values")

        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.set_xticks(range(0, 101, 20))
        ax.set_yticks(range(0, 101, 20))

        ax.grid(alpha=0.25)
        ax.legend(title="Evaluation metrics")

        fig.tight_layout()

        output_file = (
            f"evaluation_metrics_confidence_{confidence_percent}.png"
        )

        fig.savefig(
            output_file,
            dpi=300,
            bbox_inches="tight",
        )

        plt.show()
        plt.close(fig)

        print(f"Figure saved to: {output_file}")