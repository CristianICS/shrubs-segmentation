"""
Evaluation utilities for Mask R-CNN shrub segmentation.

Pixel, one-to-one mask-IoU and directional S-IoU metrics are not implemented
here: they come from `shrubs_common` so Matterport is scored with exactly the
same code as Detectron2 and SAM 3. This module only adapts Mask R-CNN's
datasets and `model.detect` outputs to the shared `(boxes, class_ids, scores,
masks)` format (boxes XYXY, masks HxWxN), and keeps the Matterport-specific
checkpoint selection (`Validation`) used during training.
"""
from __future__ import annotations

import re
import csv
import time
import shutil
from pathlib import Path
from typing import Union

import numpy as np

from mrcnn import utils
from shrubs_common import Prediction, PredictionCache, evaluate_instance_iou, run_predictor

SHRUB_CLASS_ID = 1


def _yxyx_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    """Convert Mask R-CNN's (y1, x1, y2, x2) boxes to the shared XYXY layout."""
    return np.asarray(boxes).reshape(-1, 4)[:, [1, 0, 3, 2]]


def load_ground_truths(dataset) -> dict[int, Prediction]:
    """Return every image's ground truth as a shared Prediction dict (no scores)."""
    ground_truths: dict[int, Prediction] = {}
    for image_id in dataset.image_ids:
        masks, class_ids = dataset.load_mask(image_id)
        masks = np.asarray(masks, dtype=bool)
        ground_truths[int(image_id)] = {
            "boxes": _yxyx_to_xyxy(utils.extract_bboxes(masks)),
            "class_ids": np.asarray(class_ids, dtype=np.int32),
            "masks": masks,
        }
    return ground_truths


def detect(model, image: np.ndarray) -> Prediction:
    """Run `model.detect` on one image and return a shared Prediction dict."""
    result = model.detect([image], verbose=0)[0]
    return {
        "boxes": _yxyx_to_xyxy(result["rois"]),
        "class_ids": np.asarray(result["class_ids"]),
        "scores": np.asarray(result["scores"]),
        "masks": np.asarray(result["masks"], dtype=bool),
    }


def _dataset_records(dataset) -> list[dict]:
    return [
        {"image_id": int(image_id), "file_name": dataset.image_info[image_id]["path"]}
        for image_id in dataset.image_ids
    ]


def predict_dataset(model, dataset, *, verbose: bool = False) -> PredictionCache:
    """Run inference once for every image and return a shared prediction cache."""
    return run_predictor(
        lambda image: detect(model, image),
        _dataset_records(dataset),
        lambda record: dataset.load_image(record["image_id"]),
        verbose=verbose,
    )


def write_dict_rows(
    rows: list[dict],
    output_path: Union[str, Path],
) -> None:
    """Write a list of dictionaries to CSV."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        raise ValueError("Cannot write an empty list of rows.")

    fieldnames = list(rows[0].keys())

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)


class Validation:

    def __init__(
        self,
        dataset_val,
        checkpoints_dir,
        total_epochs_planned
    ):
        """Class to produce the metric to assess training step."""
        self.cache_dataset(dataset_val)
        self.checkpoints_dir = Path(checkpoints_dir)

        # Confirm that all checkpoints were created
        checkpoint_paths = sorted(
            self.checkpoints_dir.glob(
                "mask_rcnn_shrubs_epoch_*.h5"
            ),
            key=self.checkpoint_epoch,
        )
        saved_epochs = {self.checkpoint_epoch(path) for path in checkpoint_paths}
        expected_epochs = set(range(1, total_epochs_planned + 1))

        missing_epochs = sorted(expected_epochs - saved_epochs)

        if missing_epochs:
            raise RuntimeError(
                f"Missing checkpoint epochs: {missing_epochs}"
            )

        self.checkpoint_paths = checkpoint_paths

    def cache_dataset(self, dataset_val) -> None:
        """
        Cache the validation data

        Caching prevents images and masks from being repeatedly read from disk for
        all 40 checkpoints.
        """
        self.records = _dataset_records(dataset_val)
        self.images = {
            record["image_id"]: dataset_val.load_image(record["image_id"])
            for record in self.records
        }
        self.external_ids = {
            record["image_id"]: str(
                dataset_val.image_info[record["image_id"]].get("id", record["image_id"])
            )
            for record in self.records
        }
        self.ground_truths = load_ground_truths(dataset_val)

    @staticmethod
    def checkpoint_epoch(path: Union[str, Path]) -> int:
        match = re.search(r"epoch_(\d+)\.h5$", Path(path).name)

        if match is None:
            raise ValueError(
                f"Could not extract epoch from checkpoint: {path}"
            )

        return int(match.group(1))

    def predict_set(
        self,
        inference_model,
        checkpoint_path: Union[str, Path],
        verbose: bool = False,
    ) -> PredictionCache:
        """Run inference for one checkpoint"""
        inference_model.load_weights(
            str(checkpoint_path),
            by_name=True,
        )

        return run_predictor(
            lambda image: detect(inference_model, image),
            self.records,
            lambda record: self.images[record["image_id"]],
            verbose=verbose,
        )

    def evaluate_mask_ap(
        self,
        predictions: PredictionCache,
        iou_thresholds: np.ndarray = np.arange(0.50, 0.96, 0.05),
    ) -> tuple[dict, list[dict]]:
        """
        Compute mAP mean among all the iou thresholds

        This is a Matterport-style mean per-image mask AP, averaged across
        validation images and IoU thresholds. It is suitable for comparing the
        checkpoints consistently.

        It is not exactly identical to official COCO dataset-level AP, which
        accumulates detections across the complete dataset and uses the COCO
        evaluation implementation.
        """
        per_image_rows = []

        for image_id, ground_truth in self.ground_truths.items():
            prediction = predictions[image_id]
            row = {
                "image_id": image_id,
                "external_id": self.external_ids[image_id],
                "gt_instances": ground_truth["masks"].shape[-1],
                "predicted_instances": prediction["masks"].shape[-1],
            }

            image_aps = []

            for threshold in iou_thresholds:
                # `compute_ap` only uses boxes to drop all-zero padding rows,
                # so their XYXY layout does not affect the result.
                ap, _, _, _ = utils.compute_ap(
                    ground_truth["boxes"],
                    ground_truth["class_ids"],
                    ground_truth["masks"],
                    prediction["boxes"],
                    prediction["class_ids"],
                    prediction["scores"],
                    prediction["masks"],
                    iou_threshold=float(threshold),
                )

                column = (
                    f"AP{int(round(threshold * 100)):02d}"
                )
                row[column] = float(ap)
                image_aps.append(float(ap))

            row["mAP_50_95"] = float(np.mean(image_aps))
            per_image_rows.append(row)

        summary = {
            "mask_AP50": float(
                np.mean([row["AP50"] for row in per_image_rows])
            ),
            "mask_AP75": float(
                np.mean([row["AP75"] for row in per_image_rows])
            ),
            "mask_mAP_50_95": float(
                np.mean([
                    row["mAP_50_95"]
                    for row in per_image_rows
                ])
            ),
            "validation_image_count": len(per_image_rows),
        }

        for threshold in iou_thresholds:
            column = f"AP{int(round(threshold * 100)):02d}"

            summary[f"mask_{column}"] = float(
                np.mean([
                    row[column]
                    for row in per_image_rows
                ])
            )

        return summary, per_image_rows

    def evaluate_checkpoint_ap(
        self,
        inference_model,
        checkpoint_path: Union[str, Path],
        output_directory: Union[str, Path],
    ) -> tuple[dict, PredictionCache]:
        """Evaluate one checkpoint using mask AP only."""
        checkpoint_path = Path(checkpoint_path)
        output_directory = Path(output_directory)

        evaluation_start = time.perf_counter()
        epoch = self.checkpoint_epoch(checkpoint_path)

        # Inference is performed once for this checkpoint.
        predictions = self.predict_set(
            inference_model=inference_model,
            checkpoint_path=checkpoint_path,
        )

        ap_summary, per_image_ap = self.evaluate_mask_ap(
            predictions=predictions,
        )

        write_dict_rows(
            per_image_ap,
            output_directory /
            f"epoch_{epoch:04d}_per_image_mask_ap.csv"
        )

        checkpoint_summary = {
            "epoch": epoch,
            "checkpoint_path": str(checkpoint_path.resolve()),
            **ap_summary,
            "evaluation_time_seconds": (
                time.perf_counter() - evaluation_start
            ),
        }

        return checkpoint_summary, predictions

    def select_confidence_threshold(
        self,
        predictions: PredictionCache,
        confidence_thresholds=None,
        mask_iou_threshold: float = 0.50,
    ) -> tuple[dict, list[dict]]:
        """Score each confidence threshold with the shared one-to-one mask-IoU metric."""
        if confidence_thresholds is None:
            confidence_thresholds = np.arange(0.00, 0.951, 0.05)

        rows = [
            {
                "confidence_threshold": float(confidence_threshold),
                "mask_iou_threshold": float(mask_iou_threshold),
                **evaluate_instance_iou(
                    self.ground_truths,
                    predictions,
                    class_id=SHRUB_CLASS_ID,
                    confidence_threshold=float(confidence_threshold),
                    iou_threshold=float(mask_iou_threshold),
                ),
            }
            for confidence_threshold in confidence_thresholds
        ]

        # Primary criterion: highest F1.
        #
        # Tie-break 1: higher recall.
        # Tie-break 2: higher precision.
        # Tie-break 3: higher confidence threshold.
        ranked_rows = sorted(
            rows,
            key=lambda row: (
                row["f1"],
                row["recall"],
                row["precision"],
                row["confidence_threshold"],
            ),
            reverse=True,
        )

        return ranked_rows[0], rows

    def select_best_checkpoint(self, evaluation_dir: Union[str, Path]):
        """
        Select the checkpoint with the highest validation mask AP50

        Mask AP50 is the criterion shared with the Detectron2 (segm/AP50)
        and SAM 3 (COCO segm AP50) models. The tie-breaking rule is:

        - highest mask AP50;
        - highest mask mAP50:95;
        - earliest epoch.

        Using the earliest epoch as the final tie-break favours the
        less-trained model when performance is effectively identical.
        """
        ranked_checkpoints = sorted(
            self.checkpoint_results,
            key=lambda row: (
                -row["mask_AP50"],
                -row["mask_mAP_50_95"],
                row["epoch"],
            ),
        )

        self.best_checkpoint_data = ranked_checkpoints[0]

        write_dict_rows(
            ranked_checkpoints,
            Path(evaluation_dir) / "checkpoint_ranking.csv",
        )

        selected_source = Path(
            self.best_checkpoint_data["checkpoint_path"]
        )

        self.selected_checkpoint_destination = (
            Path(evaluation_dir).parent /
            "best_validation_map_checkpoint.h5"
        )

        shutil.copy2(
            selected_source,
            self.selected_checkpoint_destination,
        )

        return self.best_checkpoint_data

    def evaluate_all_checkpoints_ap(
        self,
        inference_model,
        checkpoint_evaluation_dir: Union[str, Path],
    ):
        """Evaluate every checkpoint using mask AP only."""
        checkpoint_evaluation_dir = Path(checkpoint_evaluation_dir)
        checkpoint_evaluation_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_results = []

        for index, checkpoint_path in enumerate(
            self.checkpoint_paths,
            start=1,
        ):
            print(
                f"Evaluating checkpoint "
                f"{index}/{len(self.checkpoint_paths)}: "
                f"{checkpoint_path.name}"
            )

            summary, predictions = self.evaluate_checkpoint_ap(
                inference_model=inference_model,
                checkpoint_path=checkpoint_path,
                output_directory=checkpoint_evaluation_dir,
            )

            checkpoint_results.append(summary)

            # Predictions are no longer needed for this checkpoint.
            del predictions

            self.checkpoint_results = checkpoint_results

            write_dict_rows(
                self.checkpoint_results,
                checkpoint_evaluation_dir.parent /
                "checkpoint_validation_summary.csv"
            )

    def evaluate_best_checkpoint_threshold(
        self,
        inference_model,
        output_directory: Union[str, Path],
        confidence_thresholds=None,
        mask_iou_threshold: float = 0.50,
    ):
        """Run the confidence sweep only for the selected checkpoint."""
        if not hasattr(self, "best_checkpoint_data"):
            raise RuntimeError(
                "Select the best checkpoint before evaluating thresholds."
            )

        output_directory = Path(output_directory)
        output_directory.mkdir(parents=True, exist_ok=True)

        checkpoint_path = Path(
            self.best_checkpoint_data["checkpoint_path"]
        )

        # Run inference once for the selected checkpoint.
        predictions = self.predict_set(
            inference_model=inference_model,
            checkpoint_path=checkpoint_path,
            verbose=True,
        )

        best_threshold, threshold_results = (
            self.select_confidence_threshold(
                predictions=predictions,
                confidence_thresholds=confidence_thresholds,
                mask_iou_threshold=mask_iou_threshold,
            )
        )

        write_dict_rows(
            threshold_results,
            output_directory /
            "best_checkpoint_confidence_thresholds.csv",
        )

        self.best_threshold_data = best_threshold

        return best_threshold, threshold_results

    def store_final_run(self, run_metadata):
        run_metadata.status = "evaluation_completed"
        run_metadata.checkpoint_count = len(self.checkpoint_paths)

        run_metadata.best_checkpoint_epoch = int(
            self.best_checkpoint_data["epoch"]
        )

        run_metadata.best_checkpoint_original_path = str(
            Path(
                self.best_checkpoint_data["checkpoint_path"]
            ).resolve()
        )

        run_metadata.best_checkpoint_path = str(
            self.selected_checkpoint_destination.resolve()
        )

        run_metadata.validation_metrics = {
            "checkpoint_selection_metric": "mean_per_image_mask_AP50",
            "mask_mAP_50_95": float(
                self.best_checkpoint_data["mask_mAP_50_95"]
            ),
            "mask_AP50": float(
                self.best_checkpoint_data["mask_AP50"]
            ),
            "mask_AP75": float(
                self.best_checkpoint_data["mask_AP75"]
            ),
            "confidence_selection_metric": (
                "maximum_validation_F1_at_mask_IoU_0.50"
            ),
            "selected_confidence_threshold": float(
                self.best_threshold_data["confidence_threshold"]
            ),
            "precision": float(
                self.best_threshold_data["precision"]
            ),
            "recall": float(
                self.best_threshold_data["recall"]
            ),
            "f1": float(
                self.best_threshold_data["f1"]
            ),
            "true_positives": int(
                self.best_threshold_data["true_positive_predictions"]
            ),
            "false_positives": int(
                self.best_threshold_data["false_positive_predictions"]
            ),
            "false_negatives": int(
                self.best_threshold_data["false_negative_ground_truths"]
            ),
        }

        return run_metadata
