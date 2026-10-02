"""
Evaluate a Detectron2 Mask R-CNN model on COCO-format shrub datasets.

Run it with `python Python/detectron2_evaluate.py`. The weights path and model
config are set at the top of this file; the test splits and evaluation settings
are shared with `sam3_evaluate.py` through `shrubs_common.evaluation`.

Images are resized using Detectron2's configured test transformation.
Predicted masks and boxes are returned in the original image coordinate
system before comparison with the ground-truth annotations.

The script computes:

* micro-averaged pixel precision, recall, and F1;
* mask-IoU instance precision, recall, and F1, with standard one-to-one
  matching or the Khaldi et al. (2024) definition (`iou_metric_definition`);
* directional S-IoU metrics as defined by Khaldi et al. (2024);
* confidence-threshold sweeps saved as CSV files; and
* F1-versus-confidence plots.

Inference is performed at each image's native resolution. Detectron2 may pad
images internally to satisfy backbone size divisibility, but it does not resize
or geometrically augment them in this script.

The metric primitives (pixel/instance/S-IoU scoring, confidence sweeps, and
plotting) live in `shrubs_common` and are shared with `sam3_evaluate.py`, so
both models are scored identically on the same JuniperMapper splits.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np

# Catch a wrong environment
try:
    from detectron2.data import DatasetCatalog, MetadataCatalog
    from detectron2.data.detection_utils import annotations_to_instances, read_image
except ImportError as e:
    raise ImportError(
        "Detectron2 is not installed in the active Python environment. "
        "Activate the Detectron2 environment before running this script."
    ) from e

sys.path.append(str(Path(Path(__file__).absolute().parent.parent, 'bin')))
from shrubs_common import (
    TEST_DATASETS, EvaluationDataset, EvaluationSettings, check_dataset_inputs, run_predictor,
)
from shrubs_common.plotting import run_evaluation_report
from shrubs_detectron import ScaleMatchedPredictor, build_inference_cfg, inspect_dataset, register_juniper_dataset

# Store the project base directory
ROOT = Path(__file__).resolve().parent.parent

WEIGHTS = ROOT / "data" / "logs" / "detectron2" / "model_best.pth"
OUTPUT_DIR = ROOT / "results" / "evaluation" / "model_detectron2"

# Same architecture used in detectron2_shrubs_train.py
CONFIG = "COCO-InstanceSegmentation/mask_rcnn_R_101_C4_3x.yaml"
NUM_CLASSES = 1

# Shared with sam3_evaluate.py; override a field here only for a one-off run,
# e.g. EvaluationSettings(iou_metric_definition="standard")
SETTINGS = EvaluationSettings()
DEVICE = None  # None -> CUDA if available, otherwise CPU


def check_inputs() -> None:
    """Fail early if the weights or any test split are missing."""
    check_dataset_inputs(TEST_DATASETS)

    if not WEIGHTS.is_file():
        raise FileNotFoundError(
            f"Detectron2 weights not found at '{WEIGHTS}'. "
            "Run 'detectron2_shrubs_train.py' first, or copy the trained "
            "'model_best.pth' checkpoint into 'data/logs/detectron2'."
        )


def register_dataset(spec: EvaluationDataset) -> None:
    if spec.name in DatasetCatalog.list():
        DatasetCatalog.remove(spec.name)
        MetadataCatalog.remove(spec.name)

    register_juniper_dataset(
        dataset_name=spec.name,
        json_file=spec.json_file,
        image_root=spec.image_root
    )


def load_ground_truth(record: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """
    Return one COCO record's boxes, contiguous class IDs, and masks as
    a Prediction dict.
    """
    height = int(record["height"])
    width = int(record["width"])
    annotations = [
        ann
        for ann in record.get("annotations", [])
        if ann.get("iscrowd", 0) == 0
    ]

    if not annotations:
        return {
            "boxes": np.zeros((0, 4), dtype=np.float32),
            "class_ids": np.zeros((0,), dtype=np.int64),
            "masks": np.zeros((height, width, 0), dtype=bool),
        }

    instances = annotations_to_instances(annotations, (height, width), mask_format="bitmask")
    return {
        "boxes": instances.gt_boxes.tensor.numpy().astype(np.float32, copy=False),
        "class_ids": instances.gt_classes.numpy().astype(np.int64, copy=False),
        "masks": np.moveaxis(instances.gt_masks.tensor.numpy().astype(bool, copy=False), 0, -1),
    }


def _read_checked_image(record: Mapping[str, Any], predictor: ScaleMatchedPredictor) -> np.ndarray:
    image = read_image(record["file_name"], format=predictor.input_format)
    expected = (int(record["height"]), int(record["width"]))
    if image.shape[:2] != expected:
        raise ValueError(
            f"Image dimensions differ from COCO metadata for {record['file_name']}: "
            f"image={image.shape[:2]}, metadata={expected}."
        )
    return image


def evaluate_dataset(
    predictor: ScaleMatchedPredictor,
    spec: EvaluationDataset
) -> None:
    records = DatasetCatalog.get(spec.name)
    
    ground_truths = {
        int(record["image_id"]): load_ground_truth(record)
        for record in records    
    }

    print(f"\nDataset: {spec.name} ({len(records)} images)")
    predictions = run_predictor(
        predictor,
        records,
        lambda record: _read_checked_image(record, predictor)
    )

    run_evaluation_report(
        ground_truths,
        predictions,
        OUTPUT_DIR / spec.name,
        SETTINGS
    )


def main() -> None:
    check_inputs()

    for spec in TEST_DATASETS:
        register_dataset(spec)
        inspect_dataset(spec.name)

    predictor = ScaleMatchedPredictor(
        build_inference_cfg(WEIGHTS, config=CONFIG, num_classes=NUM_CLASSES, device=DEVICE)
    )
    for spec in TEST_DATASETS:
        evaluate_dataset(predictor, spec)


if __name__ == "__main__":
    main()
