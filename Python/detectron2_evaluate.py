"""
Evaluate a Detectron2 Mask R-CNN model on COCO-format shrub datasets.

Images are resized using Detectron2's configured test transformation.
Predicted masks and boxes are returned in the original image coordinate
system before comparison with the ground-truth annotations.

The script computes:

* micro-averaged pixel precision, recall, and F1;
* mask-IoU instance precision, recall, and F1, with standard one-to-one
  matching or the Khaldi et al. (2024) definition (`--iou-metric-definition`);
* directional S-IoU metrics as defined by Khaldi et al. (2024);
* confidence-threshold sweeps saved as CSV files; and
* F1-versus-confidence plots.

Inference is performed at each image's native resolution. Detectron2 may pad
images internally to satisfy backbone size divisibility, but it does not resize
or geometrically augment them in this script.

The metric primitives (pixel/instance/S-IoU scoring, confidence sweeps, and
plotting) live in `shrubs_common` and are shared with `sam3_evaluate.py`, so
both models are scored identically on the same JuniperMapper splits.


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
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.detection_utils import annotations_to_instances, read_image

sys.path.append(str(Path(Path(__file__).absolute().parent.parent, 'bin')))
from shrubs_common import METRIC_DEFINITIONS, run_predictor
from shrubs_common.plotting import run_evaluation_report
from shrubs_detectron import ScaleMatchedPredictor, build_inference_cfg, inspect_dataset, register_juniper_dataset


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
    parser.add_argument(
        "--iou-metric-definition",
        choices=METRIC_DEFINITIONS,
        default="standard",
        help=(
            "IoU counting: 'standard' (COCO-style one-to-one matching) or 'paper' "
            "(Khaldi et al. 2024 independent best matches, recall = TP / (TP + FN))."
        ),
    )
    parser.add_argument("--minimum-intersection-pixels", type=int, default=1)
    parser.add_argument("--device", choices=("cuda", "cpu"), default=None)
    parser.add_argument("--show-plots", action="store_true", help="Also open each saved F1 figure in a window.")
    return parser.parse_args()


def register_dataset(spec: DatasetSpec) -> None:
    if spec.name in DatasetCatalog.list():
        DatasetCatalog.remove(spec.name)
        MetadataCatalog.remove(spec.name)

    register_juniper_dataset(
        dataset_name=spec.name,
        json_file=spec.json_file,
        image_root=spec.image_root
    )


def load_ground_truth(record: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Return one COCO record's boxes, contiguous class IDs, and masks as a Prediction dict."""
    height = int(record["height"])
    width = int(record["width"])
    annotations = [ann for ann in record.get("annotations", []) if ann.get("iscrowd", 0) == 0]

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
    spec: DatasetSpec,
    args: argparse.Namespace,
) -> None:
    records = DatasetCatalog.get(spec.name)
    ground_truths = {int(record["image_id"]): load_ground_truth(record) for record in records}

    print(f"\nDataset: {spec.name} ({len(records)} images)")
    predictions = run_predictor(predictor, records, lambda record: _read_checked_image(record, predictor))
    run_evaluation_report(ground_truths, predictions, args.output_dir / spec.name, args)


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
