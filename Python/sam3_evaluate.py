"""
Evaluate a SAM 3 checkpoint (zero-shot or domain-adapted) on COCO-format shrub datasets.

Mirrors `detectron2_evaluate.py`: it scores whole, untiled images with a text
prompt and shares the same pixel/instance-IoU/S-IoU metrics, confidence sweep,
and F1-vs-confidence plots (from `shrubs_common`) so the two models are
comparable on the same JuniperMapper splits. Ground-truth polygons are decoded
with pycocotools directly, with no Detectron2 dependency, so this runs in the
lighter `sam3_shrubs` environment.

Example usage:

python sam3_evaluate.py \
  --dataset shrubs_test_pi \
    ./JuniperMapper/Photo_Interpretation_Data/Test/Annotations/Test.json \
    ./JuniperMapper/Photo_Interpretation_Data/Test/Images \
  --dataset shrubs_test_fw \
    ./JuniperMapper/Field_Work_Data/External_Val_Data/Annotations/Val.json \
    ./JuniperMapper/Field_Work_Data/External_Val_Data/Images \
  --prompt shrub --output-dir results/evaluation/model_sam3 \
  --checkpoint models/sam3/sam3.pt
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils

sys.path.append(str(Path(Path(__file__).absolute().parent.parent, 'bin')))
from shrubs_common import METRIC_DEFINITIONS, Prediction, PredictionCache, run_predictor
from shrubs_common.plotting import run_evaluation_report
from shrubs_sam3 import TextPromptPredictor, build_processor, sam3_autocast


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    json_file: Path
    image_root: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a SAM 3 checkpoint on COCO-format shrub datasets."
    )
    parser.add_argument(
        "--checkpoint", type=Path, default=None,
        help="Local release or fine-tuned checkpoint. Omit for the default zero-shot download.",
    )
    parser.add_argument("--prompt", default="shrub")
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
    parser.add_argument("--class-id", type=int, default=0)
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
    parser.add_argument("--show-plots", action="store_true", help="Also open each saved F1 figure in a window.")
    return parser.parse_args()


def _instances_from_annotation(annotation: Mapping[str, Any]) -> list[Any]:
    """JuniperMapper's PI split groups every shrub polygon of an image into one
    annotation; its FW split already stores one standard COCO RLE per shrub.
    Return one polygon or RLE per shrub instance either way."""
    segmentation = annotation.get("segmentation")
    if isinstance(segmentation, list):
        return [
            polygon for polygon in segmentation
            if isinstance(polygon, list) and len(polygon) >= 6 and len(polygon) % 2 == 0
        ]
    if isinstance(segmentation, dict):
        return [segmentation]
    raise TypeError(f"Expected a polygon-list or RLE segmentation, got {type(segmentation).__name__}.")


def _instance_rle(instance: Any, height: int, width: int) -> dict[str, Any]:
    if isinstance(instance, list):
        return mask_utils.merge(mask_utils.frPyObjects([instance], height, width))
    # Uncompressed RLE (list counts) must be compressed before decode/toBbox.
    if isinstance(instance.get("counts"), list):
        return mask_utils.frPyObjects(instance, height, width)
    return instance


def _instance_mask(instance: Any, height: int, width: int) -> np.ndarray:
    return mask_utils.decode(_instance_rle(instance, height, width)).astype(bool)


def _instance_bbox_xyxy(instance: Any, height: int, width: int) -> list[float]:
    x, y, w, h = mask_utils.toBbox(_instance_rle(instance, height, width))
    return [float(x), float(y), float(x + w), float(y + h)]


def load_ground_truth_coco(json_file: Path, image_root: Path) -> tuple[list[dict[str, Any]], PredictionCache]:
    """Load a JuniperMapper-format COCO JSON, exploding it into one shrub instance each."""
    with json_file.open(encoding="utf-8") as source:
        coco = json.load(source)

    annotations_by_image: dict[int, list[Any]] = {}
    for annotation in coco.get("annotations", []):
        if annotation.get("iscrowd", 0):
            continue
        annotations_by_image.setdefault(annotation["image_id"], []).extend(
            _instances_from_annotation(annotation)
        )

    records: list[dict[str, Any]] = []
    ground_truths: dict[int, Prediction] = {}
    for image in coco["images"]:
        image_id, height, width = image["id"], int(image["height"]), int(image["width"])
        instances = annotations_by_image.get(image_id, [])
        masks = (
            np.stack([_instance_mask(instance, height, width) for instance in instances], axis=-1)
            if instances else np.zeros((height, width, 0), dtype=bool)
        )
        boxes = np.array(
            [_instance_bbox_xyxy(instance, height, width) for instance in instances], dtype=np.float32
        ).reshape(-1, 4)
        ground_truths[image_id] = {
            "boxes": boxes, "class_ids": np.zeros(len(instances), dtype=np.int64), "masks": masks,
        }
        records.append({"image_id": image_id, "file_name": str(image_root / image["file_name"])})
    return records, ground_truths


def _read_rgb(record: Mapping[str, Any]) -> np.ndarray:
    return np.array(Image.open(record["file_name"]).convert("RGB"))


def evaluate_dataset(predictor: TextPromptPredictor, spec: DatasetSpec, args: argparse.Namespace) -> None:
    records, ground_truths = load_ground_truth_coco(spec.json_file, spec.image_root)
    print(f"\nDataset: {spec.name} ({len(records)} images)")
    predictions = run_predictor(predictor, records, _read_rgb)
    run_evaluation_report(ground_truths, predictions, args.output_dir / spec.name, args)


def main() -> None:
    args = parse_args()
    if args.checkpoint is not None and not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    specs = [
        DatasetSpec(name=name, json_file=Path(json_file), image_root=Path(image_root))
        for name, json_file, image_root in args.dataset
    ]

    processor, device = build_processor(confidence=0.0, checkpoint=args.checkpoint)
    print(f"Using {device.upper()} for inference.")
    predictor = TextPromptPredictor(processor, args.prompt)

    with sam3_autocast(device):
        for spec in specs:
            evaluate_dataset(predictor, spec, args)


if __name__ == "__main__":
    main()
