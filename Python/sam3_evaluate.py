"""
Evaluate a SAM 3 checkpoint (zero-shot or domain-adapted)

Use COCO-format shrub datasets.

Mirrors `detectron2_evaluate.py`: it scores whole, untiled images with a text
prompt and shares the same pixel/instance-IoU/S-IoU metrics, confidence sweep,
and F1-vs-confidence plots (from `shrubs_common`) so the two models are
comparable on the same JuniperMapper splits. Ground-truth polygons are decoded
with pycocotools directly, with no Detectron2 dependency, so this runs in the
lighter `sam3_shrubs` environment.

Run it with `python Python/sam3_evaluate.py` inside the `sam3_shrubs`
environment. The checkpoint, prompt and output folder are set at the top
of this file; the test splits and evaluation settings are shared with
`detectron2_evaluate.py` through `shrubs_common.evaluation`.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from PIL import Image

sys.path.append(str(Path(Path(__file__).absolute().parent.parent, 'bin')))

# Catch a wrong environment
try:
    from pycocotools import mask as mask_utils
    from shrubs_sam3 import TextPromptPredictor, build_processor, sam3_autocast
except ImportError as e:
    raise ImportError(
        "SAM 3 dependencies are not installed in the active Python environment. "
        "Activate the 'sam3_shrubs' environment (WSL2) before running this script."
    ) from e

from shrubs_common import (
    TEST_DATASETS, EvaluationDataset, EvaluationSettings, Prediction, PredictionCache,
    check_dataset_inputs, run_predictor,
)
from shrubs_common.plotting import run_evaluation_report

# Store the project base directory
ROOT = Path(__file__).resolve().parent.parent

# Domain-adapted checkpoint written by sam3_domain_adapt.py.
# Set to None for the zero-shot release model (downloaded from Hugging Face).
CHECKPOINT = ROOT / "data" / "logs" / "sam3" / "domain_adapt" / "checkpoints" / "model_best.pt"
PROMPT = "shrub"

# Keep zero-shot and domain-adapted results apart so neither overwrites the other
OUTPUT_DIR = ROOT / "results" / "evaluation" / (
    "model_sam3" if CHECKPOINT is not None else "model_sam3_zero_shot"
)

# Shared with detectron2_evaluate.py; override a field here only for a one-off run
SETTINGS = EvaluationSettings()


def check_inputs() -> None:
    """Fail early if the checkpoint or any test split is missing."""
    check_dataset_inputs(TEST_DATASETS)

    if CHECKPOINT is not None and not CHECKPOINT.is_file():
        raise FileNotFoundError(
            f"SAM 3 checkpoint not found at '{CHECKPOINT}'. "
            "Run 'sam3_domain_adapt.py' first, copy the trained 'model_best.pt' "
            "into 'data/logs/sam3/domain_adapt/checkpoints', or set CHECKPOINT = None "
            "to evaluate the zero-shot model."
        )


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


def evaluate_dataset(predictor: TextPromptPredictor, spec: EvaluationDataset) -> None:
    records, ground_truths = load_ground_truth_coco(spec.json_file, spec.image_root)
    print(f"\nDataset: {spec.name} ({len(records)} images)")
    predictions = run_predictor(predictor, records, _read_rgb)
    run_evaluation_report(ground_truths, predictions, OUTPUT_DIR / spec.name, SETTINGS)


def main() -> None:
    check_inputs()

    processor, device = build_processor(confidence=0.0, checkpoint=CHECKPOINT)
    print(f"Using {device.upper()} for inference.")
    predictor = TextPromptPredictor(processor, PROMPT)

    with sam3_autocast(device):
        for spec in TEST_DATASETS:
            evaluate_dataset(predictor, spec)


if __name__ == "__main__":
    main()
