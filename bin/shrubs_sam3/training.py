"""Dataset preparation and partial-freezing support for SAM 3 experiments."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any


def _polygon_bbox(polygon: list[float]) -> list[float]:
    xs = polygon[0::2]
    ys = polygon[1::2]
    xmin, ymin = min(xs), min(ys)
    return [xmin, ymin, max(xs) - xmin, max(ys) - ymin]


def _polygon_area(polygon: list[float]) -> float:
    points = list(zip(polygon[0::2], polygon[1::2]))
    return abs(sum(
        x0 * y1 - x1 * y0
        for (x0, y0), (x1, y1) in zip(points, points[1:] + points[:1])
    )) / 2.0


def prepare_juniper_coco(
    source_json: Path,
    output_json: Path,
    *,
    shots: int | None = None,
    seed: int = 42,
    prompt: str = "shrub",
) -> dict[str, Any]:
    """Convert JuniperMapper's grouped polygons to SAM 3 COCO instances."""
    with source_json.open(encoding="utf-8") as source:
        original = json.load(source)

    instances: list[dict[str, Any]] = []
    for annotation in original.get("annotations", []):
        segmentation = annotation.get("segmentation")
        polygons = segmentation if isinstance(segmentation, list) else []
        for polygon in polygons:
            if not isinstance(polygon, list) or len(polygon) < 6 or len(polygon) % 2:
                continue
            area = _polygon_area(polygon)
            if area <= 0:
                continue
            instances.append({
                "image_id": annotation["image_id"],
                "category_id": 1,
                "segmentation": [polygon],
                "area": area,
                "bbox": _polygon_bbox(polygon),
                "iscrowd": 0,
            })

    if shots is not None:
        if shots < 1:
            raise ValueError("shots must be positive")
        if shots > len(instances):
            raise ValueError(
                f"Requested {shots} shots, but only {len(instances)} instances exist."
            )
        instances = random.Random(seed).sample(instances, shots)

    used_image_ids = {annotation["image_id"] for annotation in instances}
    images = [
        image for image in original.get("images", [])
        if image["id"] in used_image_ids
    ]
    for annotation_id, annotation in enumerate(instances, start=1):
        annotation["id"] = annotation_id

    result = {
        "info": {
            "description": "JuniperMapper shrubs prepared for SAM 3",
            "source": str(source_json),
            "shots": shots,
            "seed": seed,
        },
        "licenses": original.get("licenses", []),
        "images": images,
        "annotations": instances,
        "categories": [{"id": 1, "name": prompt, "supercategory": "vegetation"}],
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as destination:
        json.dump(result, destination, indent=2)
    return result


try:
    from sam3.train.trainer import Trainer as _Sam3Trainer
except ImportError:  # Keep inference and preparation importable without train extras.
    _Sam3Trainer = object


class FrozenBackboneTrainer(_Sam3Trainer):
    """SAM 3 trainer that freezes both pretrained backbones before AdamW."""

    def _setup_components(self) -> None:
        super()._setup_components()
        frozen = 0
        for name, parameter in self.model.named_parameters():
            if name.startswith((
                "backbone.vision_backbone.",
                "backbone.language_backbone.",
            )):
                parameter.requires_grad_(False)
                frozen += parameter.numel()
        if frozen == 0:
            raise RuntimeError(
                "No SAM 3 vision/language backbone parameters matched for freezing."
            )
