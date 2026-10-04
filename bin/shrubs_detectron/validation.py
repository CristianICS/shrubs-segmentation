"""
Validate the original JSON against registered records

They do not expose the same organization if the original JSON has
one annotation per image containing many shrub polygons.

For example, if validation results showed:

- Original annotations: 67
- Registered shrub instances: 660

an evaluator reading the original JSON directly could interpret the
dataset as 67 objects instead of 660 shrubs.

Check this at three levels:

A. Compare global counts
B. Compare counts per image
C. Compare the decoded masks
"""
import json
import numpy as np
from typing import Any
from pathlib import Path
from collections import defaultdict

from detectron2.data.detection_utils import annotations_to_instances

from .utils import (
    DatasetSpec,
    prepare_evaluation_json,
    segmentation_area
)


def inspect_records(
    dataset_name: str,
    records: list[dict],
    *,
    examples: int = 5,
) -> None:
    """Print a summary without calling DatasetCatalog.get() again."""
    num_images = len(records)

    num_instances = sum(
        len(record.get("annotations", []))
        for record in records
    )

    instance_counts = [
        len(record.get("annotations", []))
        for record in records
    ]

    maximum_instances = max(
        instance_counts,
        default=0,
    )

    print(f"{dataset_name}:")
    print(f"  Images: {num_images}")
    print(f"  Instances: {num_instances}")
    print(
        "  Maximum instances in one image: "
        f"{maximum_instances}"
    )

    for record in records[:examples]:
        print(
            f"  {Path(record['file_name']).name}: "
            f"{len(record.get('annotations', []))} instances"
        )


def get_expected_instances_by_image(
    coco: dict[str, Any],
) -> tuple[dict[int, int], int, int]:
    """
    Count annotations that produce valid Detectron2 instances.

    Returns:
        expected_by_image
        total_polygon_components
        skipped_empty_polygons
    """
    image_dimensions = {
        int(image["id"]): (
            int(image["width"]),
            int(image["height"]),
        )
        for image in coco["images"]
    }

    expected_by_image: dict[int, int] = defaultdict(int)
    total_polygon_components = 0
    skipped_invalid_instances = 0

    for annotation in coco.get("annotations", []):
        image_id = int(annotation["image_id"])
        segmentation = annotation.get("segmentation")

        if isinstance(segmentation, list):
            width, height = image_dimensions[image_id]

            for polygon in segmentation:
                total_polygon_components += 1

                if (
                    not isinstance(polygon, list)
                    or len(polygon) < 6
                    or len(polygon) % 2 != 0
                ):
                    skipped_invalid_instances += 1
                    continue

                area = segmentation_area(
                    segmentation=polygon,
                    width=width,
                    height=height,
                )

                if area <= 0:
                    skipped_invalid_instances += 1
                    continue

                expected_by_image[image_id] += 1

        elif isinstance(segmentation, dict):
            # Each RLE annotation already represents one instance.
            width, height = image_dimensions[image_id]

            area = segmentation_area(
                segmentation=segmentation,
                width=width,
                height=height,
            )

            if area <= 0:
                skipped_invalid_instances += 1
                continue

            expected_by_image[image_id] += 1

        else:
            raise TypeError(
                "Expected segmentation to be a polygon list or "
                f"an RLE dictionary, but received "
                f"{type(segmentation).__name__} for image ID "
                f"{image_id}."
            )

    return (
        dict(expected_by_image),
        total_polygon_components,
        skipped_invalid_instances,
    )


def validate_dataset(
    spec: DatasetSpec,
    records: list[dict],
) -> None:
    """
    Validate the original JSON against already materialized
    Detectron2 records.

    This function does not create or modify a normalized JSON.
    """
    print(f"\nValidating dataset: {spec.name}")
    print(f"Original JSON: {spec.original_json}")

    inspect_records(
        dataset_name=spec.name,
        records=records,
    )

    compare_original_and_registered(
        dataset_name=spec.name,
        json_file=spec.original_json,
        records=records,
    )

    compare_instances_per_image(
        dataset_name=spec.name,
        json_file=spec.original_json,
        records=records,
    )

    validate_registered_masks(
        dataset_name=spec.name,
        records=records,
    )

    print(f"Dataset validation passed: {spec.name}")


def normalize_dataset(
    spec: DatasetSpec,
    normalized_json_dir: Path,
    records: list[dict],
) -> Path:
    """
    Export registered records to normalized COCO JSON and
    validate the resulting file.
    """
    normalized_json = (
        normalized_json_dir
        / spec.normalized_json_name
    )

    print(f"\nNormalizing dataset: {spec.name}")
    print(f"Normalized JSON: {normalized_json}")

    prepare_evaluation_json(
        dataset_name=spec.name,
        output_json=normalized_json,
    )

    validate_normalized_json(
        dataset_name=spec.name,
        normalized_json=normalized_json,
        records=records,
    )

    return normalized_json


def compare_original_and_registered(
    dataset_name: str,
    json_file: Path,
    records: list[dict],
) -> None:
    with json_file.open(encoding="utf-8") as stream:
        coco = json.load(stream)

    (
        expected_by_image,
        total_polygon_components,
        skipped_empty_instances
    ) = get_expected_instances_by_image(coco)

    original_annotation_count = len(coco.get("annotations", []))

    original_rle_count = sum(
        isinstance(annotation.get("segmentation"), dict)
        for annotation in coco.get("annotations", [])
    )

    expected_valid_instances = sum(
        expected_by_image.values()
    )

    registered_instance_count = sum(
        len(record.get("annotations", []))
        for record in records
    )

    print(f"Dataset: {dataset_name}")
    print(
        f"Original COCO annotations: "
        f"{original_annotation_count}"
    )
    print(
        f"Polygon components in original JSON: "
        f"{total_polygon_components}"
    )
    print(
        f"RLE instances in original JSON: "
        f"{original_rle_count}"
    )
    print(
        f"Skipped invalid or empty instances: "
        f"{skipped_empty_instances}"
    )
    print(
        f"Expected valid instances: "
        f"{expected_valid_instances}"
    )
    print(
        f"Registered Detectron2 instances: "
        f"{registered_instance_count}"
    )

    if registered_instance_count != expected_valid_instances:
        raise AssertionError(
            "The registered instance count does not match "
            "the expected number of valid polygon and RLE "
            f"instances: {registered_instance_count} != "
            f"{expected_valid_instances}"
        )

    print("Global organization check passed.")


def compare_instances_per_image(
    dataset_name: str,
    json_file: Path,
    records: list[dict],
) -> None:
    """
    A global comparison could hide compensating errors,
    so also compare every image    
    """
    with json_file.open(encoding="utf-8") as stream:
        coco = json.load(stream)

    expected_by_image, _, _ = get_expected_instances_by_image(coco)

    mismatches = []

    for record in records:
        image_id = int(record["image_id"])
        expected = expected_by_image.get(image_id, 0)
        observed = len(record.get("annotations", []))

        if expected != observed:
            mismatches.append(
                {
                    "image_id": image_id,
                    "file_name": record["file_name"],
                    "expected": expected,
                    "observed": observed,
                }
            )

    if mismatches:
        for mismatch in mismatches[:20]:
            print(mismatch)

        raise AssertionError(
            f"Found {len(mismatches)} images with "
            "mismatched instance counts."
        )

    print(
        f"All {len(records)} images passed the "
        "per-image count check."
    )


def validate_registered_masks(
    dataset_name: str,
    records: list[dict]
) -> None:
    """
    Counts alone are insufficient. Check that splitting
    does not change the actual pixel masks.

    Function identifies the exact image and annotation.
    
    Note: An atol=1.0 tolerance is reasonable because polygon
    rasterization can produce one-pixel boundary differences.
    """
    total_instances = 0
    empty_masks = []
    box_mismatches = []

    for record in records:
        height = int(record["height"])
        width = int(record["width"])
        annotations = record.get("annotations", [])

        instances = annotations_to_instances(
            annotations,
            (height, width),
            mask_format="bitmask",
        )

        if len(instances) != len(annotations):
            raise AssertionError(
                "Instance conversion changed the annotation count for "
                f"{record['file_name']}: "
                f"{len(instances)} != {len(annotations)}"
            )

        masks = (
            instances.gt_masks.tensor
            .cpu()
            .numpy()
            .astype(bool)
        )

        mask_boxes = (
            instances.gt_masks
            .get_bounding_boxes()
            .tensor
            .cpu()
            .numpy()
        )

        annotation_boxes = (
            instances.gt_boxes.tensor
            .cpu()
            .numpy()
        )

        for annotation_index, (
            annotation,
            mask,
            mask_box,
            annotation_box,
        ) in enumerate(
            zip(
                annotations,
                masks,
                mask_boxes,
                annotation_boxes,
            )
        ):
            total_instances += 1

            if not np.any(mask):
                empty_masks.append(
                    {
                        "image_id": record["image_id"],
                        "file_name": record["file_name"],
                        "annotation_index": annotation_index,
                        "annotation_id": annotation.get("id"),
                        "bbox": annotation.get("bbox"),
                        "segmentation": annotation.get(
                            "segmentation"
                        ),
                        "source_annotation_id": annotation.get(
                            "source_annotation_id"
                        ),
                        "source_polygon_index": annotation.get(
                            "source_polygon_index"
                        )
                    }
                )
                continue

            if not np.allclose(
                mask_box,
                annotation_box,
                atol=1.0,
            ):
                box_mismatches.append(
                    {
                        "image_id": record["image_id"],
                        "file_name": record["file_name"],
                        "annotation_index": annotation_index,
                        "annotation_box": annotation_box.tolist(),
                        "mask_box": mask_box.tolist(),
                    }
                )

    print(f"Decoded instances: {total_instances}")
    print(f"Empty masks: {len(empty_masks)}")
    print(f"Box/mask mismatches: {len(box_mismatches)}")

    if empty_masks:
        print("\nEmpty-mask annotations:")

        for item in empty_masks:
            print(
                f"  Image: {item['file_name']}\n"
                f"  Image ID: {item['image_id']}\n"
                f"  Registered annotation index: "
                f"{item['annotation_index']}\n"
                f"  Source annotation ID: "
                f"{item['source_annotation_id']}\n"
                f"  Source polygon index: "
                f"{item['source_polygon_index']}\n"
                f"  Bounding box: {item['bbox']}\n"
                f"  Segmentation: {item['segmentation']}\n"
            )

        raise AssertionError(
            f"The registered dataset contains "
            f"{len(empty_masks)} empty masks."
        )

    if box_mismatches:
        for item in box_mismatches[:20]:
            print(item)

        raise AssertionError(
            f"Found {len(box_mismatches)} box/mask mismatches."
        )


def validate_normalized_json(
    dataset_name: str,
    normalized_json: Path,
    records: list[dict],
) -> None:
    """
    Validate the normalized JSON

    Earlier validation functions compare the original JSON with the registered
    records. Add a final independent check that the exported normalized JSON
    contains the same instances.
    """
    with normalized_json.open(
        encoding="utf-8"
    ) as stream:
        coco = json.load(stream)

    registered_images = len(records)
    registered_instances = sum(
        len(record.get("annotations", []))
        for record in records
    )

    normalized_images = len(
        coco.get("images", [])
    )
    normalized_instances = len(
        coco.get("annotations", [])
    )

    if normalized_images != registered_images:
        raise AssertionError(
            "Normalized image count differs from registered "
            f"records for {dataset_name}: "
            f"{normalized_images} != {registered_images}"
        )

    if normalized_instances != registered_instances:
        raise AssertionError(
            "Normalized annotation count differs from "
            f"registered instances for {dataset_name}: "
            f"{normalized_instances} != "
            f"{registered_instances}"
        )

    annotation_ids = [
        annotation["id"]
        for annotation in coco["annotations"]
    ]

    if len(annotation_ids) != len(set(annotation_ids)):
        raise AssertionError(
            f"Duplicate annotation IDs found in {normalized_json}"
        )

    image_ids = {
        image["id"]
        for image in coco["images"]
    }

    unknown_image_ids = {
        annotation["image_id"]
        for annotation in coco["annotations"]
        if annotation["image_id"] not in image_ids
    }

    if unknown_image_ids:
        raise AssertionError(
            "Annotations reference unknown image IDs: "
            f"{sorted(unknown_image_ids)}"
        )

    print(
        f"Normalized JSON passed: {dataset_name}: "
        f"{normalized_images} images and "
        f"{normalized_instances} instances"
    )
