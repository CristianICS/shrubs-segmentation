from __future__ import annotations

import copy
from pathlib import Path
from dataclasses import dataclass
from typing import Any

from pycocotools import mask as mask_utils

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_coco_json
from detectron2.data.datasets import convert_to_coco_json
from detectron2.structures import BoxMode


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    original_json: Path
    image_root: Path
    normalized_json_name: str


def normalize_rle(
    segmentation: dict[str, Any],
    width: int,
    height: int,
) -> dict[str, Any]:
    """
    Convert uncompressed COCO RLE to compressed RLE when necessary.

    Compressed RLE has string/bytes counts. Uncompressed RLE has
    a list of integer counts.
    """
    if isinstance(segmentation.get("counts"), list):
        return mask_utils.frPyObjects(
            segmentation,
            height,
            width,
        )

    return segmentation


def prepare_evaluation_json(
    dataset_name: str,
    output_json: Path,
) -> None:
    """Force the custom loader to run before conversion."""
    output_json = output_json.resolve()

    output_json.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # The conversion function obtains the registered dataset itself
    convert_to_coco_json(
        dataset_name=dataset_name,
        output_file=str(output_json),
        allow_cached=False,
    )
    
    # Make COCOEvaluator use the transformed JSON
    metadata = MetadataCatalog.get(dataset_name)

    # Avoid changing original_json_file
    if hasattr(metadata, "json_file"):
        current_json = Path(metadata.json_file).resolve()

        if current_json != output_json:
            raise RuntimeError(
                f"{dataset_name} already has a different evaluator "
                f"JSON configured:\n"
                f"  Existing: {current_json}\n"
                f"  Requested: {output_json}"
            )
    else:
        metadata.set(
            json_file=str(output_json),
        )

    print(
        f"COCOEvaluator for {dataset_name} will use: {output_json}"
    )


def store_new_annotation(
    annotation: dict,
    polygon: list[float] | None = None,
    *,
    polygon_index: int | None = None,
    area: float | None = None,
    bbox: list[float] | None = None,
) -> dict:

    source_annotation_id = annotation.get("id")

    new_annotation = copy.deepcopy(annotation)
    new_annotation["bbox_mode"] = BoxMode.XYWH_ABS
    new_annotation["iscrowd"] = int(
        new_annotation.get("iscrowd", 0)
    )
    new_annotation["source_annotation_id"] = source_annotation_id

    if polygon is not None:
        # One polygon now represents one shrub instance.
        new_annotation["segmentation"] = [polygon]
        new_annotation["source_polygon_index"] = polygon_index

    if bbox is not None:
        new_annotation["bbox"] = bbox

    if area is not None:
        new_annotation["area"] = area

    return new_annotation


def polygon_rle(
    polygon: list[float],
    width: int,
    height: int,
) -> dict[str, Any]:
    rles = mask_utils.frPyObjects(
        [polygon],
        height,
        width,
    )

    return mask_utils.merge(rles)


def segmentation_area(
    segmentation: list[float] | dict[str, Any],
    width: int,
    height: int,
) -> float:
    """
    Return the rasterized area of one instance segmentation.

    The segmentation can be:
    - one flat COCO polygon: [x1, y1, x2, y2, ...]
    - one compressed or uncompressed COCO RLE dictionary
    """
    if isinstance(segmentation, list):
        if (
            len(segmentation) < 6
            or len(segmentation) % 2 != 0
        ):
            return 0.0

        rle = polygon_rle(
            polygon=segmentation,
            width=width,
            height=height,
        )

    elif isinstance(segmentation, dict):
        rle = normalize_rle(
            segmentation=segmentation,
            width=width,
            height=height,
        )

    else:
        raise TypeError(
            "Expected one polygon list or an RLE dictionary, "
            f"but received {type(segmentation).__name__}."
        )

    return float(mask_utils.area(rle))


def segmentation_bbox(
    segmentation: list[float] | dict[str, Any],
    width: int,
    height: int,
) -> list[float]:
    """Return the rasterized XYWH bounding box of one segmentation."""
    if isinstance(segmentation, list):
        rle = polygon_rle(
            polygon=segmentation,
            width=width,
            height=height,
        )

    elif isinstance(segmentation, dict):
        rle = normalize_rle(
            segmentation=segmentation,
            width=width,
            height=height,
        )

    else:
        raise TypeError(
            "Expected one polygon list or an RLE dictionary, "
            f"but received {type(segmentation).__name__}."
        )

    return [
        float(value)
        for value in mask_utils.toBbox(rle).tolist()
    ]


def load_juniper_instances(
    json_file: str | Path,
    image_root: str | Path,
    dataset_name: str,
) -> list[dict]:
    """
    Load JuniperMapper annotations and treat each polygon list as one shrub.

    The original JuniperMapper JSON stores all shrub polygons from one image
    inside one annotation dictionary. This function splits those polygons into
    separate Detectron2 instance annotations.

    This loader assumes that every polygon in a polygon-list segmentation
    represents a different shrub. However, in standard COCO, multiple polygon
    lists can represent disconnected parts of the same instance.
    """
    dataset_dicts = load_coco_json(
        json_file=str(json_file),
        image_root=str(image_root),
        dataset_name=dataset_name,
        extra_annotation_keys=["id"]
    )

    total_original_annotations = 0
    total_instances = 0

    for record in dataset_dicts:
        width = int(record["width"])
        height = int(record["height"])

        split_annotations = []

        for annotation in record.get("annotations", []):
            total_original_annotations += 1
            segmentation = annotation.get("segmentation")

            # Original PI format: each polygon represents a separate shrub.
            if isinstance(segmentation, list):
              
                for polygon_index, polygon in enumerate(segmentation):
                    if not isinstance(polygon, list):
                        raise TypeError(
                            "Each segmentation component must be a polygon list."
                        )

                    if len(polygon) < 6 or len(polygon) % 2 != 0:
                        print(
                            "Skipping invalid polygon in "
                            f"{record['file_name']}: {len(polygon)} coordinates"
                        )
                        continue

                    area = segmentation_area(
                        segmentation=polygon,
                        width=width,
                        height=height,
                    )

                    if area <= 0:
                        print(
                            "Skipping polygon that rasterizes to an empty mask: "
                            f"image={record['file_name']}, "
                            f"polygon_index={polygon_index}"
                        )
                        continue

                    bbox = segmentation_bbox(
                        segmentation=polygon,
                        width=width,
                        height=height,
                    )

                    new_annotation = store_new_annotation(
                        annotation,
                        polygon=polygon,
                        polygon_index=polygon_index,
                        bbox=bbox,
                        area=area
                    )
                    split_annotations.append(new_annotation)
                    total_instances += 1

            # Standard COCO RLE: the annotation already represents one instance.
            elif isinstance(segmentation, dict):
                area = segmentation_area(
                    segmentation=segmentation,
                    width=width,
                    height=height,
                )

                if area <= 0:
                    print(
                        "Skipping RLE annotation that decodes to an "
                        f"empty mask: image={record['file_name']}"
                    )
                    continue

                bbox = segmentation_bbox(
                    segmentation=segmentation,
                    width=width,
                    height=height,
                )

                new_annotation = store_new_annotation(
                    annotation,
                    area=area,
                    bbox=bbox
                )
                
                split_annotations.append(new_annotation)
                total_instances += 1

            else:
                raise TypeError(
                    "Expected segmentation to be a polygon list or an RLE "
                    f"dictionary, but received {type(segmentation).__name__} "
                    f"for image {record['file_name']}."
                )

        record["annotations"] = split_annotations

    print(
        f"{dataset_name}: "
        f"{len(dataset_dicts)} images, "
        f"{total_original_annotations} original annotations, "
        f"{total_instances} shrub instances"
    )

    return dataset_dicts


def register_juniper_dataset(
    dataset_name: str,
    json_file: Path,
    image_root: Path,
) -> None:
    DatasetCatalog.register(
        dataset_name,
        lambda json_file=json_file,
               image_root=image_root,
               dataset_name=dataset_name: load_juniper_instances(
            json_file=json_file,
            image_root=image_root,
            dataset_name=dataset_name,
        ),
    )

    metadata = MetadataCatalog.get(dataset_name)
    metadata.set(
        original_json_file=str(json_file.resolve()),
        image_root=str(image_root.resolve()),
        evaluator_type="coco",
    )


def inspect_dataset(dataset_name: str) -> None:
    """
    Check if the loader exposes each polygon as a separate shrub instance.
    """
    records = DatasetCatalog.get(dataset_name)

    num_images = len(records)
    num_instances = sum(
        len(record.get("annotations", []))
        for record in records
    )

    print(f"{dataset_name}:")
    print(f"  Images: {num_images}")
    print(f"  Instances: {num_instances}")

    for record in records[:5]:
        print(
            f"  {Path(record['file_name']).name}: "
            f"{len(record['annotations'])} instances"
        )
