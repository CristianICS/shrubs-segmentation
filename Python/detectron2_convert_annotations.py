"""
Convert:

- one georeferenced image per file;
- one matching shapefile per image;
- one polygon feature per shrub instance;

into a valid COCO instance-segmentation JSON file.

Example execution:

python shapefiles_to_coco.py ^
  --images-dir D:/dataset/Test/Images ^
  --shapefiles-dir D:/dataset/Test/Shapefiles ^
  --output-json D:/dataset/Test/annotations/Test.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import geopandas as gpd
import numpy as np
import rasterio
from pycocotools import mask as mask_utils
from rasterio.features import rasterize
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon
from shapely.geometry.base import BaseGeometry


DEFAULT_IMAGE_EXTENSIONS = (".tif", ".tiff", ".png", ".jpg", ".jpeg")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert one polygon shapefile per image into a COCO "
            "instance-segmentation annotation file."
        )
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        required=True,
        help="Directory containing the source images.",
    )
    parser.add_argument(
        "--shapefiles-dir",
        type=Path,
        required=True,
        help=(
            "Directory containing one .shp file per image. The shapefile and "
            "image must have the same filename stem."
        ),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        required=True,
        help="Output COCO JSON path.",
    )
    parser.add_argument(
        "--category-name",
        default="shrub",
        help="COCO category name. Default: shrub.",
    )
    parser.add_argument(
        "--category-id",
        type=int,
        default=1,
        help=(
            "COCO category ID written to the JSON. Detectron2 will remap the "
            "single category to contiguous class ID 0. Default: 1."
        ),
    )
    parser.add_argument(
        "--image-extensions",
        nargs="+",
        default=list(DEFAULT_IMAGE_EXTENSIONS),
        help="Image extensions to search for.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search recursively inside the image and shapefile directories.",
    )
    parser.add_argument(
        "--allow-missing-shapefiles",
        action="store_true",
        help=(
            "Include images without a matching shapefile as images containing "
            "no annotated instances."
        ),
    )
    parser.add_argument(
        "--fix-invalid-geometries",
        action="store_true",
        help="Attempt to repair invalid polygons with geometry.buffer(0).",
    )
    parser.add_argument(
        "--minimum-area-pixels",
        type=float,
        default=1.0,
        help="Discard instances smaller than this rasterized pixel area.",
    )
    return parser.parse_args()


def normalize_extensions(extensions: Iterable[str]) -> set[str]:
    normalized = set()
    for extension in extensions:
        extension = extension.lower()
        normalized.add(extension if extension.startswith(".") else f".{extension}")
    return normalized


def find_files(directory: Path, extensions: set[str], recursive: bool) -> list[Path]:
    iterator = directory.rglob("*") if recursive else directory.glob("*")
    return sorted(
        path
        for path in iterator
        if path.is_file() and path.suffix.lower() in extensions
    )


def build_unique_stem_index(paths: Iterable[Path], description: str) -> dict[str, Path]:
    index: dict[str, Path] = {}
    duplicates: dict[str, list[Path]] = {}

    for path in paths:
        key = path.stem.casefold()
        if key in index:
            duplicates.setdefault(key, [index[key]]).append(path)
        else:
            index[key] = path

    if duplicates:
        details = "\n".join(
            f"  {stem}: {', '.join(str(path) for path in duplicate_paths)}"
            for stem, duplicate_paths in duplicates.items()
        )
        raise ValueError(
            f"Duplicate {description} filename stems were found:\n{details}"
        )

    return index


def iter_polygon_parts(geometry: BaseGeometry) -> Iterable[Polygon]:
    if geometry.is_empty:
        return

    if isinstance(geometry, Polygon):
        yield geometry
    elif isinstance(geometry, MultiPolygon):
        yield from geometry.geoms
    elif isinstance(geometry, GeometryCollection):
        for part in geometry.geoms:
            yield from iter_polygon_parts(part)


def encode_binary_mask(binary_mask: np.ndarray) -> dict:
    rle = mask_utils.encode(np.asfortranarray(binary_mask.astype(np.uint8)))
    # pycocotools returns bytes, which are not JSON serializable.
    rle["counts"] = rle["counts"].decode("ascii")
    return rle


def mask_bbox(binary_mask: np.ndarray) -> list[float]:
    rows, cols = np.nonzero(binary_mask)
    if rows.size == 0:
        return [0.0, 0.0, 0.0, 0.0]

    x_min = int(cols.min())
    x_max = int(cols.max())
    y_min = int(rows.min())
    y_max = int(rows.max())
    return [
        float(x_min),
        float(y_min),
        float(x_max - x_min + 1),
        float(y_max - y_min + 1),
    ]


def load_geometries_for_image(
    shapefile_path: Path,
    image_crs,
    *,
    fix_invalid_geometries: bool,
) -> list[BaseGeometry]:
    geodataframe = gpd.read_file(shapefile_path)

    if geodataframe.crs is None:
        raise ValueError(f"Shapefile has no CRS: {shapefile_path}")
    if image_crs is None:
        raise ValueError(
            "The raster image has no CRS, so map-coordinate shapefile polygons "
            f"cannot be aligned with it: {shapefile_path}"
        )

    if geodataframe.crs != image_crs:
        geodataframe = geodataframe.to_crs(image_crs)

    geometries: list[BaseGeometry] = []
    for geometry in geodataframe.geometry:
        if geometry is None or geometry.is_empty:
            continue

        if not geometry.is_valid:
            if not fix_invalid_geometries:
                raise ValueError(
                    "Invalid geometry found in "
                    f"{shapefile_path}. Run with --fix-invalid-geometries "
                    "to attempt a repair."
                )
            geometry = geometry.buffer(0)

        geometries.extend(iter_polygon_parts(geometry))

    return geometries


def convert_dataset(args: argparse.Namespace) -> dict:
    if not args.images_dir.is_dir():
        raise NotADirectoryError(f"Images directory not found: {args.images_dir}")
    if not args.shapefiles_dir.is_dir():
        raise NotADirectoryError(
            f"Shapefiles directory not found: {args.shapefiles_dir}"
        )
    if args.category_id < 1:
        raise ValueError("--category-id must be at least 1.")
    if args.minimum_area_pixels < 0:
        raise ValueError("--minimum-area-pixels cannot be negative.")

    image_extensions = normalize_extensions(args.image_extensions)
    image_paths = find_files(args.images_dir, image_extensions, args.recursive)
    shapefile_paths = find_files(args.shapefiles_dir, {".shp"}, args.recursive)

    if not image_paths:
        raise FileNotFoundError(f"No images found in {args.images_dir}")

    shapefile_index = build_unique_stem_index(shapefile_paths, "shapefile")
    build_unique_stem_index(image_paths, "image")

    coco = {
        "info": {
            "description": "Shrub instance-segmentation dataset converted from shapefiles",
            "version": "1.0",
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [
            {
                "id": args.category_id,
                "name": args.category_name,
                "supercategory": "vegetation",
            }
        ],
    }

    annotation_id = 1
    skipped_small = 0
    missing_shapefiles: list[Path] = []

    for image_id, image_path in enumerate(image_paths, start=1):
        relative_image_path = image_path.relative_to(args.images_dir).as_posix()
        # Uncomment the next line for identical stems
        # shapefile_path = shapefile_index.get(image_path.stem.casefold())
        shapefile_stem = image_path.stem.replace("Img_", "Pol_", 1)
        shapefile_path = shapefile_index.get(shapefile_stem.casefold())

        if shapefile_path is None and not args.allow_missing_shapefiles:
            missing_shapefiles.append(image_path)
            continue

        with rasterio.open(image_path) as image_dataset:
            width = image_dataset.width
            height = image_dataset.height
            transform = image_dataset.transform
            image_crs = image_dataset.crs

            coco["images"].append(
                {
                    "id": image_id,
                    "file_name": relative_image_path,
                    "width": width,
                    "height": height,
                }
            )

            geometries = []
            if shapefile_path is not None:
                geometries = load_geometries_for_image(
                    shapefile_path,
                    image_crs,
                    fix_invalid_geometries=args.fix_invalid_geometries,
                )

            for geometry in geometries:
                binary_mask = rasterize(
                    [(geometry, 1)],
                    out_shape=(height, width),
                    transform=transform,
                    fill=0,
                    dtype=np.uint8,
                    all_touched=False,
                ).astype(bool)

                area = int(np.count_nonzero(binary_mask))
                if area < args.minimum_area_pixels:
                    skipped_small += 1
                    continue

                coco["annotations"].append(
                    {
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": args.category_id,
                        "segmentation": encode_binary_mask(binary_mask),
                        "area": float(area),
                        "bbox": mask_bbox(binary_mask),
                        "iscrowd": 0,
                    }
                )
                annotation_id += 1

        print(
            f"Processed {image_path.name}: "
            f"{sum(a['image_id'] == image_id for a in coco['annotations'])} instances"
        )

    if missing_shapefiles:
        missing_names = "\n".join(f"  {path}" for path in missing_shapefiles)
        raise FileNotFoundError(
            "No matching shapefile was found for these images:\n"
            f"{missing_names}\n"
            "Use --allow-missing-shapefiles only when these images genuinely "
            "contain no annotated shrubs."
        )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as output_file:
        json.dump(coco, output_file, ensure_ascii=False, indent=2)

    print(f"\nCOCO JSON written to: {args.output_json}")
    print(f"Images: {len(coco['images'])}")
    print(f"Instances: {len(coco['annotations'])}")
    print(f"Skipped instances below area threshold: {skipped_small}")
    return coco


def main() -> None:
    args = parse_args()
    convert_dataset(args)


if __name__ == "__main__":
    main()
