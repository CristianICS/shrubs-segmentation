"""
Merge predicted mask polygons and convert them to geographic coordinates.

Refine detections and confidence scores as a spatial grouping step:

1. Convert all tile polygons to one common projected CRS.
2. Group polygons that intersect, touch, or lie within a small distance.
3. Union the polygons in each group.
4. Compute the maximum, mean, and median of all detection scores in that group.
5. Save one feature per refined shrub.
"""
from pathlib import Path
from statistics import mean, median
import argparse
import json

import geopandas as gpd
import rasterio
from affine import Affine
from rasterio.crs import CRS
from rasterio.warp import transform_geom
from shapely.geometry import shape, mapping
from shapely.ops import unary_union


ROOT = Path(__file__).absolute().parent.parent

MASKS_DIR = ROOT / "data" / "rasters" / "deployment_tiles" / "matterport_masks"
OUT_PATH = ROOT / "results" / "predicted_shrub_masks.geojson"
IMG_EPSG = 25830


def pixel_ring_to_map_coordinates(ring, transform, pixel_center=False):
    """
    Convert a polygon ring from [column, row] to map coordinates.

    Parameters
    ----------
    ring : list
        Coordinates in [[column, row], ...] format.
    transform: affine.Affine
        Raster affine transformation.
    pixel_center : bool
        When True, add 0.5 to the column and row before transforming.
    """
    converted_ring = []

    offset = 0.5 if pixel_center else 0.0

    for column, row in ring:
        map_x, map_y = transform * (
            float(column) + offset,
            float(row) + offset,
        )

        converted_ring.append([float(map_x), float(map_y)])

    # Ensure the GeoJSON ring is closed.
    if converted_ring and converted_ring[0] != converted_ring[-1]:
        converted_ring.append(converted_ring[0])

    return converted_ring


def pixel_geometry_to_map_geometry(geometry, transform, pixel_center=False):
    """
    Convert a GeoJSON Polygon or MultiPolygon from pixel coordinates
    to the TIFF coordinate system.

    The first ring is the exterior boundary. Additional rings are holes.

    Parameters
    ----------
    geometry : json
        A GeoJSON object with target rings.
    transform: affine.Affine
        Raster affine transformation.
    pixel_center : bool
        When True, add 0.5 to the column and row before transforming.
    """
    geometry_type = geometry.get("type")
    coordinates = geometry.get("coordinates", [])

    if geometry_type == "Polygon":
        converted_coordinates = [
            pixel_ring_to_map_coordinates(ring, transform, pixel_center)
            for ring in coordinates
            if ring
        ]

    elif geometry_type == "MultiPolygon":
        converted_coordinates = [
            [
                pixel_ring_to_map_coordinates(
                    ring,
                    transform,
                    pixel_center
                )
                for ring in polygon
                if ring
            ]
            for polygon in coordinates
            if polygon
        ]

    else:
        return None

    if not converted_coordinates:
        return None

    return {
        "type": geometry_type,
        "coordinates": converted_coordinates,
    }


class UnionFind:
    """Efficiently group connected polygon detections."""

    def __init__(self, size):
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, value):
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]

        return value

    def union(self, first, second):
        root_first = self.find(first)
        root_second = self.find(second)

        if root_first == root_second:
            return

        if self.rank[root_first] < self.rank[root_second]:
            self.parent[root_first] = root_second

        elif self.rank[root_first] > self.rank[root_second]:
            self.parent[root_second] = root_first

        else:
            self.parent[root_second] = root_first
            self.rank[root_first] += 1


def _extract_records(
    features,
    prediction_path,
    image_name,
    source_crs,
    affine_transform,
    merge_crs,
    pixel_center,
):
    """Convert one prediction file's pixel-space features to map-space records."""
    records = []
    for feature_index, feature in enumerate(features):
        pixel_geometry = feature.get("geometry") or {}

        map_geometry = pixel_geometry_to_map_geometry(
            pixel_geometry,
            affine_transform,
            pixel_center=pixel_center
        )

        if map_geometry is None:
            continue

        # Put all polygons in the same CRS before grouping.
        if merge_crs is not None and source_crs != merge_crs:
            map_geometry = transform_geom(
                src_crs=source_crs,
                dst_crs=merge_crs,
                geom=map_geometry,
                precision=8,
            )

        polygon = shape(map_geometry)

        if polygon.is_empty:
            continue

        # Repair some invalid polygons produced by contours.
        if not polygon.is_valid:
            polygon = polygon.buffer(0)

        if polygon.is_empty:
            continue

        properties = feature.get("properties") or {}

        score = properties.get("score")

        if score is None:
            continue

        records.append(
            {
                "geometry": polygon,
                "score": float(score),
                "class_id": properties.get("class_id"),
                "image_name": image_name,
                "prediction_file": prediction_path.name,
                "source_feature": feature_index,
                "instance_id": properties.get("instance_id"),
                "polygon_id": properties.get("polygon_id"),
            }
        )
    return records


def load_prediction_polygons(
    predictions_dir,
    images_dir=None,
    merge_crs=None,
    prediction_pattern="_masks.json",
    pixel_center=False,
):
    """
    Read all prediction files and convert polygon coordinates to merge_crs.

    Each prediction file can georeference itself two ways:

    - Embedded (preferred): its `metadata` object carries its own affine
      `transform` (6 coefficients) and `crs`, as written by the internal-
      tiling deployment scripts (`detectron2_deploy.py`, `matterport_deploy.py`,
      `sam3_deploy.py`). No paired GeoTIFF is required in that case.
    - Paired GeoTIFF (legacy): when `metadata` has no `transform`, the loader
      falls back to opening `<stem>.tif` in `images_dir` for its transform/CRS,
      matching one pre-cropped tile file per prediction file.

    Parameters
    ----------
    predictions_dir : str or Path
        Directory containing the per-tile/per-window prediction JSON files.
    images_dir : str or Path or None
        Directory containing the georeferenced TIFF files, used only for
        prediction files with no embedded transform.
    merge_crs : str
        Projected CRS used for grouping and distance calculations.
        For example, "EPSG:25830".
    prediction_pattern : str
        Filename pattern for prediction files. It forms '*' + <prediction_patterns>
    pixel_center : bool
        False treats stored coordinates as raster grid vertices.
        True treats them as pixel centers.
    """
    records = []
    retained_crs = None

    processed_files = 0
    skipped_files = 0

    prediction_paths = Path(predictions_dir).rglob(
        "*" + prediction_pattern
    )

    for prediction_path in prediction_paths:
        try:
            with prediction_path.open("r", encoding="utf-8") as file:
                prediction = json.load(file)

        except (OSError, json.JSONDecodeError) as error:
            print(
                f"Skipped invalid JSON {prediction_path}: {error}"
            )
            continue

        features = prediction.get("features", [])

        if not features:
            continue

        metadata = prediction.get("metadata") or {}
        embedded_transform = metadata.get("transform")

        if embedded_transform is not None:
            image_name = prediction_path.name.replace(prediction_pattern, "")
            source_crs = CRS.from_user_input(metadata["crs"])
            affine_transform = Affine(*embedded_transform[:6])

            if merge_crs is None:
                if retained_crs is None:
                    retained_crs = source_crs
                elif source_crs != retained_crs:
                    raise ValueError(
                        "The prediction files use different CRSs. "
                        "Provide target_crs to reproject them to one CRS."
                    )

            records.extend(_extract_records(
                features, prediction_path, image_name,
                source_crs, affine_transform, merge_crs, pixel_center,
            ))
            processed_files += 1
            continue

        if images_dir is None:
            print(f"Skipped file with no embedded transform and no images_dir: {prediction_path}")
            skipped_files += 1
            continue

        image_name = prediction_path.name.replace(prediction_pattern, ".tif")
        tif_path = Path(images_dir) / image_name

        if not tif_path.exists():
            print(f"Skipped file: {prediction_path}")
            print(f"Invalid reference image: {tif_path}")
            skipped_files += 1
            continue

        with rasterio.open(tif_path) as src:
            if src.crs is None:
                print(f"Skipped TIFF without CRS: {tif_path}")
                skipped_files += 1
                continue

            source_crs = src.crs
            affine_transform = src.transform

            if merge_crs is None:
                if retained_crs is None:
                    retained_crs = source_crs
                elif source_crs != retained_crs:
                    raise ValueError(
                        "The TIFF files use different CRSs. "
                        "Provide target_crs to reproject them to one CRS."
                    )

            records.extend(_extract_records(
                features, prediction_path, image_name,
                source_crs, affine_transform, merge_crs, pixel_center,
            ))

        processed_files += 1

    print(f"Prediction files processed: {processed_files}")
    print(f"Prediction files skipped: {skipped_files}")
    print(f"Polygon features saved: {len(records)}")

    if len(records) == 0:
        raise ValueError("There are no available masks to work with.")

    return gpd.GeoDataFrame(
        records,
        geometry="geometry",
        crs=merge_crs,
    )


def group_related_detections(
    detections,
    merge_distance=0.0,
    minimum_overlap_ratio=0.0,
    merge_same_tile=True,
):
    """
    Group detections belonging to the same shrub.

    Two polygons are grouped when they:

    - intersect or touch; or
    - are separated by no more than merge_distance.

    Parameters
    ----------
    detections : GeoDataFrame
        Individual mask polygons.
    merge_distance : float
        Maximum allowed distance between polygons, in CRS units.
        Use 0 to merge only touching or overlapping polygons.
    minimum_overlap_ratio : float
        Optional minimum intersection ratio.

        The ratio is:

            intersection area / minimum polygon area

        A value of 0 allows touching polygons to merge.
    merge_same_tile : bool
        If False, only polygons from different tiles can be grouped.
        This can reduce accidental merging of distinct detections
        produced within the same tile.
    """
    number_of_detections = len(detections)

    if number_of_detections == 0:
        return []

    union_find = UnionFind(number_of_detections)
    spatial_index = detections.sindex

    geometries = detections.geometry.tolist()
    tile_names = detections["image_name"].tolist()

    for first_index, first_geometry in enumerate(geometries):
        if merge_distance > 0:
            search_geometry = first_geometry.buffer(merge_distance)
        else:
            search_geometry = first_geometry

        candidate_indices = list(
            spatial_index.intersection(search_geometry.bounds)
        )

        for second_index in candidate_indices:
            if second_index <= first_index:
                continue

            if (
                not merge_same_tile
                and tile_names[first_index] == tile_names[second_index]
            ):
                continue

            second_geometry = geometries[second_index]

            intersects = first_geometry.intersects(second_geometry)

            if intersects:
                if minimum_overlap_ratio > 0:
                    intersection_area = first_geometry.intersection(
                        second_geometry
                    ).area

                    smallest_area = min(
                        first_geometry.area,
                        second_geometry.area,
                    )

                    overlap_ratio = (
                        intersection_area / smallest_area
                        if smallest_area > 0
                        else 0
                    )

                    should_merge = (
                        overlap_ratio >= minimum_overlap_ratio
                    )

                else:
                    # Includes touching at the tile boundary.
                    should_merge = True

            elif merge_distance > 0:
                should_merge = (
                    first_geometry.distance(second_geometry)
                    <= merge_distance
                )

            else:
                should_merge = False

            if should_merge:
                union_find.union(first_index, second_index)

    groups = {}

    for detection_index in range(number_of_detections):
        root = union_find.find(detection_index)

        groups.setdefault(root, []).append(detection_index)

    return list(groups.values())


def refine_detection_groups(
    detections,
    groups,
    close_gaps=False,
    merge_distance=0.0,
):
    """
    Union each polygon group and aggregate its confidence scores.
    """
    refined_records = []

    for shrub_id, member_indices in enumerate(groups):
        members = detections.iloc[member_indices]

        member_geometries = members.geometry.tolist()
        member_scores = members["score"].astype(float).tolist()

        if close_gaps and merge_distance > 0:
            # Optional morphological closing. This creates a connected
            # polygon when parts have a very small gap.
            half_distance = merge_distance / 2.0

            merged_geometry = unary_union(
                [
                    geometry.buffer(half_distance)
                    for geometry in member_geometries
                ]
            ).buffer(-half_distance)

        else:
            merged_geometry = unary_union(member_geometries)

        if merged_geometry.is_empty:
            continue

        if not merged_geometry.is_valid:
            merged_geometry = merged_geometry.buffer(0)

        class_ids = [
            class_id
            for class_id in members["class_id"].tolist()
            if class_id is not None
        ]

        # Usually all grouped detections have the same class.
        class_id = class_ids[0] if class_ids else None

        source_tiles = sorted(
            set(members["image_name"].tolist())
        )

        refined_record = {
            "shrub_id": int(shrub_id),
            "class_id": (
                int(class_id)
                if class_id is not None
                else None
            ),
            "score_max": float(max(member_scores)),
            "score_mean": float(mean(member_scores)),
            "score_median": float(median(member_scores)),
            "detection_count": int(len(member_scores)),
            "source_tile_count": int(len(source_tiles)),
            "source_tiles": source_tiles,
            "original_scores": member_scores,
            "area": float(merged_geometry.area),
            "geometry": merged_geometry,
        }

        refined_records.append(refined_record)

    return gpd.GeoDataFrame(
        refined_records,
        geometry="geometry",
        crs=detections.crs,
    )


def merge_and_refine_predictions(
    predictions_dir,
    output_path,
    merge_crs,
    images_dir=None,
    output_crs=None,
    prediction_pattern="_masks.json",
    merge_distance=0.0,
    minimum_overlap_ratio=0.0,
    merge_same_tile=True,
    close_gaps=False,
):
    """
    Merge all mask predictions and refine duplicated tile detections.

    Parameters
    ----------
    merge_crs : str
        Projected CRS used for spatial grouping, such as "EPSG:25830".

    output_crs : str or None
        CRS used in the output GeoJSON. When None, merge_crs is retained.

    merge_distance : float
        Maximum distance between polygons considered part of the same shrub.
        The unit is the unit of merge_crs, normally metres.

    minimum_overlap_ratio : float
        Minimum overlap relative to the smaller polygon. Use 0 to allow
        polygons that only touch at a tile boundary.

    merge_same_tile : bool
        True permits overlapping detections within the same tile to merge.
        False restricts refinement to detections from different tiles.

    close_gaps : bool
        When True, small gaps are geometrically closed using buffering.
    """
    detections = load_prediction_polygons(
        predictions_dir=predictions_dir,
        images_dir=images_dir,
        merge_crs=merge_crs,
        prediction_pattern=prediction_pattern,
    )

    if detections.empty:
        print("No valid mask polygons were found.")
        return None

    groups = group_related_detections(
        detections=detections,
        merge_distance=merge_distance,
        minimum_overlap_ratio=minimum_overlap_ratio,
        merge_same_tile=merge_same_tile,
    )

    refined = refine_detection_groups(
        detections=detections,
        groups=groups,
        close_gaps=close_gaps,
        merge_distance=merge_distance,
    )

    if output_crs is not None:
        refined = refined.to_crs(output_crs)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert list-valued fields to JSON strings because some GeoJSON
    # writing backends do not accept list attributes directly.
    refined_to_save = refined.copy()

    refined_to_save["source_tiles"] = refined_to_save[
        "source_tiles"
    ].apply(json.dumps)

    refined_to_save["original_scores"] = refined_to_save[
        "original_scores"
    ].apply(json.dumps)

    refined_to_save.to_file(
        output_path,
        driver="GeoJSON",
        index=False,
    )

    print(f"Original detections: {len(detections)}")
    print(f"Refined shrubs: {len(refined)}")
    print(f"Merged detections: {len(detections) - len(refined)}")
    print(f"Saved to: {output_path}")

    return refined


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge per-tile mask polygons across tile boundaries into georeferenced shrubs."
    )
    parser.add_argument(
        "--predictions", type=Path, default=MASKS_DIR,
        help=f"Directory with per-tile '*_masks.json' files (default: {MASKS_DIR}).",
    )
    parser.add_argument(
        "--images", type=Path, default=None,
        help=(
            "Directory with paired georeferenced tile GeoTIFFs. Only needed for legacy "
            "prediction files with no embedded 'transform' in their metadata."
        ),
    )
    parser.add_argument("--output", type=Path, default=OUT_PATH, help=f"Output GeoJSON (default: {OUT_PATH}).")
    parser.add_argument("--crs", default=f"EPSG:{IMG_EPSG}", help="Projected CRS used for merging and output.")
    parser.add_argument("--prediction-pattern", default="_masks.json")
    parser.add_argument("--merge-distance", type=float, default=0.05)
    parser.add_argument("--minimum-overlap-ratio", type=float, default=0.0)
    parser.add_argument("--merge-same-tile", action="store_true", default=False)
    parser.add_argument(
        "--close-gaps", dest="close_gaps", action="store_true",
        help="Close small gaps between merged parts by buffering (default).",
    )
    parser.add_argument(
        "--no-close-gaps", dest="close_gaps", action="store_false",
        help="Merge parts without closing the gaps between them.",
    )
    parser.set_defaults(close_gaps=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    merge_and_refine_predictions(
        predictions_dir=args.predictions,
        images_dir=args.images,
        output_path=args.output,
        merge_crs=args.crs,
        output_crs=None,
        prediction_pattern=args.prediction_pattern,
        merge_distance=args.merge_distance,
        minimum_overlap_ratio=args.minimum_overlap_ratio,
        merge_same_tile=args.merge_same_tile,
        close_gaps=args.close_gaps,
    )