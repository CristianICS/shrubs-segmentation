"""
Merge SAM 3's per-tile shrub detections across tile boundaries.

`sam3_shrubs.py` writes one `<stem>_shrub_instances.csv` of pixel bounding
boxes per input GeoTIFF; a shrub that straddles a tile seam can appear as two
separate rows. This reuses the grouping/union logic already written for the
Matterport pipeline in `refine_detections.py` (`group_related_detections`,
`refine_detection_groups`), swapping in a loader for SAM3's box+CSV output
instead of `*_masks.json` polygons.

Example usage:

python sam3_refine_detections.py results/sam3/zero-shot \
  --output results/sam3_refined.geojson --crs EPSG:25830
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import geopandas as gpd
import rasterio
from shapely.geometry import box

from refine_detections import group_related_detections, refine_detection_groups


def load_sam3_instances(results_dir: Path, crs: str) -> gpd.GeoDataFrame:
    """Read every `*_shrub_instances.csv` and project its boxes to `crs`."""
    records = []
    for csv_path in sorted(results_dir.glob("*_shrub_instances.csv")):
        tif_path = csv_path.with_name(csv_path.name.replace("_shrub_instances.csv", "_shrubs.tif"))
        with rasterio.open(tif_path) as source:
            transform = source.transform
            for row in csv.DictReader(csv_path.open(encoding="utf-8")):
                x0, y0 = transform * (float(row["xmin"]), float(row["ymin"]))
                x1, y1 = transform * (float(row["xmax"]), float(row["ymax"]))
                records.append({
                    "geometry": box(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)),
                    "score": float(row["score"]),
                    "class_id": None,
                    "image_name": tif_path.name,
                })
    if not records:
        raise ValueError(f"No *_shrub_instances.csv files found in {results_dir}.")
    return gpd.GeoDataFrame(records, geometry="geometry", crs=crs)


def merge_sam3_detections(
    results_dir: Path,
    output_path: Path,
    crs: str,
    merge_distance: float = 0.0,
    minimum_overlap_ratio: float = 0.0,
) -> gpd.GeoDataFrame:
    """Group overlapping/touching SAM 3 boxes across tiles into one shrub each."""
    detections = load_sam3_instances(results_dir, crs)
    groups = group_related_detections(
        detections, merge_distance=merge_distance, minimum_overlap_ratio=minimum_overlap_ratio,
    )
    refined = refine_detection_groups(
        detections, groups, close_gaps=merge_distance > 0, merge_distance=merge_distance,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    to_save = refined.copy()
    to_save["source_tiles"] = to_save["source_tiles"].apply(json.dumps)
    to_save["original_scores"] = to_save["original_scores"].apply(json.dumps)
    to_save.to_file(output_path, driver="GeoJSON", index=False)

    print(f"Original detections: {len(detections)}")
    print(f"Refined shrubs: {len(refined)}")
    print(f"Saved to: {output_path}")
    return refined


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge SAM 3's per-tile shrub boxes across tile boundaries."
    )
    parser.add_argument("results_dir", type=Path, help="Folder with *_shrub_instances.csv and *_shrubs.tif.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--crs", required=True, help='Projected CRS used for merging, e.g. "EPSG:25830".')
    parser.add_argument("--merge-distance", type=float, default=0.0)
    parser.add_argument("--minimum-overlap-ratio", type=float, default=0.0)
    args = parser.parse_args()
    merge_sam3_detections(
        args.results_dir, args.output, args.crs, args.merge_distance, args.minimum_overlap_ratio,
    )


if __name__ == "__main__":
    main()
