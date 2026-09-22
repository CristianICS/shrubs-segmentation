"""
Refine predicted shrub masks with the LiDAR canopy height model (CHM).

Architecture-agnostic post-processing: every polygon is assessed against the
CHM independently, so the same thresholds apply to the output of any model.

For each mask the CHM cells whose centre falls inside it are summarised
(coverage, p90 height, fraction of tall cells) and one decision is made:

    review      not enough direct LiDAR support (coverage < --min-coverage or
                fewer than --min-cells valid cells). The mask is kept.
    tree        h_p90 > --h-tall AND tall_fraction >= --tall-fraction.
                The mask is rejected as a tree.
    clipped     shrub-compatible, but some cells exceed --h-tall (bleed into a
                neighbouring crown). Those cells are cut away; the mask is
                rejected instead if what remains is smaller than --min-area.
    keep        shrub-compatible, unchanged.

The CHM must be the non-interpolated one, so that NaN really means "no
LiDAR support". Use --stats-only on the annotations to inspect the height
distributions before choosing thresholds.

Example usage:

python lidar_refine_masks.py results/predicted_shrub_masks.geojson \
  --chm data/rasters/encinacorba_chm.tif \
  --output results/lidar_refined/predicted_shrub_masks.gpkg \
  --h-tall 1.7 --tall-fraction 0.1 --min-coverage 0.3
"""
from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import geometry_mask, shapes
from rasterio.windows import Window
from shapely.geometry import shape
from shapely.ops import unary_union

KEPT_STATUSES = ("keep", "clipped", "review")


def polygon_cells(chm: rasterio.DatasetReader, geometry) -> tuple[np.ndarray, Window, np.ndarray]:
    """CHM values of the cells whose centre is inside `geometry` (NaN elsewhere)."""
    # Smallest whole-cell window covering the bounds, clamped to the raster. It is
    # built by hand because Window.intersection raises on an empty result, which
    # happens for masks smaller than a cell or outside the raster
    if geometry.is_empty:
        return np.empty((0, 0)), None, np.empty((0, 0), dtype=bool)
    # Pixel coordinates come from the affine inverse (plain Python) rather than
    # rasterio.windows.from_bounds, whose numpy.linalg.solve call crashes in some
    # Windows conda builds
    xmin, ymin, xmax, ymax = geometry.bounds
    inverse = ~chm.transform
    corners = [inverse * (x, y) for x in (xmin, xmax) for y in (ymin, ymax)]
    cols, rows = zip(*corners)
    col0 = max(int(np.floor(min(cols))), 0)
    row0 = max(int(np.floor(min(rows))), 0)
    col1 = min(int(np.ceil(max(cols))), chm.width)
    row1 = min(int(np.ceil(max(rows))), chm.height)
    if col1 <= col0 or row1 <= row0:
        return np.empty((0, 0)), None, np.empty((0, 0), dtype=bool)
    window = Window(col0, row0, col1 - col0, row1 - row0)
    values = chm.read(1, window=window, masked=True).astype("float64").filled(np.nan)
    inside = ~geometry_mask(
        [geometry], out_shape=values.shape, transform=chm.window_transform(window),
    )
    values[~inside] = np.nan
    values[~np.isfinite(values)] = np.nan
    return values, window, inside


def summarise(chm: rasterio.DatasetReader, geometry, h_tall: float) -> dict:
    """Coverage and height statistics of one mask over the CHM."""
    values, _, inside = polygon_cells(chm, geometry)
    n_cells = int(inside.sum())
    valid = values[np.isfinite(values)]
    if valid.size == 0:
        return {"n_cells": n_cells, "n_valid": 0, "coverage": 0.0, "h_p90": np.nan,
                "h_mean": np.nan, "h_max": np.nan, "tall_fraction": np.nan}
    return {
        "n_cells": n_cells,
        "n_valid": int(valid.size),
        "coverage": valid.size / max(n_cells, 1),
        "h_p90": float(np.percentile(valid, 90)),
        "h_mean": float(valid.mean()),
        "h_max": float(valid.max()),
        "tall_fraction": float((valid > h_tall).mean()),
    }


def clip_tall_cells(chm: rasterio.DatasetReader, geometry, h_tall: float):
    """Remove from `geometry` the CHM cells taller than `h_tall`; keep the largest part."""
    values, window, _ = polygon_cells(chm, geometry)
    tall = np.where(np.isfinite(values), values > h_tall, False).astype("uint8")
    transform = chm.window_transform(window)
    tall_shapes = [shape(g) for g, v in shapes(tall, mask=tall.astype(bool), transform=transform) if v == 1]
    if not tall_shapes:
        return geometry
    remaining = geometry.difference(unary_union(tall_shapes))
    if remaining.is_empty:
        return remaining
    if remaining.geom_type == "MultiPolygon":
        remaining = max(remaining.geoms, key=lambda g: g.area)
    return remaining


def refine(
    predictions: gpd.GeoDataFrame,
    chm: rasterio.DatasetReader,
    h_tall: float,
    tall_fraction: float,
    min_coverage: float,
    min_cells: int,
    min_area: float,
    clip: bool,
) -> gpd.GeoDataFrame:
    """Return `predictions` with CHM statistics, a `status` and (if clipped) new geometry."""
    rows, geometries = [], []
    for geometry in predictions.geometry:
        stats = summarise(chm, geometry, h_tall)
        geometry_out = geometry
        if stats["coverage"] < min_coverage or stats["n_valid"] < min_cells:
            status = "review"
        elif stats["h_p90"] > h_tall and stats["tall_fraction"] >= tall_fraction:
            status = "tree"
        elif stats["tall_fraction"] > 0 and clip:
            geometry_out = clip_tall_cells(chm, geometry, h_tall)
            status = "tree" if geometry_out.is_empty or geometry_out.area < min_area else "clipped"
        else:
            status = "keep"
        rows.append({**stats, "status": status})
        geometries.append(geometry_out)

    refined = predictions.copy()
    for column in rows[0]:
        refined[column] = [row[column] for row in rows]
    refined.geometry = geometries
    return refined


def main() -> None:
    parser = argparse.ArgumentParser(description="Refine predicted masks with the LiDAR CHM.")
    parser.add_argument("predictions", type=Path, help="Polygon file (GeoJSON, GPKG, ...).")
    parser.add_argument("--chm", type=Path, required=True, help="Non-interpolated CHM GeoTIFF.")
    parser.add_argument("--output", type=Path, required=True,
                        help="Output vector file with the retained masks; rejected masks and "
                             "the statistics table are saved next to it.")
    parser.add_argument("--h-tall", type=float, default=1.7,
                        help="Height (m) above which a cell counts as tall vegetation.")
    parser.add_argument("--tall-fraction", type=float, default=0.1,
                        help="Minimum share of tall cells to reject a mask as a tree.")
    parser.add_argument("--min-coverage", type=float, default=0.3,
                        help="Minimum share of mask cells with direct LiDAR support.")
    parser.add_argument("--min-cells", type=int, default=5,
                        help="Minimum number of valid CHM cells per mask.")
    parser.add_argument("--min-area", type=float, default=0.25,
                        help="Minimum area (CRS units squared) left after clipping.")
    parser.add_argument("--no-clip", action="store_true",
                        help="Do not cut tall cells out of shrub-compatible masks.")
    parser.add_argument("--stats-only", action="store_true",
                        help="Only write the statistics table (e.g. to inspect annotations).")
    args = parser.parse_args()

    predictions = gpd.read_file(args.predictions)
    with rasterio.open(args.chm) as chm:
        if predictions.crs != chm.crs:
            predictions = predictions.to_crs(chm.crs)
        refined = refine(
            predictions, chm, args.h_tall, args.tall_fraction, args.min_coverage,
            args.min_cells, args.min_area, clip=not args.no_clip,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    stats_path = args.output.with_name(args.output.stem + "_stats.csv")
    refined.drop(columns="geometry").to_csv(stats_path, index=False)
    print(refined["status"].value_counts().to_string())
    print(f"Statistics saved to: {stats_path}")
    if args.stats_only:
        return

    kept = refined[refined["status"].isin(KEPT_STATUSES)]
    rejected = refined[~refined["status"].isin(KEPT_STATUSES)]
    driver = "GeoJSON" if args.output.suffix.lower() == ".geojson" else None
    kept.to_file(args.output, driver=driver)
    if len(rejected):
        rejected.to_file(args.output.with_name(args.output.stem + "_rejected" + args.output.suffix),
                         driver=driver)
    print(f"Input: {len(refined)} | retained: {len(kept)} | rejected: {len(rejected)}")
    print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()
