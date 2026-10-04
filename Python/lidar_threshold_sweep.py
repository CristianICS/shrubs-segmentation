"""
Calibrate the LiDAR refinement thresholds on the field-reference masks.

Runs `lidar_refine_masks.refine` on the annotated masks for every combination
of `--h-tall` and `--tall-fraction` and counts, per growth form (tree / shrub),
how many masks end up in each status. Growth forms come from the `shrub` column
of the annotations (1 shrub, 0 tree), as in `R/mask_height_stats.R`.

A tree is counted as rejected when its status is `tree` (directly or because
almost nothing was left after clipping); the same status on a shrub is a loss.

With `--loo`, the calibration is also checked by leave-one-ROI-out: each mask is
assigned to the `square_rois` square it overlaps most (or the nearest one), and
for every ROI the thresholds are selected on the masks of the other ROIs and
applied to the held-out masks. The selection rule (`--criterion`) is:

    no-shrub-change   among the combinations that neither reject nor clip any
                      shrub, the one that rejects most trees
    no-shrub-loss     same, but clipping shrubs is allowed

Ties go to the higher `h_tall` and then the higher `tall_fraction`. Applied to
all masks with `no-shrub-change` this gives the thresholds used in the paper
(1.7 m, 0.1). The selection is made separately for each `--min-coverage`.

Example usage (inside the `mrcnn_masks` environment):

python Python/lidar_threshold_sweep.py --loo
"""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import geopandas as gpd
import pandas as pd
import rasterio

from lidar_refine_masks import refine

ROOT = Path(__file__).resolve().parent.parent
STATUSES = ["tree", "clipped", "keep", "review"]
FORMS = ["tree", "shrub"]


def count_statuses(status: pd.DataFrame) -> pd.DataFrame:
    """Masks per growth form and status for each threshold combination."""
    counts = (status.groupby(["h_tall", "tall_fraction", "min_coverage", "form", "status"])
              .size().unstack(["form", "status"])
              .reindex(columns=pd.MultiIndex.from_product([FORMS, STATUSES]), fill_value=0)
              .fillna(0).astype(int))
    counts.columns = [f"{form}_{s}" for form, s in counts.columns]
    return counts.reset_index()


def select(counts: pd.DataFrame, criterion: str) -> pd.Series | None:
    """Thresholds chosen by `criterion` from a table built by `count_statuses`."""
    safe = counts["shrub_tree"] == 0
    if criterion == "no-shrub-change":
        safe &= counts["shrub_clipped"] == 0
    candidates = counts[safe]
    if candidates.empty:
        return None
    return candidates.sort_values(["tree_tree", "h_tall", "tall_fraction"]).iloc[-1]


def assign_rois(masks: gpd.GeoDataFrame, rois: gpd.GeoDataFrame) -> pd.Series:
    """Id of the ROI each mask overlaps most, or of the nearest ROI if none."""
    roi_ids = []
    for geometry in masks.geometry:
        overlap = rois.geometry.intersection(geometry).area
        best = overlap.idxmax() if overlap.max() > 0 else rois.geometry.distance(geometry).idxmin()
        roi_ids.append(rois.loc[best, "id"])
    return pd.Series(roi_ids, index=masks.index)


def leave_one_roi_out(status: pd.DataFrame, criterion: str) -> pd.DataFrame:
    """Select thresholds without each ROI and report the status of its masks."""
    rows = []
    for (min_coverage, roi), held_out in status.groupby(["min_coverage", "roi"]):
        training = status[(status["min_coverage"] == min_coverage) & (status["roi"] != roi)]
        chosen = select(count_statuses(training), criterion)
        row = {"min_coverage": min_coverage, "roi": roi,
               "h_tall": chosen["h_tall"] if chosen is not None else None,
               "tall_fraction": chosen["tall_fraction"] if chosen is not None else None}
        if chosen is not None:
            held_out = held_out[(held_out["h_tall"] == chosen["h_tall"])
                                & (held_out["tall_fraction"] == chosen["tall_fraction"])]
            for form, s in itertools.product(FORMS, STATUSES):
                row[f"{form}_{s}"] = int(((held_out["form"] == form) & (held_out["status"] == s)).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--annotations", type=Path, default=ROOT / "data" / "encinacorba_annotations.gpkg")
    parser.add_argument("--layer", default="field_work_masks")
    parser.add_argument("--rois-layer", default="square_rois")
    parser.add_argument("--chm", type=Path, default=ROOT / "data" / "rasters" / "encinacorba_chm.tif")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results" / "lidar_refined" / "threshold_sweep.csv",
                        help="Sweep table; the leave-one-ROI-out table is saved next to it.")
    parser.add_argument("--h-tall", type=float, nargs="+", default=[1.3, 1.5, 1.7, 1.9])
    parser.add_argument("--tall-fraction", type=float, nargs="+", default=[0.0, 0.1, 0.2, 0.3, 0.5])
    parser.add_argument("--min-coverage", type=float, nargs="+", default=[0.3])
    parser.add_argument("--min-cells", type=int, default=5)
    parser.add_argument("--min-area", type=float, default=0.25)
    parser.add_argument("--loo", action="store_true", help="Also run the leave-one-ROI-out check.")
    parser.add_argument("--criterion", choices=["no-shrub-change", "no-shrub-loss"],
                        default="no-shrub-change", help="Threshold selection rule (default: %(default)s).")
    args = parser.parse_args()

    masks = gpd.read_file(args.annotations, layer=args.layer)
    form = masks["shrub"].map({1: "shrub", 0: "tree"})
    rois = gpd.read_file(args.annotations, layer=args.rois_layer).to_crs(masks.crs)
    roi = assign_rois(masks, rois)

    # Status of every mask for every combination; the sweep and each
    # leave-one-ROI-out fold only aggregate this table
    frames = []
    with rasterio.open(args.chm) as chm:
        if masks.crs != chm.crs:
            masks = masks.to_crs(chm.crs)
        for h_tall, tall_fraction, min_coverage in itertools.product(
            args.h_tall, args.tall_fraction, args.min_coverage
        ):
            refined = refine(masks, chm, h_tall, tall_fraction, min_coverage,
                             args.min_cells, args.min_area, clip=True)
            frames.append(pd.DataFrame({
                "Id": masks["Id"], "roi": roi, "form": form, "status": refined["status"],
                "h_tall": h_tall, "tall_fraction": tall_fraction, "min_coverage": min_coverage,
            }))
    status = pd.concat(frames, ignore_index=True)

    table = count_statuses(status)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output, index=False)
    print(f"Trees: {(form == 'tree').sum()} | shrubs: {(form == 'shrub').sum()}")
    print(table.to_string(index=False))
    print(f"Saved to: {args.output}")
    if not args.loo:
        return

    for min_coverage, counts in table.groupby("min_coverage"):
        chosen = select(counts, args.criterion)
        if chosen is not None:
            print(f"\nAll masks, min_coverage {min_coverage}: h_tall {chosen['h_tall']}, "
                  f"tall_fraction {chosen['tall_fraction']} ({args.criterion})")

    loo = leave_one_roi_out(status, args.criterion)
    loo_path = args.output.with_name(args.output.stem + "_loo.csv")
    loo.to_csv(loo_path, index=False)
    print(f"\nLeave-one-ROI-out ({loo['roi'].nunique()} ROIs, {args.criterion}):")
    print(loo.to_string(index=False))
    for min_coverage, folds in loo.groupby("min_coverage"):
        print(f"\nmin_coverage {min_coverage}, thresholds selected (folds):")
        print(folds.groupby(["h_tall", "tall_fraction"]).size().to_string())
        print("Held-out masks pooled over folds:")
        print(folds[[f"{f}_{s}" for f in FORMS for s in STATUSES]].sum().to_string())
    print(f"Saved to: {loo_path}")


if __name__ == "__main__":
    main()
