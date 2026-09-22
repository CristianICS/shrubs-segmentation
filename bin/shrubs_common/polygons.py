"""Shared pixel-mask -> georeferenced polygon JSON helpers.

Every deployment script (Detectron2, Matterport, SAM 3) tiles its source
orthomosaic internally into windows (see `tiling.py`), runs inference per
window, and writes one `*_masks.json` per window with real per-instance mask
polygons -- not bounding boxes. Coordinates are window-local pixel space, and
each file embeds that window's own affine transform and CRS in `metadata`,
so `refine_detections.py` can georeference and merge every model's output
the same way, with no paired per-window GeoTIFF file required on disk.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


def masks_to_features(
    masks: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    min_area_pixels: float = 0.0,
) -> list[dict]:
    """Extract one polygon Feature per contour of a boolean (H, W, N) mask stack."""
    features = []
    for instance_id in range(masks.shape[-1]):
        mask = masks[:, :, instance_id].astype(np.uint8)
        if not mask.any():
            continue

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for polygon_id, contour in enumerate(contours):
            area_pixels = cv2.contourArea(contour)
            if area_pixels < min_area_pixels or len(contour) < 3:
                continue

            coordinates = contour[:, 0, :].astype(float).tolist()
            if coordinates[0] != coordinates[-1]:
                coordinates.append(coordinates[0])

            features.append({
                "type": "Feature",
                "properties": {
                    "instance_id": int(instance_id),
                    "polygon_id": int(polygon_id),
                    "score": float(scores[instance_id]),
                    "area_pixels": float(area_pixels),
                    "coordinate_system": "pixel",
                    "class_id": int(class_ids[instance_id]),
                },
                "geometry": {"type": "Polygon", "coordinates": [coordinates]},
            })
    return features


def write_window_masks(
    output_path: Path,
    masks: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    window_transform,
    crs,
    min_area_pixels: float = 0.0,
) -> Path | None:
    """Write one window's mask polygons with its embedded georeferencing.

    Returns `output_path`, or None when the window had no surviving instances
    (nothing is written in that case).
    """
    features = masks_to_features(masks, scores, class_ids, min_area_pixels)
    if not features:
        return None

    feature_collection = {
        "type": "FeatureCollection",
        "metadata": {
            "coordinate_system": "pixel",
            "coordinate_order": ["x", "y"],
            "image_width": int(masks.shape[1]),
            "image_height": int(masks.shape[0]),
            "transform": [
                window_transform.a, window_transform.b, window_transform.c,
                window_transform.d, window_transform.e, window_transform.f,
            ],
            "crs": str(crs),
        },
        "features": features,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(feature_collection, file)
    return output_path
