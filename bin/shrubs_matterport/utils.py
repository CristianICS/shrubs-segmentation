from datetime import datetime, timezone
from pathlib import Path
import os
import json
import random

import cv2
import numpy as np


def save_masks_as_geojson(
    img_id,
    masks,
    scores,
    dataset,
    out_dir,
    class_ids=None,
    min_area_pixels=0,
    model_score_threshold=0.5
):
    """
    Save instance masks as GeoJSON polygons in pixel coordinates.

    Coordinates are stored as [x, y], where:
        x = image column
        y = image row

    The output has no CRS or real-world coordinates.
    """
    if masks.ndim != 3:
        raise ValueError(
            "masks must have shape (height, width, num_instances)"
        )

    num_instances = masks.shape[2]

    if len(scores) != num_instances:
        raise ValueError("scores must contain one value per mask")

    if class_ids is not None and len(class_ids) != num_instances:
        raise ValueError("class_ids must contain one value per mask")

    # Skip tiles with no predicted instances or no positive mask pixels.
    if num_instances == 0 or not np.any(masks):
        return None

    features = []

    for instance_id in range(num_instances):
        mask = masks[:, :, instance_id].astype(np.uint8)

        if not mask.any():
            continue

        if float(scores[instance_id]) < model_score_threshold:
            continue

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        for polygon_id, contour in enumerate(contours):
            area_pixels = cv2.contourArea(contour)

            if area_pixels < min_area_pixels:
                continue

            # A valid polygon ring needs at least three points.
            if len(contour) < 3:
                continue

            coordinates = contour[:, 0, :].astype(float).tolist()

            # GeoJSON polygon rings must be closed.
            if coordinates[0] != coordinates[-1]:
                coordinates.append(coordinates[0])

            properties = {
                "image_id": int(img_id),
                "instance_id": int(instance_id),
                "polygon_id": int(polygon_id),
                "score": float(scores[instance_id]),
                "area_pixels": float(area_pixels),
                "coordinate_system": "pixel",
            }

            if class_ids is not None:
                properties["class_id"] = int(class_ids[instance_id])

            feature = {
                "type": "Feature",
                "properties": properties,
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [coordinates],
                },
            }

            features.append(feature)

    # Skip the tile if all contours were rejected, for example because
    # they were smaller than min_area_pixels.
    if not features:
        return None
    
    feature_collection = {
        "type": "FeatureCollection",
        "metadata": {
            "coordinate_system": "pixel",
            "coordinate_order": ["x", "y"],
            "image_width": int(masks.shape[1]),
            "image_height": int(masks.shape[0]),
        },
        "features": features,
    }

    image_path = dataset.image_info[img_id]["path"]
    image_name = os.path.basename(image_path)
    output_name = os.path.splitext(image_name)[0] + "_masks.json"
    output_path = os.path.join(out_dir, output_name)

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(feature_collection, file)

    return output_path


def save_masks_deprecated(img_id, masks, dataset, out_dir):
    """Store the predicted masks as png images during deployment."""
    N = masks.shape[2]
    if not N:
        # No instances to display
        return
    
    merged_mask = np.any(masks, axis=2).astype(np.uint8)

    # Save as png (add the CRS with rasterio later)
    img_name = os.path.basename(dataset.image_info[img_id]["path"])
    out_name = os.path.splitext(img_name)[0] + "_mask.png"
    out_path = os.path.join(out_dir, out_name)

    cv2.imwrite(out_path, merged_mask)
