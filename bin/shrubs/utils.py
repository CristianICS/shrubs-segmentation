import os
import sys
import json

import cv2 # type: ignore
import numpy as np # type: ignore

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config # type: ignore
from mrcnn import utils # type: ignore

# Define the configuration for training Mask R-CNN
# https://github.com/matterport/Mask_RCNN/blob/master/mrcnn/config.py
class ShrubConfigV2(Config):
    """
    Configuration for training on the JuniperMapper dataset mimic Khaldi et al. 2024
    paper's model configuration.

    Derives from the base Config class and overrides values specific
    to the shrub shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "shrubs_cfg_v2"

    # Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 2

    NUM_CLASSES = 1 + 1  # background + shrub

    # Preserve original image content and pad dimensions to multiples of 64
    IMAGE_RESIZE_MODE = "square"
    IMAGE_MIN_DIM = 448
    IMAGE_MAX_DIM = 448
    IMAGE_MIN_SCALE = 0

    # COCO weights are usually on ResNet101
    # Available approximation: ResNet101-FPN, not ResNet101-C4
    BACKBONE = "resnet101"
    BACKBONE_STRIDES = [4, 8, 16, 32, 64]
    TOP_DOWN_PYRAMID_SIZE = 256

    # Maximum final detections per image
    DETECTION_MAX_INSTANCES = 256

    # Freeze batch normalization because the batch size is small
    TRAIN_BN = False

    # Optimizer
    # Learning rate and momentum
    LEARNING_RATE = 0.0025
    LEARNING_MOMENTUM = 0.9
    WEIGHT_DECAY = 0.0001
    GRADIENT_CLIP_NORM = 5.0

    # Match the 4000 maximum number of iterations
    STEPS_PER_EPOCH = 100
    TOTAL_EPOCHS = 40
    # Each artificial epoch processes: 100 batches × 2 images = 200 image samples
    # Thus, an 'epoch' no longer means one full pass over all the training images.
    # That is acceptable when the source experiment is defined primarily in terms of
    # iterations rather than dataset passes.


class ShrubConfigV1(Config):
    """
    Configuration for training on the JuniperMapper, best results for 
    current Mask RCNN implementation.

    Derives from the base Config class and overrides values specific
    to the shrub shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "shrubs_cfg_v1"

    # Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 4

    NUM_CLASSES = 1 + 1  # background + shrub

    # Preserve original image content and pad dimensions to multiples of 64
    IMAGE_RESIZE_MODE = "square"
    IMAGE_MIN_DIM = 448
    IMAGE_MAX_DIM = 448
    IMAGE_MIN_SCALE = 0

    # Speed up the process
    # COCO weights are usually on ResNet101
    # Switch to resnet50 cuts compute and memory usage by ~35%.
    BACKBONE = "resnet101"


class ShrubInferenceConfig(Config):
    """Configurations for the model in inference mode."""
    NAME = "shrubs_cfg_inference"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    NUM_CLASSES = 1 + 1  # background + shrub
    
    IMAGE_RESIZE_MODE = "square"
    IMAGE_MIN_DIM = 448
    IMAGE_MAX_DIM = 448
    IMAGE_MIN_SCALE = 0

    # Evaluate the full confidence range from zero
    DETECTION_MIN_CONFIDENCE = 0.0


class ShrubMapDataset(utils.Dataset):
    """Define the shrub dataset to create the final map. No mask configuration"""

    def load_dataset(self, img_dir):
        """
        Adds information (image ID, image path and annotation file path)
        about each image in a dictionary.
        """
        # Add classes
        self.add_class("dataset", 1, "shrub")

        # Add images
        for i, img_name in enumerate(os.listdir(img_dir)):
            
            if not img_name.endswith(".tif"):
                continue
            
            img_path = os.path.join(img_dir, img_name)
            self.add_image('dataset', image_id=i, path=img_path)

    def image_reference(self, image_id):
        """Return image path"""
        return self.image_info[image_id]["path"]

    def load_mask(self, image_id):
        """
        Return an empty mask and avoid the warning.
        """
        mask = np.empty([0, 0, 0])
        class_ids = np.empty([0], np.int32)
        return mask, class_ids


class ShrubDataset(utils.Dataset):
    """Define the shrub dataset."""

    def load_dataset(self, img_dir, ann_pnts_file):
        """
        Adds information (image ID, image path and annotation file path)
        about each image in a dictionary.
        """
        # Add classes
        self.add_class("dataset", 1, "shrub")

        # Open the json with the associated annotation polygons
        with open(ann_pnts_file, "rb") as f:
            masks_dict = json.load(f)

        img_ids_with_ann = masks_dict.keys()

        # Add images
        for img_name in os.listdir(img_dir):
            
            if not img_name.endswith(".tif"):
                continue
            
            img_path = os.path.join(img_dir, img_name)
            # Extract image associated annotation file
            # E.g., if img_name is Img_1.tif, its ann file is Pol_1.shp
            img_id = os.path.splitext(img_name)[0].split("_")[1]
            
            if img_id in img_ids_with_ann:
                self.add_image('dataset', image_id=img_id, path=img_path, pnts_dict=ann_pnts_file)


    def load_mask(self, image_id):
        """Generate instance masks for shrubs of the given image ID from shp files."""
        info = self.image_info[image_id]
        img_dict_id = info["id"]
        ann_dict_path = info['pnts_dict']

        with open(ann_dict_path, "rb") as f:
            # The dictionary contains a list for each image with multipolygon points
            masks_dict = json.load(f)
        
        pnts_list = masks_dict[img_dict_id]["pnts"]
        height = masks_dict[img_dict_id]["height"]
        width = masks_dict[img_dict_id]["width"]
        # Initialize a new numpy mask: 0 no shrub, 1 shrub
        # Each annotation layer could have several polygons/masks
        masks = []
        class_ids = list()
        for pnts in pnts_list:

            mask = np.zeros((height, width), dtype=np.uint8)
            # Transform points to the cv2 format
            pnts = np.array(pnts).round().astype("int32")
            cv2.fillPoly(mask, [pnts], 1)
            masks.append(mask)
            class_ids.append(self.class_names.index("shrub"))

        if len(masks) == 0:
            # There is no annotations
            masks = np.zeros((height, width, 0), dtype="uint8")
            class_ids = np.array([], dtype="int32")
        else:
            # Join all the multipolygon masks together
            masks = np.stack(masks, axis=-1).astype("uint8")
            # All instances are shrubs
            class_ids = np.asarray(class_ids, dtype="int32")

        return masks, class_ids

    def image_reference(self, image_id):
        """Return image path"""
        return self.image_info[image_id]["path"]


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
