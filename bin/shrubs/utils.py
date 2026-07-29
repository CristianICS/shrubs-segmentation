import os
import sys
import math
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
    """Define the shrub dataset to create the final map."""

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
