from pathlib import Path, PurePosixPath, PureWindowsPath
import os
import sys
import json

import cv2
import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))

# Import local Mask RCNN utilities
sys.path.append(ROOT_DIR)
from mrcnn.config import Config # type: ignore
from mrcnn import utils # type: ignore


# Define the configuration for training Mask R-CNN
# https://github.com/matterport/Mask_RCNN/blob/master/mrcnn/config.py
class BaselineShrubConfig(Config):
    """Base configuration options inherited for all the models."""
    NAME = "shrubs_baseline"

    NUM_CLASSES = 1 + 1  # background + shrub

    # Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 4

    IMAGE_MIN_DIM = 448
    IMAGE_MAX_DIM = 448

    # Keep this deliberately smaller than those in the original Mask R-CNN
    # paper because higher values can cause unstable optimization with small batches.
    LEARNING_RATE = 0.001
    LEARNING_MOMENTUM = 0.9
    WEIGHT_DECAY = 1e-4

    BACKBONE = "resnet101"
    TRAIN_BN = False


class BestModelShrubInferenceConfig(Config):
    """Inference configuration reconstructed from saved experiment metadata."""

    _COMPUTED_PARAMETERS = {
        "BATCH_SIZE",
        "IMAGE_META_SIZE",
        "IMAGE_SHAPE",
    }
    _ARRAY_PARAMETERS = {
        "BBOX_STD_DEV",
        "MEAN_PIXEL",
        "RPN_BBOX_STD_DEV",
    }

    def __init__(self, metadata_path, weights_path=None):
        """
        Parameters
        ----------
        metadata_path : str or Path
            `best_model_configuration.json` written by the training notebook.
        weights_path : str or Path or None
            Checkpoint to load instead of the one stored in the metadata.
        """
        with open(metadata_path, "r", encoding="utf-8") as metadata_file:
            metadata = json.load(metadata_file)

        try:
            saved_configuration = metadata["configuration"]
        except KeyError as error:
            raise ValueError(
                "The model metadata must contain a 'configuration' object."
            ) from error

        for parameter, value in saved_configuration.items():
            if parameter in self._COMPUTED_PARAMETERS:
                continue
            if parameter in self._ARRAY_PARAMETERS:
                value = np.asarray(value)
            setattr(self, parameter, value)

        # Inference always processes one image at a time. Keep every prediction
        # so confidence thresholds can be evaluated after running the model.
        self.GPU_COUNT = 1
        self.IMAGES_PER_GPU = 1
        self.DETECTION_MIN_CONFIDENCE = 0.0

        if weights_path is not None:
            weights_path = Path(weights_path)
        else:
            try:
                stored_path = metadata["best_checkpoint_path"]
            except KeyError as error:
                raise ValueError(
                    "The model metadata must contain 'best_checkpoint_path'."
                ) from error
            weights_path = resolve_checkpoint_path(stored_path, metadata_path)

        if not weights_path.is_file():
            raise FileNotFoundError(
                f"Best-model weights were not found: {weights_path}. "
                "Pass the checkpoint path explicitly to override the one "
                f"stored in {metadata_path}."
            )

        self.best_model_path = str(weights_path)

        super().__init__()


def resolve_checkpoint_path(stored_path, metadata_path):
    """
    Locate the checkpoint recorded in the best-model metadata.

    A relative path is resolved against the metadata file's folder, so the
    repository can be moved or cloned elsewhere. An absolute path that no
    longer exists (e.g. written on another machine) falls back to
    `<metadata folder>/<run folder>/<checkpoint file>`, which is where the
    training notebook stores every run.
    """
    metadata_dir = Path(metadata_path).resolve().parent
    # Parse Windows-style paths explicitly so they also split correctly on
    # Linux/WSL, where backslashes are not path separators.
    if "\\" in stored_path:
        stored = PureWindowsPath(stored_path)
    else:
        stored = PurePosixPath(stored_path)

    if not stored.is_absolute():
        return metadata_dir.joinpath(*stored.parts)

    if Path(stored_path).is_file():
        return Path(stored_path)

    return metadata_dir / stored.parent.name / stored.name


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
