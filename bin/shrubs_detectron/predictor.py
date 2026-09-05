"""Detectron2 Mask R-CNN inference helpers shared by evaluation and prediction scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np
import torch

from detectron2 import model_zoo
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data.transforms import ResizeShortestEdge
from detectron2.modeling import build_model

Prediction = Mapping[str, np.ndarray]


def build_inference_cfg(
    weights: Path,
    *,
    config: str = "COCO-InstanceSegmentation/mask_rcnn_R_101_C4_3x.yaml",
    num_classes: int = 1,
    device: str | None = None,
    detections_per_image: int = 256,
    min_size_test: int = 800,
    max_size_test: int = 1333,
):
    """Build a frozen Detectron2 config for inference with a trained checkpoint."""
    cfg = get_cfg()
    config_path = Path(config)
    if config_path.is_file():
        cfg.merge_from_file(str(config_path))
    else:
        cfg.merge_from_file(model_zoo.get_config_file(config))

    cfg.MODEL.WEIGHTS = str(weights)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.0
    cfg.MODEL.DEVICE = device or ("cuda" if torch.cuda.is_available() else "cpu")

    cfg.TEST.DETECTIONS_PER_IMAGE = detections_per_image

    # Detectron2 normally loads OpenCV images in BGR order.
    cfg.INPUT.FORMAT = "BGR"
    cfg.INPUT.MIN_SIZE_TEST = min_size_test
    cfg.INPUT.MAX_SIZE_TEST = max_size_test

    cfg.freeze()
    return cfg


class ScaleMatchedPredictor:
    """Run inference using the configured Detectron2 test resizing."""

    def __init__(self, cfg) -> None:
        self.cfg = cfg.clone()
        self.model = build_model(self.cfg)
        self.model.eval()

        DetectionCheckpointer(self.model).load(self.cfg.MODEL.WEIGHTS)

        self.input_format = self.cfg.INPUT.FORMAT

        self.augmentation = ResizeShortestEdge(
            short_edge_length=[
                self.cfg.INPUT.MIN_SIZE_TEST,
                self.cfg.INPUT.MIN_SIZE_TEST,
            ],
            max_size=self.cfg.INPUT.MAX_SIZE_TEST,
        )

    @torch.inference_mode()
    def __call__(self, image: np.ndarray) -> Prediction:
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(
                f"Expected an HxWx3 image; received {image.shape}."
            )

        original_height, original_width = image.shape[:2]

        transformed_image = self.augmentation.get_transform(
            image
        ).apply_image(image)

        tensor = torch.as_tensor(
            np.ascontiguousarray(
                transformed_image.transpose(2, 0, 1)
            ),
            dtype=torch.float32,
        )

        output = self.model(
            [
                {
                    "image": tensor,
                    "height": original_height,
                    "width": original_width,
                }
            ]
        )[0]

        instances = output["instances"].to("cpu")

        if instances.has("pred_masks"):
            masks_nhw = (
                instances.pred_masks.numpy()
                .astype(bool, copy=False)
            )
        else:
            masks_nhw = np.zeros(
                (0, original_height, original_width),
                dtype=bool,
            )

        boxes_xyxy = (
            instances.pred_boxes.tensor.numpy().astype(
                np.float32,
                copy=False,
            )
            if instances.has("pred_boxes")
            else np.zeros((0, 4), dtype=np.float32)
        )

        return {
            "boxes": boxes_xyxy,
            "class_ids": (
                instances.pred_classes.numpy()
                .astype(np.int64, copy=False)
            ),
            "scores": (
                instances.scores.numpy()
                .astype(np.float32, copy=False)
            ),
            "masks": np.moveaxis(masks_nhw, 0, -1),
        }
