"""
Quick prediction, non-georeferenced sanity checks on loose images

Run a trained Detectron2 Mask R-CNN checkpoint on a plain set of RGB images.

Unlike `detectron2_evaluate.py`, this script requires no COCO ground-truth
annotations: it just segments whatever images it is given and writes the
predicted masks, per-pixel confidence, and per-instance boxes/scores.

Example usage:

python detectron2_predict.py \
  --weights data/logs/detectron2/model_best.pth \
  --input data/unlabeled_images \
  --output results/inference/model_detectron2
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from detectron2.data.detection_utils import read_image

sys.path.append(str(Path(Path(__file__).absolute().parent.parent, 'bin')))
from shrubs_detectron import ScaleMatchedPredictor, build_inference_cfg

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
OUTPUT_SUFFIXES = ("_shrubs", "_shrub_confidence")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Segment shrubs in RGB images with a trained Detectron2 Mask R-CNN checkpoint."
    )
    parser.add_argument("input", type=Path, help="Image file or folder of images.")
    parser.add_argument("--weights", type=Path, required=True, help="Trained Detectron2 .pth checkpoint.")
    parser.add_argument("--output", type=Path, required=True, help="Output directory.")
    parser.add_argument(
        "--config", default="COCO-InstanceSegmentation/mask_rcnn_R_101_C4_3x.yaml",
        help="Detectron2 model-zoo config name or a local YAML path (must match the weights).",
    )
    parser.add_argument("--num-classes", type=int, default=1)
    parser.add_argument("--confidence", type=float, default=0.5, help="Score threshold to keep a prediction.")
    parser.add_argument("--device", choices=("cuda", "cpu"), default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def find_images(input_path: Path) -> list[Path]:
    """Resolve a single image or the top-level images in a folder."""
    if input_path.is_file():
        if input_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"Input file is not a supported image: {input_path}")
        return [input_path]

    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    images = sorted(
        path for path in input_path.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
        and not path.stem.endswith(OUTPUT_SUFFIXES)
    )
    if not images:
        raise FileNotFoundError(f"No images found in {input_path}")
    return images


def predict_image(
    image_path: Path,
    output_folder: Path,
    predictor: ScaleMatchedPredictor,
    confidence: float,
    overwrite: bool,
) -> dict[str, int]:
    """Segment one image and write its mask, confidence, and instance outputs."""
    mask_path = output_folder / f"{image_path.stem}_shrubs.png"
    confidence_path = output_folder / f"{image_path.stem}_shrub_confidence.png"
    csv_path = output_folder / f"{image_path.stem}_shrub_instances.csv"
    if not overwrite and all(path.exists() for path in (mask_path, confidence_path, csv_path)):
        print(f"Skipping {image_path.name}: outputs already exist.")
        return {"instances": 0, "skipped": 1}

    image = read_image(str(image_path), format=predictor.input_format)
    result = predictor(image)

    keep = result["scores"] >= confidence
    masks = result["masks"][..., keep]
    scores = result["scores"][keep]
    boxes = result["boxes"][keep]

    union = np.any(masks, axis=-1)
    pixel_confidence = np.zeros(union.shape, dtype=np.float32)
    for instance_mask, score in zip(np.moveaxis(masks, -1, 0), scores):
        pixel_confidence[instance_mask] = np.maximum(pixel_confidence[instance_mask], score)

    Image.fromarray((union.astype(np.uint8) * 255)).save(mask_path)
    Image.fromarray((pixel_confidence * 255).astype(np.uint8)).save(confidence_path)

    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=[
            "instance_id", "score", "xmin", "ymin", "xmax", "ymax",
        ])
        writer.writeheader()
        for instance_id, (score, box) in enumerate(zip(scores, boxes), start=1):
            x0, y0, x1, y1 = box.tolist()
            writer.writerow({
                "instance_id": instance_id,
                "score": f"{float(score):.6f}",
                "xmin": f"{x0:.2f}", "ymin": f"{y0:.2f}",
                "xmax": f"{x1:.2f}", "ymax": f"{y1:.2f}",
            })

    return {"instances": int(scores.size), "skipped": 0}


def main() -> None:
    args = parse_args()
    if not args.weights.is_file():
        raise FileNotFoundError(f"Weights file not found: {args.weights}")

    images = find_images(args.input)
    args.output.mkdir(parents=True, exist_ok=True)

    predictor = ScaleMatchedPredictor(
        build_inference_cfg(
            args.weights, config=args.config, num_classes=args.num_classes, device=args.device,
        )
    )

    total_instances = total_skipped = 0
    for number, image_path in enumerate(images, start=1):
        print(f"Image {number}/{len(images)}: {image_path.name}")
        result = predict_image(image_path, args.output, predictor, args.confidence, args.overwrite)
        total_instances += result["instances"]
        total_skipped += result["skipped"]

    print(f"Finished: {total_instances} shrub instance(s); {total_skipped} image(s) skipped.")
    print(f"Results: {args.output}")


if __name__ == "__main__":
    main()
