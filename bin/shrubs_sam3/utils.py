"""Utilities for tiled, georeferenced SAM 3 shrub inference."""

from __future__ import annotations

import csv
import os
from contextlib import nullcontext
from pathlib import Path
from typing import Callable

import numpy as np
import rasterio
import torch
from PIL import Image
from rasterio.windows import transform as window_transform

from shrubs_common import greedy_nms, iter_windows, write_window_masks

SUPPORTED_EXTENSIONS = {".tif", ".tiff"}
_NUMPY_SCALAR_GLOBALS = [
    np.core.multiarray.scalar,
    np.dtype,
    *(getattr(np.dtypes, name) for name in np.dtypes.__all__),
]


def parse_rgb_bands(value: str) -> tuple[int, int, int]:
    """Parse three one-based raster band numbers."""
    try:
        bands = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise ValueError("RGB bands must be three comma-separated integers.") from exc
    if len(bands) != 3 or any(band < 1 for band in bands):
        raise ValueError("RGB bands must contain exactly three positive integers.")
    return bands


def _to_rgb_uint8(data: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Percentile-stretch a three-band raster tile to an RGB uint8 array."""
    rgb = np.zeros(data.shape, dtype=np.uint8)
    for index, band in enumerate(data):
        samples = band[valid & np.isfinite(band)]
        if samples.size == 0:
            continue
        low, high = np.percentile(samples, (2, 98))
        if high <= low:
            high = low + 1.0
        stretched = np.clip((band.astype(np.float32) - low) / (high - low), 0, 1)
        stretched[~np.isfinite(stretched)] = 0
        rgb[index] = np.round(stretched * 255).astype(np.uint8)
    rgb[:, ~valid] = 0
    return np.moveaxis(rgb, 0, -1)


def sam3_autocast(device: str):
    """Return the autocast context SAM 3 inference must run under.

    SAM 3's weights/kernels expect bfloat16 activations on CUDA; without this,
    a bf16 weight meeting a float32 activation raises a dtype mismatch.
    """
    if device == "cuda" and torch.cuda.is_bf16_supported():
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()


def build_processor(confidence: float, checkpoint: Path | None = None):
    """Build SAM 3 and safely load either release or fine-tuned weights."""
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if checkpoint is None:
        model = build_sam3_image_model(device=device)
    else:
        # Release checkpoints use ``detector.*`` keys, whereas the image
        # fine-tuning trainer saves plain image-model keys. The upstream
        # builder currently drops the latter silently, so distinguish them.
        # Fine-tuned checkpoints store the best-epoch AP meters as numpy
        # scalars (see `training.py`), so allowlist numpy scalars and dtypes.
        with torch.serialization.safe_globals(_NUMPY_SCALAR_GLOBALS):
            payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        state_dict = payload.get("model", payload)
        state_dict = {
            key.removeprefix("module."): value for key, value in state_dict.items()
        }
        if any(key.startswith("detector.") for key in state_dict):
            model = build_sam3_image_model(
                device=device, checkpoint_path=str(checkpoint), load_from_HF=False
            )
        else:
            model = build_sam3_image_model(device="cpu", load_from_HF=False)
            incompatible = model.load_state_dict(state_dict, strict=False)
            matched = len(state_dict) - len(incompatible.unexpected_keys)
            if matched == 0:
                raise ValueError(
                    f"No weights in {checkpoint} match the SAM 3 image model."
                )
            model.to(device).eval()
    return Sam3Processor(model, device=device, confidence_threshold=confidence), device


def _tensor_to_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
        # NumPy has no native bfloat16 dtype. SAM 3 can return bfloat16 scores
        # and boxes when CUDA autocast is enabled, so promote those tensors
        # before transferring them to the CPU. Keep bool masks as bool.
        if value.dtype == torch.bfloat16:
            value = value.to(dtype=torch.float32)
        value = value.cpu().numpy()
    return np.asarray(value)


class TextPromptPredictor:
    """Adapt a text-prompted `Sam3Processor` to the shared Prediction format
    (`boxes`/`class_ids`/`scores`/`masks`, masks HxWxN) used by `shrubs_common`,
    so whole-image SAM 3 inference can reuse the same evaluation code as Detectron2.
    """

    input_format = "RGB"

    def __init__(self, processor, prompt: str) -> None:
        self.processor = processor
        self.prompt = prompt

    def __call__(self, image: np.ndarray) -> dict[str, np.ndarray]:
        height, width = image.shape[:2]
        state = self.processor.set_image(Image.fromarray(image))
        result = self.processor.set_text_prompt(prompt=self.prompt, state=state)
        masks = _tensor_to_numpy(result["masks"])
        scores = _tensor_to_numpy(result["scores"]).reshape(-1).astype(np.float32)
        boxes = _tensor_to_numpy(result["boxes"]).astype(np.float32).reshape(-1, 4)
        masks = (
            masks.reshape((-1, height, width)).astype(bool)
            if masks.size else np.zeros((0, height, width), dtype=bool)
        )
        return {
            "boxes": boxes,
            "class_ids": np.zeros(len(scores), dtype=np.int64),
            "scores": scores,
            "masks": np.moveaxis(masks, 0, -1),
        }


def process_image(
    image_path: Path,
    output_folder: Path,
    processor,
    prompt: str,
    rgb_bands: tuple[int, int, int],
    tile_size: int,
    overlap: int,
    min_area: int = 0,
    nms_iou: float = 1.0,
    overwrite: bool = False,
    progress: Callable[[str], None] = print,
) -> dict[str, int | Path]:
    """Segment one raster and write union, confidence, and instance outputs.

    `min_area` drops instance masks smaller than that many pixels, and
    `nms_iou` (a threshold in (0, 1]) suppresses lower-score masks that
    overlap a kept one by at least that much, within the same tile. Both are
    no-ops at their defaults, so existing zero-shot output is unchanged unless
    a caller opts in.
    """
    mask_path = output_folder / f"{image_path.stem}_shrubs.tif"
    confidence_path = output_folder / f"{image_path.stem}_shrub_confidence.tif"
    csv_path = output_folder / f"{image_path.stem}_shrub_instances.csv"
    outputs = (mask_path, confidence_path, csv_path)
    if not overwrite and all(path.exists() for path in outputs):
        progress(f"Skipping {image_path.name}: outputs already exist.")
        return {"image": image_path, "instances": 0, "skipped": 1}

    temp_mask = mask_path.with_suffix(".partial.tif")
    temp_confidence = confidence_path.with_suffix(".partial.tif")
    temp_csv = csv_path.with_suffix(".partial.csv")

    try:
        with rasterio.open(image_path) as source:
            if max(rgb_bands) > source.count:
                raise ValueError(
                    f"{image_path.name} has {source.count} band(s), but RGB band "
                    f"{max(rgb_bands)} was requested."
                )

            mask_profile = source.profile.copy()
            mask_profile.update(
                driver="GTiff", count=1, dtype="uint8", nodata=0,
                compress="deflate", predictor=1, BIGTIFF="IF_SAFER",
            )
            confidence_profile = mask_profile.copy()
            confidence_profile.update(dtype="float32", predictor=3)

            windows = list(iter_windows(source.width, source.height, tile_size, overlap))
            instance_count = 0
            # ``w+`` is required because overlapping tiles are merged by
            # reading the values already written to each output window.
            with rasterio.open(temp_mask, "w+", **mask_profile) as mask_dst, \
                    rasterio.open(temp_confidence, "w+", **confidence_profile) as conf_dst, \
                    temp_csv.open("w", newline="", encoding="utf-8") as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=[
                    "instance_id", "score", "tile_row", "tile_col",
                    "xmin", "ymin", "xmax", "ymax",
                ])
                writer.writeheader()
                tags = {
                    "SAM3_PROMPT": prompt,
                    "SAM3_CONFIDENCE": str(processor.confidence_threshold),
                    "RGB_BANDS": ",".join(map(str, rgb_bands)),
                }
                mask_dst.update_tags(**tags)
                conf_dst.update_tags(**tags)

                for tile_number, window in enumerate(windows, start=1):
                    progress(f"  Tile {tile_number}/{len(windows)}")
                    tile = source.read(rgb_bands, window=window, masked=True)
                    array = tile.astype(np.float32).filled(np.nan)
                    valid = ~np.any(np.ma.getmaskarray(tile), axis=0)
                    valid &= np.all(np.isfinite(array), axis=0)
                    if not valid.any():
                        continue

                    state = processor.set_image(Image.fromarray(_to_rgb_uint8(array, valid)))
                    result = processor.set_text_prompt(prompt=prompt, state=state)
                    masks = _tensor_to_numpy(result["masks"])
                    scores = _tensor_to_numpy(result["scores"]).reshape(-1)
                    boxes = _tensor_to_numpy(result["boxes"])
                    if masks.size == 0:
                        continue
                    masks = masks.reshape((-1, int(window.height), int(window.width)))

                    keep = masks.reshape(len(masks), -1).sum(axis=1) >= min_area
                    if nms_iou < 1.0 and keep.any():
                        keep[keep] &= greedy_nms(np.moveaxis(masks[keep], 0, -1), scores[keep], nms_iou)
                    masks, scores, boxes = masks[keep], scores[keep], boxes[keep]
                    if masks.size == 0:
                        continue

                    # Real per-instance polygons (not the boxes above), so this
                    # window's shrubs can be merged the same generic way as
                    # Detectron2/Matterport (see bin/shrubs_common/polygons.py).
                    write_window_masks(
                        output_folder / f"{image_path.stem}_r{int(window.row_off)}_c{int(window.col_off)}_masks.json",
                        np.moveaxis(masks, 0, -1) & valid[..., None],
                        scores,
                        np.zeros(len(scores), dtype=np.int64),
                        window_transform(window, source.transform),
                        source.crs,
                    )

                    old_mask = mask_dst.read(1, window=window)
                    old_confidence = conf_dst.read(1, window=window)
                    union = np.any(masks, axis=0) & valid
                    confidence = np.zeros(union.shape, dtype=np.float32)
                    for instance_mask, score, box in zip(masks, scores, boxes):
                        instance_count += 1
                        confidence[instance_mask & valid] = np.maximum(
                            confidence[instance_mask & valid], float(score)
                        )
                        x0, y0, x1, y1 = box.tolist()
                        writer.writerow({
                            "instance_id": instance_count,
                            "score": f"{float(score):.6f}",
                            "tile_row": int(window.row_off),
                            "tile_col": int(window.col_off),
                            "xmin": f"{x0 + window.col_off:.2f}",
                            "ymin": f"{y0 + window.row_off:.2f}",
                            "xmax": f"{x1 + window.col_off:.2f}",
                            "ymax": f"{y1 + window.row_off:.2f}",
                        })
                    mask_dst.write(
                        np.maximum(old_mask, union.astype(np.uint8)), 1, window=window
                    )
                    conf_dst.write(
                        np.maximum(old_confidence, confidence), 1, window=window
                    )

        for temporary, final in zip(
            (temp_mask, temp_confidence, temp_csv), outputs
        ):
            os.replace(temporary, final)
        return {"image": image_path, "instances": instance_count, "skipped": 0}
    except Exception:
        for path in (temp_mask, temp_confidence, temp_csv):
            path.unlink(missing_ok=True)
        raise


def _find_images(input_path: Path) -> list[Path]:
    """Resolve a single GeoTIFF or the top-level GeoTIFFs in a folder."""
    if input_path.is_file():
        if input_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"Input file is not a GeoTIFF: {input_path}")
        return [input_path]

    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    images = sorted(
        path for path in input_path.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
        and not path.stem.endswith(("_shrubs", "_shrub_confidence"))
        and ".partial" not in path.stem
    )
    if not images:
        raise FileNotFoundError(f"No GeoTIFF images found in {input_path}")
    return images


def process_input(
    input_path: Path,
    output_folder: Path,
    prompt: str,
    confidence: float,
    rgb_bands: tuple[int, int, int],
    tile_size: int,
    overlap: int,
    min_area: int = 0,
    nms_iou: float = 1.0,
    checkpoint: Path | None = None,
    overwrite: bool = False,
    progress: Callable[[str], None] = print,
) -> list[dict[str, int | Path]]:
    """Load SAM 3 once, then process one GeoTIFF or a folder of GeoTIFFs."""
    images = _find_images(input_path)
    output_folder.mkdir(parents=True, exist_ok=True)

    progress("Loading SAM 3 (the first run may download the gated checkpoint)...")
    processor, device = build_processor(confidence, checkpoint)
    progress(f"Using {device.upper()} for inference.")
    if device == "cpu":
        progress("Warning: SAM 3 CPU inference is supported but will be very slow.")

    results = []
    with sam3_autocast(device):
        for number, image_path in enumerate(images, start=1):
            progress(f"Image {number}/{len(images)}: {image_path.name}")
            results.append(process_image(
                image_path, output_folder, processor, prompt, rgb_bands,
                tile_size, overlap, min_area, nms_iou, overwrite, progress,
            ))
    return results
