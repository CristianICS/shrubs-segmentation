"""Dataset preparation and partial-freezing support for SAM 3 experiments."""

from __future__ import annotations

import functools
import json
import logging
import random
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist


def _polygon_bbox(polygon: list[float]) -> list[float]:
    xs = polygon[0::2]
    ys = polygon[1::2]
    xmin, ymin = min(xs), min(ys)
    return [xmin, ymin, max(xs) - xmin, max(ys) - ymin]


def _polygon_area(polygon: list[float]) -> float:
    points = list(zip(polygon[0::2], polygon[1::2]))
    return abs(sum(
        x0 * y1 - x1 * y0
        for (x0, y0), (x1, y1) in zip(points, points[1:] + points[:1])
    )) / 2.0


# JuniperMapper's Photo Interpretation split has a couple of digitizing slips
# (a 1x1 pixel, area-0.5 "shrub") with nothing between them and the real
# small-shrub population, which starts at area >= ~10 px^2. Left in, such a
# polygon can resize/augment into an all-empty mask, which crashes SAM 3's
# Hungarian matcher with a NaN/Inf cost partway through training.
MIN_INSTANCE_AREA_PX = 4.0


def prepare_juniper_coco(
    source_json: Path,
    output_json: Path,
    *,
    shots: int | None = None,
    seed: int = 42,
    prompt: str = "shrub",
) -> dict[str, Any]:
    """Convert JuniperMapper's grouped polygons to SAM 3 COCO instances."""
    with source_json.open(encoding="utf-8") as source:
        original = json.load(source)

    instances: list[dict[str, Any]] = []
    for annotation in original.get("annotations", []):
        segmentation = annotation.get("segmentation")
        polygons = segmentation if isinstance(segmentation, list) else []
        for polygon in polygons:
            if not isinstance(polygon, list) or len(polygon) < 6 or len(polygon) % 2:
                continue
            area = _polygon_area(polygon)
            if area < MIN_INSTANCE_AREA_PX:
                continue
            instances.append({
                "image_id": annotation["image_id"],
                "category_id": 1,
                "segmentation": [polygon],
                "area": area,
                "bbox": _polygon_bbox(polygon),
                "iscrowd": 0,
            })

    if shots is not None:
        if shots < 1:
            raise ValueError("shots must be positive")
        if shots > len(instances):
            raise ValueError(
                f"Requested {shots} shots, but only {len(instances)} instances exist."
            )
        instances = random.Random(seed).sample(instances, shots)

    used_image_ids = {annotation["image_id"] for annotation in instances}
    images = [
        image for image in original.get("images", [])
        if image["id"] in used_image_ids
    ]
    for annotation_id, annotation in enumerate(instances, start=1):
        annotation["id"] = annotation_id

    result = {
        "info": {
            "description": "JuniperMapper shrubs prepared for SAM 3",
            "source": str(source_json),
            "shots": shots,
            "seed": seed,
        },
        "licenses": original.get("licenses", []),
        "images": images,
        "annotations": instances,
        "categories": [{"id": 1, "name": prompt, "supercategory": "vegetation"}],
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as destination:
        json.dump(result, destination, indent=2)
    return result


try:
    from sam3.train.trainer import Trainer as _Sam3Trainer
except ImportError:  # Keep inference and preparation importable without train extras.
    _Sam3Trainer = object


def _wrap_forward_no_grad(module: torch.nn.Module) -> None:
    """Force `module.forward` to always run under `torch.no_grad()`.

    Freezing a submodule's parameters (`requires_grad_(False)`) stops it from
    accumulating gradients, but does not disable the ambient autograd context
    its forward pass runs under -- the rest of the model still needs grad
    enabled to backprop into the trainable head, and a frozen submodule
    inherits that same context unless told otherwise. SAM 3's ViT MLP
    (`sam3.model.vitdet.Mlp`) calls a fused kernel
    (`sam3.perflib.fused.addmm_act`) that hard-asserts grad is globally
    disabled, so without this wrapper a frozen vision backbone still crashes
    with `ValueError("Expected grad to be disabled.")` the moment training
    (not just inference) runs a forward pass through it.
    """
    original_forward = module.forward

    @functools.wraps(original_forward)
    def forward_no_grad(*args, **kwargs):
        with torch.no_grad():
            return original_forward(*args, **kwargs)

    module.forward = forward_no_grad


class FrozenBackboneTrainer(_Sam3Trainer):
    """SAM 3 trainer that freezes both pretrained backbones before AdamW."""

    def _setup_components(self) -> None:
        # `super()._setup_components()` builds `self.model` and immediately
        # logs a "Trainable parameters" summary of it, before this method
        # gets a chance to freeze anything -- that summary always shows 100%
        # trainable and is not a sign the freeze below failed to run.
        super()._setup_components()
        frozen = 0
        total = 0
        for name, parameter in self.model.named_parameters():
            total += parameter.numel()
            if name.startswith((
                "backbone.vision_backbone.",
                "backbone.language_backbone.",
            )):
                parameter.requires_grad_(False)
                frozen += parameter.numel()
        if frozen == 0:
            raise RuntimeError(
                "No SAM 3 vision/language backbone parameters matched for freezing."
            )
        logging.info(
            "FrozenBackboneTrainer: froze %d/%d parameters (%.1f%%); "
            "%d trainable.",
            frozen, total, 100 * frozen / total, total - frozen,
        )
        _wrap_forward_no_grad(self.model.backbone.vision_backbone)
        _wrap_forward_no_grad(self.model.backbone.language_backbone)

    def _setup_ddp_distributed_training(self, distributed_conf, accelerator) -> None:
        # sam3_domain_adapt.py always trains as a single local process
        # (world_size=1), where DDP wrapping buys nothing but does add a
        # failure mode of its own: with find_unused_parameters=True, DDP's
        # gradient-bucket rebuild requires the SAME set of parameters to
        # participate in the loss on every iteration after the first. SAM 3's
        # auxiliary/one-to-many matching heads don't guarantee that -- which
        # parameters end up used legitimately varies batch to batch -- so DDP
        # crashes with "Expected to have finished reduction in the prior
        # iteration" a few steps in. A plain (non-DDP) forward/backward pass
        # has no such requirement, and the rest of the trainer already
        # tolerates a non-DDP-wrapped `self.model` (see `unwrap_ddp_if_wrapped`
        # in checkpoint saving).
        if dist.is_initialized() and dist.get_world_size() == 1:
            logging.info(
                "FrozenBackboneTrainer: world_size=1, skipping DDP wrapping."
            )
            return
        super()._setup_ddp_distributed_training(distributed_conf, accelerator)

    def _step(self, batch, model, phase):
        """Turn a batch whose Hungarian matching cost is NaN/Inf into a true
        no-op step instead of crashing the whole run.

        JuniperMapper's 448x448 tiles include some very small shrub instances;
        after random resize and box-noise augmentation, one occasionally
        degenerates into a near-zero-area box or empty mask, and SAM 3's
        matcher raises `ValueError: matrix contains invalid numeric entries`
        from inside the forward pass.

        The upstream trainer's own non-finite-loss handling (`_run_step`)
        looks like the natural place to route this, but it returns before
        ever calling `scaler.scale(loss).backward()` -- and `train_epoch`
        unconditionally calls `scaler.step(optimizer)` afterwards regardless,
        which then raises "No inf checks were recorded for this optimizer."
        for *any* skipped batch under AMP, not just this one. So instead of
        skipping the step, this returns a genuine zero loss wired to every
        trainable parameter: its backward pass produces exactly zero
        gradients (a real no-op update) while still driving the scaler
        through its normal, expected per-step bookkeeping.
        """
        # `super()._step` consumes `batch`'s one key via `popitem()` before it
        # can fail, so the meter key it would have used (e.g. "shrubs_train")
        # must be captured here first -- the per-dataset loss meter this
        # trainer already tracks is only registered under that exact key, not
        # any name we might invent in the except block below.
        key = next(iter(batch))
        try:
            return super()._step(batch, model, phase)
        except ValueError as exc:
            if "invalid numeric entries" not in str(exc):
                raise
            logging.error(
                "FrozenBackboneTrainer: skipping a batch with a non-finite matching cost (%s).", exc
            )
            # A parameter already holding inf/nan (itself a sign of real
            # instability, not just this one bad batch) would otherwise turn
            # `inf * 0.0` into nan here too -- nan_to_num first guarantees a
            # true, finite zero regardless.
            zero_loss = sum(
                (torch.nan_to_num(p) * 0.0).sum() for p in model.parameters() if p.requires_grad
            )
            return {f"Losses/{phase}_{key}_loss": zero_loss}, 1, {}
