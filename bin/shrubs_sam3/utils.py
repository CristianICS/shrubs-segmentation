"""SAM 3 model loading and text-prompted inference utilities.

The tiled, georeferenced deployment loop lives in `Python/sam3_deploy.py`,
which shares it with `detectron2_deploy.py` through `shrubs_common`.
"""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from PIL import Image

_NUMPY_SCALAR_GLOBALS = [
    np.core.multiarray.scalar,
    np.dtype,
    *(getattr(np.dtypes, name) for name in np.dtypes.__all__),
]


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
