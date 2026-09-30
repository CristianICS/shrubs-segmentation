"""SAM 3 experiment utilities for georeferenced shrub segmentation."""

import os

# On Windows, conda-forge's rasterio/GDAL links LLVM OpenMP (libomp.dll) while
# the pip-installed PyTorch wheel bundles Intel's OpenMP (libiomp5md.dll).
# Loading both in one process aborts with "OMP: Error #15" unless one runtime
# is allowed to coexist. Neither library runs its thread pool concurrently
# with the other in this codebase, so allowing the duplicate is safe here.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from .utils import (
    TextPromptPredictor, build_processor, parse_rgb_bands, process_input, sam3_autocast,
)
from .training import FrozenBackboneTrainer, prepare_juniper_coco

__all__ = [
    "FrozenBackboneTrainer",
    "TextPromptPredictor",
    "build_processor",
    "parse_rgb_bands",
    "prepare_juniper_coco",
    "process_input",
    "sam3_autocast",
]
