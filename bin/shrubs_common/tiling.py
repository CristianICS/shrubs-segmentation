"""Shared internal-tiling utilities for georeferenced deployment scripts.

Every deployment script (Detectron2, Matterport, SAM 3) tiles its source
orthomosaic internally into optionally overlapping windows instead of
consuming a folder of pre-cropped tile GeoTIFFs. Overlap gives each model
more than one look at whatever lands on a tile seam, which
`refine_detections.py`'s cross-window grouping can then reconcile into one
shrub instead of two clipped halves.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import rasterio
from rasterio.windows import Window


def check_tiling_inputs(
    orthomosaic: Path,
    tile_size: int,
    overlap: int,
    rgb_bands: Sequence[int] = (1, 2, 3),
    alpha_band: int | None = 4,
) -> None:
    """Fail early, with informative messages, on a bad orthomosaic or tiling options.

    Shared by the scripts that tile an orthomosaic (`tile_orthomosaic.py`,
    `detectron2_deploy.py`, `sam3_deploy.py`), so a missing raster, an impossible window layout or
    a band the raster doesn't have is reported before any model is built or any
    window is written. Messages name the command-line flags those scripts share.
    `alpha_band` of 0 or None disables the alpha-band check.
    """
    orthomosaic = Path(orthomosaic)
    if not orthomosaic.is_file():
        raise FileNotFoundError(
            f"Orthomosaic not found at '{orthomosaic}'. Place the RGB GeoTIFF "
            "there, or pass its location with --orthomosaic."
        )

    if tile_size <= 0:
        raise ValueError(f"--tile-size must be positive, got {tile_size}.")
    if not 0 <= overlap < tile_size:
        raise ValueError(
            f"--overlap must be in [0, tile size), got {overlap} for a "
            f"tile size of {tile_size}."
        )

    with rasterio.open(orthomosaic) as source:
        band_count = source.count
    requested = [*rgb_bands, *([alpha_band] if alpha_band else [])]
    invalid = [band for band in requested if not 1 <= band <= band_count]
    if invalid:
        raise ValueError(
            f"Band(s) {invalid} do not exist in '{orthomosaic.name}', which has "
            f"{band_count} band(s). Adjust --rgb-bands, or set --alpha-band 0 if "
            "the raster has no alpha band."
        )


def starts(length: int, tile_size: int, overlap: int) -> list[int]:
    """Return the 0-based start offsets covering `length` with overlapping tiles."""
    if length <= tile_size:
        return [0]
    stride = tile_size - overlap
    values = list(range(0, length - tile_size + 1, stride))
    last = length - tile_size
    if values[-1] != last:
        values.append(last)
    return values


def iter_windows(width: int, height: int, tile_size: int, overlap: int) -> Iterable[Window]:
    """Yield full-coverage, optionally overlapping windows, including image edges."""
    if tile_size < 1:
        raise ValueError("Tile size must be positive.")
    if overlap < 0 or overlap >= tile_size:
        raise ValueError("Overlap must be at least 0 and smaller than tile size.")
    for row in starts(height, tile_size, overlap):
        for col in starts(width, tile_size, overlap):
            yield Window(col, row, min(tile_size, width - col), min(tile_size, height - row))


def read_rgb_window(
    source: rasterio.DatasetReader,
    window: Window,
    bands: tuple[int, int, int] = (1, 2, 3),
    alpha_band: int | None = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Read one window's RGB bands plus a validity mask.

    Validity comes from `alpha_band` (nonzero = valid pixel) when the source
    has one, since `data/rasters/*.tif` carries transparency in an alpha band
    rather than declaring a nodata value. Falls back to "everything valid"
    when `alpha_band` is None or out of range.
    """
    rgb = np.moveaxis(source.read(list(bands), window=window), 0, -1)
    if alpha_band is not None and alpha_band <= source.count:
        valid = source.read(alpha_band, window=window) > 0
    else:
        valid = np.ones(rgb.shape[:2], dtype=bool)
    return rgb, valid
