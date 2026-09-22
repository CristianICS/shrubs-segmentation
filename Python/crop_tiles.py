"""
Crop the original orthomosaic into smaller tiles to create the shrub map.
"""
from pathlib import Path
import rasterio
from rasterio.windows import Window

# Store the project base directory
ROOT = Path(__file__).resolve().parent.parent

IMG_PATH = ROOT / "data" / "rasters" / "orthomosaic_clip.tif"


def tilling(
    img_path: Path,
    target_width: int = 448,
    target_height: int = 448,
    compression: bool = False
):
    """
    Cropping the original orthoimage.

    If the image source has rotation/shear (non-north-up geotransform),
    the `tile_gt` math below still handles it correctly because it uses `Window`.
    """
    # Create a temporal folder to store the chunks
    tiles_temp_folder = img_path.parent / "deployment_tiles"
    tiles_temp_folder.mkdir(exist_ok=True)

    # Store the tiles metadata in a dictionary
    tiles_meta = {}

    # Optional GeoTIFF creation tweaks
    GTIFF_CREATION_KWARGS = {
        "driver": "GTiff",
        "tiled": True,
        "compress": "deflate",
        "predictor": 2,
        "BIGTIFF": "IF_SAFER"
    }

    with rasterio.open(img_path) as src:
        img_width = src.width
        img_height = src.height

        base_profile = src.profile.copy()
        base_tags = src.tags()
        
        # Per-band nodata (rasterio has .nodata and .nodatavals);
        # keep per-band where possible
        nodatavals = src.nodatavals

        tile_id = 1
        for y in range(0, img_height, target_height):
            for x in range(0, img_width, target_width):
                w = min(target_width, img_width - x)
                h = min(target_height, img_height - y)

                window = Window(col_off=x, row_off=y, width=w, height=h)

                # Read tile. Keep original band count; shape is (count, h, w)
                chunk = src.read(window=window)

                out_name = f"tile_{y}_{x}.tif"
                tile_path = tiles_temp_folder / out_name

                tile_profile = base_profile.copy()
                tile_profile.update(
                    {
                        "width": chunk.shape[2],
                        "height": chunk.shape[1],
                        "transform": src.window_transform(window),
                    }
                )
                if compression:
                    tile_profile.update({**GTIFF_CREATION_KWARGS})

                # If source uses internal tiling, rasterio/GDAL may require
                # block sizes not to exceed tile dimensions.
                # Make block sizes <= output dims.
                if tile_profile.get("tiled", False):
                    bsx = min(256, tile_profile["width"])
                    bsy = min(256, tile_profile["height"])
                    tile_profile.update({"blockxsize": bsx, "blockysize": bsy})

                with rasterio.open(tile_path, "w", **tile_profile) as dst:
                    dst.write(chunk)

                    # Preserve dataset-level tags (optional but often useful)
                    if base_tags:
                        dst.update_tags(**base_tags)

                    # Preserve per-band nodata values where present
                    for bidx, nd in enumerate(nodatavals, start=1):
                        if nd is not None:
                            dst.update_tags(bidx, nodata=nd)  # metadata tag
                            dst.write_nodata(nd, bidx)

                    # Preserve per-band tags too
                    for bidx in range(1, src.count + 1):
                        band_tags = src.tags(bidx)
                        if band_tags:
                            dst.update_tags(bidx, **band_tags)

                tiles_meta[tile_id] = {"y": y, "x": x, "path": tile_path}
                tile_id += 1
    
    return tiles_meta

tilling(IMG_PATH)
