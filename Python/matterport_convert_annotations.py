"""
Transform the JuniperMapper annotations into a valid format for Matterport
Mask R-CNN model.
"""
from pathlib import Path
import json

from shapely.geometry import Polygon, MultiPolygon
import rasterio
import geopandas as gpd

# Store the project base directory
ROOT = Path(__file__).resolve().parent.parent

JM_DIR = ROOT / "data" / "JuniperMapper"

if not JM_DIR.exists():
    raise FileNotFoundError(
        f"JuniperMapper database not found at '{JM_DIR.resolve()}'. "
        "Place a valid database copy inside the 'data' directory."
    )


def extract_masks(img_folder: Path, ann_folder: Path):
    """
    Extract shrub shapes from JuniperMapper dataset.
    
    Parameters
    ----------
    img_folder : Path
        Folder containing the annotated RGB images.

    ann_folder : Path
        Folder containing the shapefiles with the shrub
        masks related with RGB images in 'img_folder'.
    """
    with rasterio.open(img_folder) as src:
        img = src.read([1, 2, 3]) # RGB channels
        height, width = img.shape[1:]
        img_crs = src.crs
        img_transform = src.transform

    # Read shapefiles
    shapes = gpd.read_file(ann_folder)
    
    # Reproject shapefile to match raster CRS
    if shapes.crs != img_crs:
        shapes = shapes.to_crs(img_crs)
    
    # Save polygon coordinates in row,col format
    poly_pnts = []

    for geom in shapes.geometry:
        if geom is None:
            continue

        if isinstance(geom, (Polygon, MultiPolygon)):

            try:
                shapes = [geom] if isinstance(geom, Polygon) else geom.geoms
                for shape in shapes:
                    # shape.exterior.coords gives coordinates in the shapefile’s CRS (meters or lat/lon).
                    # cv2.fillPoly needs coordinates like (x_pixel, y_pixel) in image space.
                    xs, ys = shape.exterior.xy
                    # Convert world coordinates to pixel coordinates
                    coords = [~img_transform * (x, y) for x, y in zip(xs, ys)]
                    # Convert row, col coordinates to integers
                    coords = [[int(x), int(y)] for (x, y) in coords]
                    poly_pnts.append(coords)

            except Exception as e:
                continue

    return height, width, poly_pnts


def search_mask(img_path: Path, dataset_folder: Path):
    """Search if a valid mask exist for the current image."""
    parts = img_path.stem.split("_")

    if len(parts) < 2 or not parts[1].isdigit():
        raise ValueError(
            f"Invalid image filename: '{img_path.name}'. "
            "Expected a name such as 'Img_1.tif'."
        )

    # Extract image id, in Img_1 = 1
    img_id = int(parts[1])
    
    # Search annotation linked with the image
    ann_path = (
        dataset_folder
        / "Annotations"
        / "Shapefiles"
        / f"Pol_{img_id}.shp"
    )
    
    if not ann_path.is_file():
        return None

    return img_id, ann_path


def save_masks(images_dict: dict, out_pnts_path: Path):
    """Extract and save masks"""
    pnts_dict = {}

    for img_id, paths in images_dict.items():

        img, ann = paths.values()
        # Save the boundary points of the annotation file's geometries
        h, w, points_list = extract_masks(img, ann)

        pnts_dict[img_id] = {
            "pnts": points_list,
            "height": h,
            "width": w
        }

    with open(out_pnts_path, "w") as f:
        json.dump(pnts_dict, f)

    print(f"{len(pnts_dict.keys())} masks were written in {out_pnts_path}")


def create_mask_pnts(folder_path: Path):
    """
    Create the file with the mask coordinates
    
    It works for folders containing Images/Annotations subfolders.
    """
    images_folder = folder_path / "Images"
    if not images_folder.exists():
        raise FileNotFoundError(images_folder)
    
    for img_path in images_folder.rglob("*.tif"):
        result = search_mask(img_path, folder_path)

        if result is None:
            continue

        img_id, ann_path = result

        if img_id in images_dict:
            raise ValueError(
                f"Duplicate image ID {img_id}. "
                f"Existing image: '{images_dict[img_id]['img_path']}'. "
                f"Duplicate image: '{img_path}'."
            )

        images_dict[img_id] = {
            "img_path": str(img_path),
            "ann_path": str(ann_path)
        }
    
    out_pnts_path = Path(folder_path, "masks_points.json")
    save_masks(images_dict, out_pnts_path)


# Store the folder names inside JuniperMapper directory containing Images/Annotations folders
folder_names = {
    "Photo_Interpretation_Data": ["Test", "Train", "Val"],
    "Field_Work_Data": ["External_Val_Data"]
}

for folder_name, subfolders in folder_names.items():

    # Store images and their annotations (1 image per annotation file)
    images_dict: dict[int, dict[str, str]] = {}
    
    dataset_folder = JM_DIR / folder_name

    for subfolder_name in subfolders:
        create_mask_pnts(dataset_folder / subfolder_name)
