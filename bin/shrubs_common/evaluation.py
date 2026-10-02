"""
Test datasets and evaluation settings shared by `detectron2_evaluate.py` and
`sam3_evaluate.py`.

Both scripts read the same JuniperMapper splits and the same settings from
here, so the two models are always scored on identical inputs. Each script
only adds its own model checkpoint and output folder.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .metrics import METRIC_DEFINITIONS

# Store the project base directory (bin/shrubs_common -> project root)
ROOT = Path(__file__).resolve().parents[2]

JM_DIR = ROOT / "data" / "JuniperMapper"


@dataclass(frozen=True)
class EvaluationDataset:
    """One COCO-format test split: its name, annotation JSON and image folder."""
    name: str
    json_file: Path
    image_root: Path


TEST_DATASETS = (
    EvaluationDataset(
        name="shrubs_test_pi",
        json_file=JM_DIR / "Photo_Interpretation_Data" / "Test" / "Annotations" / "Test.json",
        image_root=JM_DIR / "Photo_Interpretation_Data" / "Test" / "Images",
    ),
    EvaluationDataset(
        name="shrubs_test_fw",
        json_file=JM_DIR / "Field_Work_Data" / "External_Val_Data" / "Annotations" / "Val.json",
        image_root=JM_DIR / "Field_Work_Data" / "External_Val_Data" / "Images",
    ),
)


@dataclass(frozen=True)
class EvaluationSettings:
    """
    Evaluation settings passed to `run_evaluation_report`.

    Field names match the attributes that `shrubs_common.plotting`
    reads, so an instance can replace the former argparse namespace.

    Attributes
    ----------
    class_id : int
        Contiguous class ID to evaluate. One-class COCO datasets
        (Shrub) normally use 0.

    confidence_thresholds : tuple of float or None
        Prediction-score thresholds swept when building the CSV files
        and F1 curves. If None, uses 0.00, 0.05, ..., 1.00.

    evaluation_thresholds : tuple of float
        Matching thresholds, each in (0, 1]. Each value is used as both
        the IoU and the S-IoU threshold, producing one IoU CSV, one
        S-IoU CSV and one F1 figure per value.

    pixel_confidence : float
        Prediction-score threshold for the standalone micro-averaged
        pixel precision, recall and F1.

    iou_metric_definition : {"standard", "paper"}
        How mask-IoU instances are counted. "standard" uses COCO-style
        one-to-one matching. "paper" uses the independent best matches
        of Khaldi et al. (2024), with recall = TP / (TP + FN). Results
        go to separate files, so the two definitions never overwrite
        each other. S-IoU does not depend on this setting.

    minimum_intersection_pixels : int
        Minimum number of overlapping pixels for a prediction and a
        ground-truth mask to count as S-IoU match candidates.

    show_plots : bool
        If True, also opens each saved F1 figure in a window. Figures
        are always saved to disk.
    """
    class_id: int = 0
    confidence_thresholds: tuple[float, ...] | None = None
    evaluation_thresholds: tuple[float, ...] = (0.50, 0.75)
    pixel_confidence: float = 0.50
    iou_metric_definition: str = "paper"
    minimum_intersection_pixels: int = 1
    show_plots: bool = False

    def __post_init__(self) -> None:
        if self.iou_metric_definition not in METRIC_DEFINITIONS:
            raise ValueError(
                f"Invalid iou_metric_definition '{self.iou_metric_definition}'. "
                f"Choose one of: {', '.join(METRIC_DEFINITIONS)}."
            )


def check_dataset_inputs(datasets: tuple[EvaluationDataset, ...] = TEST_DATASETS) -> None:
    """Fail early, listing every missing annotation file or image folder at once."""
    if not JM_DIR.exists():
        raise FileNotFoundError(
            f"JuniperMapper database not found at '{JM_DIR}'. "
            "Place a valid database copy inside the 'data' directory."
        )

    missing = [
        f"  - {dataset.name}: {path}"
        for dataset in datasets
        for path in (dataset.json_file, dataset.image_root)
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            "The following evaluation inputs are missing:\n" + "\n".join(missing)
        )
