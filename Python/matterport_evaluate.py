"""
Evaluate the winning Matterport Mask R-CNN checkpoint on the JuniperMapper
test splits.

Mirrors `detectron2_evaluate.py` and `sam3_evaluate.py`: it scores whole,
untiled images and shares the same pixel/instance-IoU/S-IoU metrics,
confidence sweep, and F1-vs-confidence plots (from `shrubs_common`) so the
three models are comparable on the same splits. Ground truths are read with
`ShrubDataset` from the `masks_points.json` files written by
`matterport_convert_annotations.py`.

Run it with `python Python/matterport_evaluate.py` inside the `mrcnn_tf2`
environment. The checkpoint metadata and output folder are set at the top of
this file; the test splits and evaluation settings are shared with the other
models through `shrubs_common.evaluation`.
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

sys.path.append(str(Path(Path(__file__).absolute().parent.parent, 'bin')))

# Catch a wrong environment
try:
    import mrcnn.model as modellib
    from shrubs_matterport import (
        BestModelShrubInferenceConfig, ShrubDataset, load_ground_truths, predict_dataset,
    )
except ImportError as e:
    raise ImportError(
        "Matterport Mask R-CNN dependencies are not installed in the active Python "
        "environment. Activate the 'mrcnn_tf2' environment before running this script."
    ) from e

from shrubs_common import TEST_DATASETS, EvaluationDataset, EvaluationSettings
from shrubs_common.plotting import run_evaluation_report

# Store the project base directory
ROOT = Path(__file__).resolve().parent.parent

MODEL_DIR = ROOT / "data" / "logs" / "matterport"
# Written by matterport_train.py; it stores the winning configuration and checkpoint
BEST_MODEL_METADATA = MODEL_DIR / "best_model_configuration.json"
# Set to a checkpoint path to override the one stored in the metadata
WEIGHTS = None

OUTPUT_DIR = ROOT / "results" / "evaluation" / "model_matterport"

# Shared with detectron2_evaluate.py and sam3_evaluate.py. Matterport keeps the
# background as class 0, so the shrub class is 1.
SETTINGS = replace(EvaluationSettings(), class_id=1)


def _annotation_file(spec: EvaluationDataset) -> Path:
    """Matterport's polygon file sits next to the split's `Images` folder."""
    return spec.image_root.parent / "masks_points.json"


def check_inputs() -> None:
    """Fail early if the model metadata or any test split is missing."""
    if not BEST_MODEL_METADATA.is_file():
        raise FileNotFoundError(
            f"Matterport model metadata not found at '{BEST_MODEL_METADATA}'. "
            "Run 'matterport_train.py' first, or copy 'best_model_configuration.json' "
            "and the run folder it points to into 'data/logs/matterport'."
        )

    missing = [
        f"  - {spec.name}: {path}"
        for spec in TEST_DATASETS
        for path in (spec.image_root, _annotation_file(spec))
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            "The following evaluation inputs are missing:\n" + "\n".join(missing)
            + "\nRun 'matterport_convert_annotations.py' to create the 'masks_points.json' files."
        )


def load_dataset(spec: EvaluationDataset) -> ShrubDataset:
    dataset = ShrubDataset()
    dataset.load_dataset(str(spec.image_root), str(_annotation_file(spec)))
    dataset.prepare()
    return dataset


def evaluate_dataset(model, spec: EvaluationDataset) -> None:
    dataset = load_dataset(spec)
    print(f"\nDataset: {spec.name} ({len(dataset.image_ids)} images)")
    ground_truths = load_ground_truths(dataset)
    predictions = predict_dataset(model, dataset, verbose=True)
    run_evaluation_report(ground_truths, predictions, OUTPUT_DIR / spec.name, SETTINGS)


def main() -> None:
    check_inputs()

    # Raises FileNotFoundError if the checkpoint stored in the metadata is missing.
    config = BestModelShrubInferenceConfig(BEST_MODEL_METADATA, weights_path=WEIGHTS)
    model = modellib.MaskRCNN(mode="inference", config=config, model_dir=str(MODEL_DIR))
    print(f"Loading weights from {config.best_model_path}")
    model.load_weights(config.best_model_path, by_name=True)

    for spec in TEST_DATASETS:
        evaluate_dataset(model, spec)


if __name__ == "__main__":
    main()
