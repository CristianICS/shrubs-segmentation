"""
Select and train the Matterport Mask R-CNN model on the JuniperMapper
Photo Interpretation data.

Runs a sequential design instead of a full hyperparameter grid, which would be
costly and more likely to pick a configuration that only looks good by chance:

* Phase A, training method (lr 0.001, ResNet-101):
    A1 heads only, no augmentation;
    A2 staged (heads -> 4+ -> all), no augmentation;
    A3 staged, geometric augmentation.
* Phase B, learning rate of the Phase A winner: B1 0.0005, B2 0.0025.
* Phase C, backbone: C1 ResNet-50 with the Phase A + B winner.

Each run keeps the checkpoint with the highest validation mask AP50 (ties
broken by mask mAP50:95), the same criterion used for Detectron2 and SAM 3,
and the winner of each phase is chosen with that criterion too
(`shrubs_matterport.select_best_experiment`). The final winner is written to
`data/logs/matterport/best_model_configuration.json`, which
`matterport_evaluate.py` and `matterport_deploy.py` read.

Experiments that already have a completed run in `data/logs/matterport` are
reused instead of retrained, so the script can be restarted after an
interruption. Delete (or move) a run folder to retrain that experiment.

Run it with `python Python/matterport_train.py` inside the `mrcnn_tf2`
environment, after `matterport_convert_annotations.py`.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

sys.path.append(str(Path(Path(__file__).absolute().parent.parent, 'bin')))

# Catch a wrong environment
try:
    from imgaug import augmenters as iaa
    from mrcnn import utils
    from shrubs_matterport import (
        BaselineShrubConfig,
        ShrubDataset,
        TrainingExperiment,
        scale_training_schedule,
        select_best_experiment,
    )
except ImportError as e:
    raise ImportError(
        "Matterport Mask R-CNN dependencies are not installed in the active Python "
        "environment. Activate the 'mrcnn_tf2' environment before running this script."
    ) from e

# Store the project base directory
ROOT = Path(__file__).resolve().parent.parent

TRAIN_DIR = ROOT / "data" / "JuniperMapper" / "Photo_Interpretation_Data"
EXPERIMENT_ROOT = ROOT / "data" / "logs" / "matterport"
COCO_MODEL_PATH = EXPERIMENT_ROOT / "mask_rcnn_coco.h5"
BEST_MODEL_METADATA = EXPERIMENT_ROOT / "best_model_configuration.json"

DATASET_ID = "JunipperMapper_train_val_partition"
SCREENING_SEED = 20260803

# Phase A schedules. In Matterport the epochs are cumulative, and the model is
# not reinitialized between stages.
HEADS_ONLY_STAGES = [
    {"name": "heads", "layers": "heads", "initial_epoch": 0, "final_epoch": 10, "learning_rate": 1e-3},
]
STAGED_STAGES = [
    {"name": "heads", "layers": "heads", "initial_epoch": 0, "final_epoch": 10, "learning_rate": 1e-3},
    {"name": "resnet_stage_4_and_above", "layers": "4+", "initial_epoch": 10, "final_epoch": 25, "learning_rate": 5e-4},
    {"name": "all_layers", "layers": "all", "initial_epoch": 25, "final_epoch": 40, "learning_rate": 1e-4},
]

PHASE_B_LEARNING_RATES = {"phase2_lr_0005": 5e-4, "phase2_lr_0025": 0.0025}
PHASE_C_NAME = "phase3_resnet50"
PHASE_C_BACKBONE = "resnet50"


# Flips and right-angle rotations only: shrub orientation has no meaning in
# nadir imagery, and arbitrary rotations would interpolate mask boundaries.
# Validation and test images are never augmented.
AUGMENTATIONS = {
    "none": (None, {"name": "none"}),
    "geometric": (
        iaa.Sequential(
            [
                iaa.Fliplr(0.5),
                iaa.Flipud(0.5),
                iaa.Sometimes(0.75, iaa.Rot90((1, 3))),
            ],
            random_order=True,
        ),
        {
            "name": "geometric",
            "horizontal_flip_probability": 0.5,
            "vertical_flip_probability": 0.5,
            "rotation_probability": 0.75,
            "rotation_quarter_turns": [1, 2, 3],
        },
    ),
}


def check_inputs() -> None:
    """Fail early if any training or validation input is missing."""
    missing = [
        str(path)
        for split in ("Train", "Val")
        for path in (TRAIN_DIR / split / "Images", TRAIN_DIR / split / "masks_points.json")
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            "The following training inputs are missing:\n  - " + "\n  - ".join(missing)
            + "\nRun 'matterport_convert_annotations.py' to create the 'masks_points.json' files."
        )


def load_dataset(split: str) -> ShrubDataset:
    dataset = ShrubDataset()
    dataset.load_dataset(
        str(TRAIN_DIR / split / "Images"),
        str(TRAIN_DIR / split / "masks_points.json"),
    )
    dataset.prepare()
    return dataset


def build_configs(dataset_train, dataset_val, learning_rate=1e-3, backbone="resnet101"):
    """Return the (training, inference) configurations of one experiment."""
    attributes = {"LEARNING_RATE": learning_rate, "BACKBONE": backbone}
    config = type("ShrubConfig", (BaselineShrubConfig,), attributes)()

    # One epoch is one full pass over the training images. The generator only
    # yields complete batches, so the last validation batch may repeat an image.
    config.STEPS_PER_EPOCH = math.ceil(len(dataset_train.image_ids) / config.BATCH_SIZE)
    config.VALIDATION_STEPS = math.ceil(len(dataset_val.image_ids) / config.BATCH_SIZE)

    # Same architecture, one image per batch, and low-confidence predictions
    # retained for the confidence-threshold selection.
    inference_config = type("ShrubInferenceConfig", (BaselineShrubConfig,), {
        **attributes,
        "GPU_COUNT": 1,
        "IMAGES_PER_GPU": 1,
        "DETECTION_MIN_CONFIDENCE": 0.0,
        "DETECTION_MAX_INSTANCES": 256,
    })()
    return config, inference_config


def run_experiment(name, config, inference_config, stages, augmentation_name, dataset_train, dataset_val) -> None:
    """Train and evaluate one experiment, unless a completed run already exists."""
    try:
        previous = select_best_experiment(EXPERIMENT_ROOT, experiment_names={name})
    except RuntimeError:
        previous = None
    if previous is not None:
        print(f"\nSkipping '{name}': completed run found at {previous['run_directory']}")
        return

    print(f"\nTraining '{name}'")
    augmentation, augmentation_metadata = AUGMENTATIONS[augmentation_name]
    TrainingExperiment(
        name=name,
        config=config,
        inference_config=inference_config,
        dataset_train=dataset_train,
        dataset_val=dataset_val,
        experiment_root=EXPERIMENT_ROOT,
        coco_model_path=COCO_MODEL_PATH,
        stages=stages,
        seed=SCREENING_SEED,
        dataset_partition=DATASET_ID,
        augmentation=augmentation,
        augmentation_metadata=augmentation_metadata,
    ).run()


def select_phase_winner(phase: str, names: set[str]) -> dict:
    best = select_best_experiment(EXPERIMENT_ROOT, experiment_names=names)
    print(
        f"\nBest {phase} experiment: {best['experiment_name']} "
        f"(mask AP50 = {best['validation_metrics']['mask_AP50']:.4f})"
    )
    return best


def save_best_model(best: dict) -> None:
    final_model_configuration = {
        "experiment_name": best["experiment_name"],
        "configuration": best["configuration"],
        "best_learning_rate": best["training_schedule"][0]["learning_rate"],
        "training_schedule": best["training_schedule"],
        "augmentation": best["augmentation_metadata"],
        "random_seed": best["random_seed"],
        "dataset_partition": best["dataset_partition"],
        "best_checkpoint_epoch": best["best_checkpoint_epoch"],
        # Relative to EXPERIMENT_ROOT (where this JSON is saved), so the
        # repository can be moved or cloned on another machine.
        "best_checkpoint_path": Path(best["best_checkpoint_path"]).resolve()
        .relative_to(EXPERIMENT_ROOT.resolve()).as_posix(),
        "validation_metrics": best["validation_metrics"],
    }
    with BEST_MODEL_METADATA.open("w", encoding="utf-8") as file:
        json.dump(final_model_configuration, file, indent=2, ensure_ascii=False)

    metrics = best["validation_metrics"]
    print("\nBest final experiment")
    print("---------------------")
    print("Experiment:", best["experiment_name"])
    print("Backbone:", best["configuration"]["BACKBONE"])
    print("Base learning rate:", best["configuration"]["LEARNING_RATE"])
    print("Augmentation:", best["augmentation_metadata"]["name"])
    print("Best epoch:", best["best_checkpoint_epoch"])
    print("Mask AP50:", metrics["mask_AP50"])
    print("Mask mAP50:95:", metrics["mask_mAP_50_95"])
    print("Mask AP75:", metrics["mask_AP75"])
    print("Confidence threshold:", metrics["selected_confidence_threshold"])
    print("F1:", metrics["f1"])
    print("Best weights:", best["best_checkpoint_path"])
    print("Saved to:", BEST_MODEL_METADATA)


def main() -> None:
    check_inputs()
    EXPERIMENT_ROOT.mkdir(parents=True, exist_ok=True)
    # Download the COCO-trained weights from the Matterport releases if needed
    if not COCO_MODEL_PATH.exists():
        utils.download_trained_weights(str(COCO_MODEL_PATH))

    dataset_train = load_dataset("Train")
    dataset_val = load_dataset("Val")
    print(f"Training images: {len(dataset_train.image_ids)}; validation images: {len(dataset_val.image_ids)}")

    def run(name, stages, augmentation_name, **config_kwargs):
        config, inference_config = build_configs(dataset_train, dataset_val, **config_kwargs)
        run_experiment(name, config, inference_config, stages, augmentation_name, dataset_train, dataset_val)

    # Phase A: training method
    phase_a = {
        "baseline_heads_noaug": (HEADS_ONLY_STAGES, "none"),
        "baseline_staged_noaug": (STAGED_STAGES, "none"),
        "baseline_staged_aug_geometric": (STAGED_STAGES, "geometric"),
    }
    for name, (stages, augmentation_name) in phase_a.items():
        run(name, stages, augmentation_name)
    best_phase_a = select_phase_winner("Phase A", set(phase_a))

    # Phase B: learning rate, keeping the Phase A schedule and augmentation
    best_a_augmentation = best_phase_a["augmentation_metadata"]["name"]
    for name, learning_rate in PHASE_B_LEARNING_RATES.items():
        run(
            name,
            scale_training_schedule(best_phase_a["training_schedule"], learning_rate),
            best_a_augmentation,
            learning_rate=learning_rate,
        )
    best_phase_b = select_phase_winner(
        "Phase A + B", {best_phase_a["experiment_name"], *PHASE_B_LEARNING_RATES}
    )

    # Phase C: backbone, keeping the Phase A + B schedule, learning rate and augmentation
    best_learning_rate = best_phase_b["configuration"]["LEARNING_RATE"]
    run(
        PHASE_C_NAME,
        scale_training_schedule(best_phase_b["training_schedule"], best_learning_rate),
        best_phase_b["augmentation_metadata"]["name"],
        learning_rate=best_learning_rate,
        backbone=PHASE_C_BACKBONE,
    )
    # ResNet-101 candidate (Phase B winner) against the ResNet-50 one
    best_final = select_phase_winner("final", {best_phase_b["experiment_name"], PHASE_C_NAME})

    save_best_model(best_final)


if __name__ == "__main__":
    main()
