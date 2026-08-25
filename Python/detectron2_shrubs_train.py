
import math
import sys
from pathlib import Path

import torch

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import (
    DatasetCatalog,
    DatasetMapper,
    build_detection_train_loader,
)
import detectron2.data.transforms as T
from detectron2.engine import DefaultTrainer, hooks
from detectron2.evaluation import COCOEvaluator

sys.path.append(str(Path(Path(__file__).absolute().parent.parent, 'bin')))
from shrubs_detectron import (
    DatasetSpec,
    normalize_dataset,
    register_juniper_dataset,
    validate_dataset,
)

ROOT = Path(__file__).absolute().parent.parent

DATASET_ROOT = Path(ROOT, "data/JuniperMapper/Photo_Interpretation_Data")
OUTPUT_DIR = Path(ROOT, "data/logs/detectron2")

TRAIN_DATASET = "shrubs_train"
VAL_DATASET = "shrubs_val"

# Only class Shrub
NUM_CLASSES = 1 

DATASET_SPECS = (
    DatasetSpec(
        name=TRAIN_DATASET,
        original_json=(
            DATASET_ROOT
            / "Train"
            / "Annotations"
            / "Train.json"
        ),
        image_root=(
            DATASET_ROOT
            / "Train"
            / "Images"
        ),
        normalized_json_name="shrubs_train_instances.json",
    ),
    DatasetSpec(
        name=VAL_DATASET,
        original_json=(
            DATASET_ROOT
            / "Val"
            / "Annotations"
            / "Val.json"
        ),
        image_root=(
            DATASET_ROOT
            / "Val"
            / "Images"
        ),
        normalized_json_name="shrubs_val_instances.json",
    ),
)


def register_datasets() -> None:
    for spec in DATASET_SPECS:
        register_juniper_dataset(
            dataset_name=spec.name,
            json_file=spec.original_json,
            image_root=spec.image_root,
        )


class MaskRCNNTrainer(DefaultTrainer):

    @classmethod
    def build_train_loader(cls, cfg):
        mapper = DatasetMapper(
            cfg,
            is_train=True,
            # Preserve deterministic resizing and avoid data augmentation
            augmentations=[
                T.ResizeShortestEdge(
                    short_edge_length=cfg.INPUT.MIN_SIZE_TRAIN,
                    max_size=cfg.INPUT.MAX_SIZE_TRAIN,
                    sample_style=cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING,
                )
            ],
        )

        return build_detection_train_loader(
            cfg,
            mapper=mapper,
        )

    @classmethod
    def build_evaluator(
        cls,
        cfg,
        dataset_name,
        output_folder=None
    ):
        return COCOEvaluator(
            dataset_name,
            tasks=("segm",),
            distributed=True,
            max_dets_per_image=(
                cfg.TEST.DETECTIONS_PER_IMAGE
            ),
            output_dir=(
                output_folder
                or str(Path(cfg.OUTPUT_DIR) / "inference" / dataset_name)
            )
        )

    def build_hooks(self):
        """This will produce 'model_best.pth' whenever segm/AP50 improves."""
        trainer_hooks = super().build_hooks()

        best_checkpointer = hooks.BestCheckpointer(
            eval_period=self.cfg.TEST.EVAL_PERIOD,
            checkpointer=self.checkpointer,
            val_metric="segm/AP50",
            mode="max",
            file_prefix="model_best",
        )

        # The hook must run after EvalHook because segm/AP50 must already
        # have been written to EventStorage.
        eval_hook_index = next(
            index
            for index, hook in enumerate(trainer_hooks)
            if isinstance(hook, hooks.EvalHook)
        )
        trainer_hooks.insert(eval_hook_index + 1, best_checkpointer)

        return trainer_hooks


def build_config(num_training_images: int):
    """
    Best-model selection scheme
    """
    cfg = get_cfg()

    config_name = (
        "COCO-InstanceSegmentation/"
        "mask_rcnn_R_101_C4_3x.yaml"
    )

    cfg.merge_from_file(model_zoo.get_config_file(config_name))

    # NOTE: CUDA operations can still be nondeterministic
    cfg.SEED = 42

    # Initialize using COCO-pretrained R101-C4 weights.
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config_name)

    cfg.SOLVER.BASE_LR = 0.0025
    cfg.SOLVER.LR_SCHEDULER_NAME = "WarmupCosineLR"
    cfg.SOLVER.WARMUP_ITERS = 2000
    cfg.SOLVER.WARMUP_FACTOR = 0.1
    cfg.SOLVER.WARMUP_METHOD = "linear"
    cfg.SOLVER.BASE_LR_END = 0.0

    cfg.DATASETS.TRAIN = (TRAIN_DATASET,)
    cfg.DATASETS.TEST = (VAL_DATASET,)

    cfg.DATALOADER.NUM_WORKERS = 4

    # Number of foreground classes. Do not add a background class.
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = NUM_CLASSES

    # Training batch size across all GPUs.
    # Reduce this if you encounter CUDA out-of-memory errors.
    cfg.SOLVER.IMS_PER_BATCH = 2

    # Replace these with values appropriate for your dataset size.
    cfg.SOLVER.MAX_ITER = 4000

    # RoIs sampled per image during training.
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256

    # Evaluate periodically.
    iterations_per_epoch = math.ceil(
        num_training_images
        / cfg.SOLVER.IMS_PER_BATCH
    )

    eval_every_epochs = 2
    evaluation_period = (
        iterations_per_epoch
        * eval_every_epochs
    )
    cfg.TEST.EVAL_PERIOD = evaluation_period
    cfg.SOLVER.CHECKPOINT_PERIOD = evaluation_period

    cfg.TEST.DETECTIONS_PER_IMAGE = 256
    
    # Deterministic preprocessing during resizing
    cfg.INPUT.MIN_SIZE_TRAIN = (800,)
    cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING = "choice"
    cfg.INPUT.MAX_SIZE_TRAIN = 1333

    cfg.INPUT.MIN_SIZE_TEST = 800
    cfg.INPUT.MAX_SIZE_TEST = 1333

    # Detectron2 normally loads OpenCV images in BGR order.
    cfg.INPUT.FORMAT = "BGR"

    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    cfg.OUTPUT_DIR = str(OUTPUT_DIR)

    # Note: Khaldi et al. 2024 paper does not state the precise COCO-pretrained
    # checkpoint, Detectron2 version, random seed, weight decay, momentum value,
    # or other inherited configuration values.
    # Exact numerical reproduction is therefore not guaranteed.

    return cfg


def maximum_instances_per_image(
    records: list[dict],
) -> int:
    return max(
        (
            len(record.get("annotations", []))
            for record in records
        ),
        default=0,
    )


def main() -> None:
    register_datasets()

    # Materialize each registered dataset exactly once for
    # the custom validation workflow.
    records_by_name = {
        spec.name: DatasetCatalog.get(spec.name)
        for spec in DATASET_SPECS
    }

    train_records = records_by_name[TRAIN_DATASET]
    val_records = records_by_name[VAL_DATASET]

    num_training_images = len(train_records)

    maximum_train_instances = (
        maximum_instances_per_image(train_records)
    )
    maximum_val_instances = (
        maximum_instances_per_image(val_records)
    )
    maximum_instances = max(
        maximum_train_instances,
        maximum_val_instances,
    )

    print(
        "\nDataset density:"
        f"\n  Maximum training instances/image: "
        f"{maximum_train_instances}"
        f"\n  Maximum validation instances/image: "
        f"{maximum_val_instances}"
        f"\n  Overall maximum instances/image: "
        f"{maximum_instances}"
    )

    cfg = build_config(num_training_images=num_training_images)

    # Increase the 256-instance limit if required by dataset density.
    if maximum_instances > cfg.TEST.DETECTIONS_PER_IMAGE:
        print(
            "\nWarning: at least one image contains more ground-truth "
            "instances than TEST.DETECTIONS_PER_IMAGE."
        )

        cfg.TEST.DETECTIONS_PER_IMAGE = max(
            256,
            maximum_instances,
        )

        print(
            "Increasing TEST.DETECTIONS_PER_IMAGE to "
            f"{cfg.TEST.DETECTIONS_PER_IMAGE}."
        )

    output_dir = Path(cfg.OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    # All validation functions receive the same materialized
    # list of dataset dictionaries.
    for spec in DATASET_SPECS:
        validate_dataset(
            spec=spec,
            records=records_by_name[spec.name]
        )

    normalized_json_dir = (
        output_dir
        / "evaluation_annotations"
    )

    # Only the validation dataset is required by COCOEvaluator.
    val_spec = next(
        spec
        for spec in DATASET_SPECS
        if spec.name == VAL_DATASET
    )

    normalized_val_json = normalize_dataset(
        spec=val_spec,
        normalized_json_dir=normalized_json_dir,
        records=val_records,
    )

    print(
        "\nValidation evaluator JSON:"
        f"\n  {normalized_val_json}"
    )

    trainer = MaskRCNNTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()


if __name__ == "__main__":
    main()