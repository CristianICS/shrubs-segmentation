from pathlib import Path
from datetime import datetime, timezone
from dataclasses import asdict, dataclass, fields
from typing import Union, List, Set
import re
import gc
import sys
import json
import time
import random

from keras.callbacks import CSVLogger
from keras.callbacks import ModelCheckpoint
import numpy as np
import imgaug as ia
import tensorflow as tf
from keras import backend as K

sys.path.append(
    str(Path(__file__).resolve().parent.parent / "bin")
)
import mrcnn.model as modellib
from shrubs_matterport.evaluate import Validation


EXECUTION_MODES = {
    "full",
    "evaluation_only",
    "resume_training",
}


def set_global_seed(seed: int) -> None:
    """
    Set the main Python, NumPy, TensorFlow and imgaug seeds.

    This improves reproducibility, although exact GPU reproducibility
    can still depend on TensorFlow, CUDA and cuDNN versions.
    """
    random.seed(seed)
    np.random.seed(seed)
    ia.seed(seed)

    try:
        tf.random.set_seed(seed)
    except AttributeError:
        tf.compat.v1.set_random_seed(seed)

    # Available in some TensorFlow versions.
    try:
        tf.config.experimental.enable_op_determinism()
    except (AttributeError, RuntimeError):
        pass


def calculate_iterations(config, training_stages):
    """
    Return the number of epochs run during the training step
    
    Parameters
    ----------
    config :
        A mrcnn.config.Config class instance.
    """
    for stage in training_stages:
        stage["epoch_count"] = (
            stage["final_epoch"] - stage["initial_epoch"]
        )
        stage["iteration_count"] = (
            stage["epoch_count"] * config.STEPS_PER_EPOCH
        )

    total_epochs = training_stages[-1]["final_epoch"]

    total_iterations = sum(
        stage["iteration_count"]
        for stage in training_stages
    )

    return total_epochs, total_iterations


@dataclass
class RunDirectory:
    root: Path
    checkpoints: Path
    evaluation: Path
    evaluation_checkpoints: Path
    logs: Path
    training_csv: Path


def create_run_directory(
    experiment_name: str,
    experiment_root: Path,
    seed: int,
) -> RunDirectory:
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", experiment_name)

    run_root = experiment_root / f"{safe_name}_seed{seed}_{timestamp}"

    run_dir = RunDirectory(
        root=run_root,
        checkpoints=run_root / "checkpoints",
        evaluation=run_root / "evaluation",
        evaluation_checkpoints=run_root / "evaluation" / "checkpoints",
        logs=run_root / "logs",
        # Store the loss and other metrics produced during training phase into a CSV file
        training_csv = run_root / "logs" / "training_history.csv"
    )
    
    run_dir.checkpoints.mkdir(parents=True, exist_ok=False)
    run_dir.evaluation.mkdir()
    run_dir.logs.mkdir()

    return run_dir


def load_run_directory(
    run_root: Union[str, Path],
) -> RunDirectory:
    """Reconstruct the directory structure of an existing run."""
    run_root = Path(run_root).resolve()

    run_dir = RunDirectory(
        root=run_root,
        checkpoints=run_root / "checkpoints",
        evaluation=run_root / "evaluation",
        evaluation_checkpoints=(
            run_root / "evaluation" / "checkpoints"
        ),
        logs=run_root / "logs",
        training_csv=(
            run_root / "logs" / "training_history.csv"
        ),
    )

    required_paths = [
        run_dir.root,
        run_dir.checkpoints,
        run_dir.root / "run_metadata.json",
    ]

    missing_paths = [
        str(path)
        for path in required_paths
        if not path.exists()
    ]

    if missing_paths:
        raise FileNotFoundError(
            "The prior run is incomplete. Missing paths: "
            + ", ".join(missing_paths)
        )

    run_dir.evaluation.mkdir(
        parents=True,
        exist_ok=True,
    )
    run_dir.evaluation_checkpoints.mkdir(
        parents=True,
        exist_ok=True,
    )
    run_dir.logs.mkdir(
        parents=True,
        exist_ok=True,
    )

    return run_dir


def to_json_compatible(value):
    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, tuple):
        return [to_json_compatible(item) for item in value]

    if isinstance(value, list):
        return [to_json_compatible(item) for item in value]

    if isinstance(value, dict):
        return {
            str(key): to_json_compatible(item)
            for key, item in value.items()
        }

    if isinstance(value, (str, int, float, bool)) or value is None:
        return value

    return repr(value)


def config_to_dict(config) -> dict:
    return {
        name: to_json_compatible(getattr(config, name))
        for name in dir(config)
        if name.isupper() and not callable(getattr(config, name))
    }


def save_json(data: dict, path: Union[str, Path]) -> None:
    # Transform RunMetadata instances into a correct dictionary
    if hasattr(data, "__dataclass_fields__"):
        data = asdict(data)

    with Path(path).open("w", encoding="utf-8") as file:
        json.dump(
            to_json_compatible(data),
            file,
            indent=2,
            ensure_ascii=False,
        )


@dataclass
class RunMetadata:
    metadata_path: str
    experiment_name: str
    dataset_partition: str
    status: str
    run_directory: str
    created_at_utc: str

    random_seed: int
    configuration: dict
    initial_weights: dict
    augmentation_metadata: dict
    training_schedule: List[dict]

    total_epochs_planned: int
    total_iterations_planned: int

    training_time_seconds: Union[float, None] = None
    training_time_hours: Union[float, None] = None
    training_finished_at_utc: Union[str, None] = None
    training_error: Union[str, None] = None

    checkpoint_count: Union[int, None] = None
    best_checkpoint_epoch: Union[int, None] = None
    best_checkpoint_path: Union[str, None] = None
    best_checkpoint_original_path: Union[str, None] = None
    validation_metrics: Union[dict, None] = None

    execution_mode: str = "full"

    last_completed_epoch: int = 0
    resumed_from_epoch: Union[int, None] = None
    resumed_from_checkpoint: Union[str, None] = None
    resume_count: int = 0

    evaluation_error: Union[str, None] = None
    evaluation_finished_at_utc: Union[str, None] = None


def initialize_run_metadata(
    metadata_path: Path,
    *,
    experiment_name: str,
    run_dir: RunDirectory,
    dataset_partition: str,
    screening_seed: int,
    configuration: dict,
    initial_weights: dict,
    augmentation_metadata: dict,
    training_stages: List[dict],
    total_epochs: int,
    total_iterations: int,
    execution_mode: str = "full"
):
    metadata = RunMetadata(
        metadata_path=str(metadata_path.resolve()),
        experiment_name=experiment_name,
        dataset_partition=dataset_partition,
        status="initialized",
        run_directory=str(run_dir.root.resolve()),
        created_at_utc=datetime.now(timezone.utc).isoformat(),

        random_seed=screening_seed,
        configuration=configuration,
        initial_weights=initial_weights,
        augmentation_metadata=augmentation_metadata,
        training_schedule=training_stages,

        total_epochs_planned=total_epochs,
        total_iterations_planned=total_iterations,
        execution_mode=execution_mode,
    )

    save_json(metadata, metadata_path)
    return metadata


def load_run_metadata(
    metadata_path: Union[str, Path],
) -> RunMetadata:
    """Load a previous metadata while supporting older metadata files."""
    metadata_path = Path(metadata_path)

    with metadata_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        stored = json.load(file)

    valid_field_names = {
        field.name
        for field in fields(RunMetadata)
    }

    compatible_data = {
        key: value
        for key, value in stored.items()
        if key in valid_field_names
    }

    return RunMetadata(**compatible_data)


class TrainingExperiment:
    def __init__(
        self,
        *,
        name,
        config,
        inference_config,
        dataset_train,
        dataset_val,
        experiment_root,
        coco_model_path,
        stages,
        seed,
        dataset_partition,
        augmentation=None,
        augmentation_metadata=None,
        execution_mode="full",
        prior_run_directory=None,
    ):
        """
        Three execution mode:

        full: create a new run, train from COCO weights,
            then evaluate;
        evaluation_only: load an existing run and evaluate its
            checkpoints without training;
        resume_training: load an interrupted run, restore the
            newest checkpoint, continue from the next epoch, then evaluate.

        Note that resuming restores the model weights but not necessarily the
        complete optimizer state, including momentum-related variables. Consequently,
        resumed training is operationally valid, but it may not produce bit-for-bit
        identical results to an uninterrupted run.
        """
        if execution_mode not in EXECUTION_MODES:
            raise ValueError(
                "execution_mode must be one of: "
                + ", ".join(sorted(EXECUTION_MODES))
            )

        self.execution_mode = execution_mode

        self.prior_run_directory = (
            None
            if prior_run_directory is None
            else Path(prior_run_directory).resolve()
        )

        if (
            execution_mode in {
                "evaluation_only",
                "resume_training",
            }
            and self.prior_run_directory is None
        ):
            raise ValueError(
                "prior_run_directory is required when "
                "execution_mode is 'evaluation_only' or "
                "'resume_training'."
            )

        if (
            execution_mode == "full"
            and self.prior_run_directory is not None
        ):
            raise ValueError(
                "prior_run_directory must not be provided "
                "when execution_mode='full'."
            )

        if (
            execution_mode
            in {"full", "resume_training"}
            and dataset_train is None
        ):
            raise ValueError(
                "dataset_train is required for training."
            )

        self.name = name
        self.config = config
        self.inference_config = inference_config
        self.dataset_train = dataset_train
        self.dataset_val = dataset_val
        self.experiment_root = Path(experiment_root)
        self.coco_model_path = Path(coco_model_path)
        self.stages = stages
        self.dataset_partition = dataset_partition
        self.seed = seed
        self.augmentation = augmentation

        self.augmentation_metadata = (
            augmentation_metadata or {
                "name": "none",
                "operations": []
            }
        )
    
        self.excluded_layers = [
            "mrcnn_class_logits",
            "mrcnn_bbox_fc",
            "mrcnn_bbox",
            "mrcnn_mask",
        ]

    def run(self):
        try:
            if self.execution_mode == "full":
                self._prepare()
                self._train(completed_epoch=0)

            elif self.execution_mode == "evaluation_only":
                self._load_existing_run()

            elif self.execution_mode == "resume_training":
                self._load_existing_run()
                self._prepare_resume_training()

                self._train(
                    completed_epoch=self.run_metadata.resumed_from_epoch
                )

            self._evaluate()

            return self.run_metadata

        finally:
            self.cleanup()

    def _prepare(self):
        set_global_seed(self.seed)

        self.run_dir = create_run_directory(
            experiment_name=self.name,
            experiment_root=self.experiment_root,
            seed=self.seed,
        )

        self.total_epochs, self.total_iterations = (
            calculate_iterations(self.config, self.stages)
        )

        self.metadata_path = (
            self.run_dir.root / "run_metadata.json"
        )

        self.run_metadata = initialize_run_metadata(
            self.metadata_path,
            experiment_name=self.name,
            run_dir=self.run_dir,
            dataset_partition=self.dataset_partition,
            screening_seed=self.seed,
            configuration=config_to_dict(self.config),
            initial_weights={
                "source": "MS COCO",
                "path": str(self.coco_model_path.resolve()),
                "excluded_layers": self.excluded_layers,
            },
            augmentation_metadata=self.augmentation_metadata,
            training_stages=self.stages,
            total_epochs=self.total_epochs,
            total_iterations=self.total_iterations,
            execution_mode=self.execution_mode,
        )

        self.model = modellib.MaskRCNN(
            mode="training",
            config=self.config,
            model_dir=str(self.run_dir.logs),
        )

        self.model.load_weights(
            str(self.coco_model_path),
            by_name=True,
            exclude=self.excluded_layers,
        )

        self.callbacks = self._create_callbacks()

    def _train(self, completed_epoch=0):
        start = time.perf_counter()

        if completed_epoch == 0:
            stages_to_run = self.stages
        else:
            stages_to_run = (
                self._remaining_training_stages(
                    completed_epoch
                )
            )

        try:
            self.run_metadata.status = (
                "training_in_progress"
            )

            save_json(
                self.run_metadata,
                self.metadata_path
            )

            for stage in stages_to_run:

                self.model.epoch = completed_epoch

                self.model.train(
                    self.dataset_train,
                    self.dataset_val,
                    learning_rate=stage["learning_rate"],
                    epochs=stage["final_epoch"],
                    layers=stage["layers"],
                    augmentation=self.augmentation,
                    custom_callbacks=self.callbacks,
                    validation_freq=1,
                    save_default_checkpoints=False,
                )

                completed_epoch = stage["final_epoch"]

                self.run_metadata.last_completed_epoch = (
                    completed_epoch
                )

                save_json(
                    self.run_metadata,
                    self.metadata_path,
                )

            self.run_metadata.status = (
                "training_completed"
            )
            self.run_metadata.training_error = None

        except Exception as error:
            self.run_metadata.status = "training_interrupted"
            self.run_metadata.training_error = repr(error)

            try:
                latest_checkpoint = (
                    self._find_latest_checkpoint()
                )
                self.run_metadata.last_completed_epoch = (
                    self._checkpoint_epoch(
                        latest_checkpoint
                    )
                )
            except FileNotFoundError:
                pass

            raise

        finally:
            elapsed = time.perf_counter() - start

            previous_seconds = (
                self.run_metadata.training_time_seconds
                or 0.0
            )

            self.run_metadata.training_time_seconds = (
                previous_seconds + elapsed
            )
            self.run_metadata.training_time_hours = (
                self.run_metadata.training_time_seconds
                / 3600
            )

            self.run_metadata.training_finished_at_utc = (
                datetime.now(timezone.utc).isoformat()
            )

            save_json(
                self.run_metadata,
                self.metadata_path,
            )

    def _evaluate(self):
        self.run_metadata.status = "evaluation_in_progress"
        self.run_metadata.evaluation_error = None
        save_json(
            self.run_metadata,
            self.metadata_path,
        )
        
        inference_model = modellib.MaskRCNN(
            mode="inference",
            config=self.inference_config,
            model_dir=str(self.run_dir.evaluation),
        )

        try:
            validation = Validation(
                dataset_val=self.dataset_val,
                checkpoints_dir=self.run_dir.checkpoints,
                total_epochs_planned=self.total_epochs,
            )

            validation.evaluate_all_checkpoints_ap(
                inference_model,
                self.run_dir.evaluation_checkpoints,
            )

            validation.select_best_checkpoint(
                self.run_dir.evaluation
            )

            validation.evaluate_best_checkpoint_threshold(
                inference_model,
                self.run_dir.evaluation,
                confidence_thresholds=np.arange(
                    0.0, 0.951, 0.05
                ),
                mask_iou_threshold=0.50,
            )

            validation.store_final_run(
                self.run_metadata
            )

            self.run_metadata.evaluation_finished_at_utc = (
                datetime.now(timezone.utc).isoformat()
            )

            save_json(
                self.run_metadata,
                self.metadata_path,
            )

        except Exception as error:
            self.run_metadata.status = "evaluation_failed"
            self.run_metadata.evaluation_error = repr(error)

            self.run_metadata.evaluation_finished_at_utc = (
                datetime.now(timezone.utc).isoformat()
            )

            save_json(
                self.run_metadata,
                self.metadata_path,
            )
            raise

        finally:
            del inference_model

    def _create_callbacks(self):
        """Define the required callbacks to train the models."""
        return [
            ModelCheckpoint(
                str(
                    self.run_dir.checkpoints /
                    "mask_rcnn_shrubs_epoch_{epoch:04d}.h5"
                ),
                save_weights_only=True,
                save_best_only=False,
                verbose=1,
            ),
            CSVLogger(
                str(self.run_dir.training_csv),
                append=True,
            ),
        ]

    @staticmethod
    def _normalized_json(value):
        """
        JSON serialization can produce minor floating-point differences.
        
        Values such as 0.0005 should normally compare correctly, but
        a normalized comparison is safer.
        """
        return json.dumps(
            to_json_compatible(value),
            sort_keys=True,
            separators=(",", ":"),
        )

    @staticmethod
    def _checkpoint_epoch(path: Union[str, Path]) -> int:
        match = re.search(r"epoch_(\d+)\.h5$", Path(path).name)

        if match is None:
            raise ValueError(
                f"Could not extract epoch from checkpoint: {path}"
            )

        return int(match.group(1))

    @staticmethod
    def _canonical_schedule(stages):
        """Remove derived values before comparing schedules."""
        return [
            {
                "initial_epoch": int(stage["initial_epoch"]),
                "final_epoch": int(stage["final_epoch"]),
                "layers": stage["layers"],
                "learning_rate": float(
                    stage["learning_rate"]
                ),
            }
            for stage in stages
        ]

    def _find_latest_checkpoint(self):
        """Return the checkpoint with the greatest epoch number."""
        checkpoint_paths = list(
            self.run_dir.checkpoints.glob(
                "mask_rcnn_shrubs_epoch_*.h5"
            )
        )

        if not checkpoint_paths:
            raise FileNotFoundError(
                "No training checkpoints were found in: "
                + str(self.run_dir.checkpoints)
            )

        return max(
            checkpoint_paths,
            key=self._checkpoint_epoch,
        )

    def _load_existing_run(self):
        """Load directories and metadata from a prior run."""
        self.run_dir = load_run_directory(
            self.prior_run_directory
        )

        self.metadata_path = (
            self.run_dir.root / "run_metadata.json"
        )

        self.run_metadata = load_run_metadata(
            self.metadata_path
        )

        self.total_epochs = (
            self.run_metadata.total_epochs_planned
        )
        self.total_iterations = (
            self.run_metadata.total_iterations_planned
        )

        self._validate_existing_run()

        self.run_metadata.execution_mode = (
            self.execution_mode
        )

        save_json(
            self.run_metadata,
            self.metadata_path,
        )
        

    def _validate_existing_run(self):
        """Confirm that the prior run matches this experiment."""
        expected_configuration = config_to_dict(
            self.config
        )

        comparisons = {
            "experiment_name": (
                self.run_metadata.experiment_name,
                self.name,
            ),
            "dataset_partition": (
                self.run_metadata.dataset_partition,
                self.dataset_partition,
            ),
            "random_seed": (
                self.run_metadata.random_seed,
                self.seed,
            ),
            "configuration": (
                self._normalized_json(
                    self.run_metadata.configuration
                ),
                self._normalized_json(
                    expected_configuration
                ),
            ),
            "training_schedule": (
                self._normalized_json(
                    self._canonical_schedule(
                        self.run_metadata.training_schedule
                    )
                ),
                self._normalized_json(
                    self._canonical_schedule(
                        self.stages
                    )
                )
            ),
            "augmentation_metadata": (
                self._normalized_json(
                    self.run_metadata.augmentation_metadata
                ),
                self._normalized_json(
                    self.augmentation_metadata
                )
            ),
        }

        mismatches = [
            field_name
            for field_name, (stored, expected) in comparisons.items()
            if stored != expected
        ]

        if mismatches:
            raise ValueError(
                "The prior run does not match the current "
                "experiment definition. Mismatched fields: "
                + ", ".join(mismatches)
            )

    def _prepare_resume_training(self):
        """Restore the most recent checkpoint of an interrupted run."""
        set_global_seed(self.seed)

        latest_checkpoint = (
            self._find_latest_checkpoint()
        )
        completed_epoch = self._checkpoint_epoch(
            latest_checkpoint
        )

        if completed_epoch >= self.total_epochs:
            raise RuntimeError(
                "Training is already complete. Use "
                "execution_mode='evaluation_only' instead."
            )

        self.model = modellib.MaskRCNN(
            mode="training",
            config=self.config,
            model_dir=str(self.run_dir.logs),
        )

        self.model.load_weights(
            str(latest_checkpoint),
            by_name=True,
        )

        self.run_metadata.status = (
            "resuming_training"
        )
        self.run_metadata.execution_mode = (
            "resume_training"
        )
        self.run_metadata.last_completed_epoch = (
            completed_epoch
        )
        self.run_metadata.resumed_from_epoch = (
            completed_epoch
        )
        self.run_metadata.resumed_from_checkpoint = str(
            latest_checkpoint.resolve()
        )
        self.run_metadata.resume_count += 1
        self.run_metadata.training_error = None

        self.callbacks = self._create_callbacks()

        save_json(
            self.run_metadata,
            self.metadata_path,
        )

    def _remaining_training_stages(
        self,
        completed_epoch,
    ):
        """Return the portions of the schedule not yet completed."""
        remaining = []

        for stage in self.stages:
            stage_final_epoch = stage["final_epoch"]

            if stage_final_epoch <= completed_epoch:
                continue

            resumed_stage = dict(stage)
            resumed_stage["initial_epoch"] = max(
                completed_epoch,
                stage["initial_epoch"],
            )

            remaining.append(resumed_stage)

        if not remaining:
            raise RuntimeError(
                "No remaining training stages were found."
            )

        return remaining

    def cleanup(self):
        """Release model references after completing an experiment."""
        if hasattr(self, "model"):
            del self.model

        K.clear_session()
        gc.collect()


def select_best_experiment(
    experiment_root: Union[str, Path],
    experiment_names: Union[Set[str], None] = None,
) -> dict:
    """
    Detect the best experiments
    
    Select the latest completed run for each experiment
    name before comaring them.
    """
    latest_by_name = {}
    
    for metadata_path in Path(experiment_root).glob(
        "*/run_metadata.json"
    ):
        with metadata_path.open("r", encoding="utf-8") as file:
            metadata = json.load(file)

        if metadata.get("status") != "evaluation_completed":
            continue

        name = metadata["experiment_name"]

        if (
            experiment_names is not None
            and name not in experiment_names
        ):
            continue

        metrics = metadata.get("validation_metrics") or {}
        if "mask_mAP_50_95" not in metrics:
            continue

        previous = latest_by_name.get(name)

        if (
            previous is None
            or metadata["created_at_utc"]
            > previous["created_at_utc"]
        ):
            latest_by_name[name] = metadata

    records = list(latest_by_name.values())

    if not records:
        raise RuntimeError(
            "No completed experiments were found."
        )

    return max(
        records,
        key=lambda item: (
            item["validation_metrics"]["mask_mAP_50_95"],
            item["validation_metrics"]["mask_AP75"],
            item["validation_metrics"]["mask_AP50"],
        ),
    )


def scale_training_schedule(stages, base_learning_rate):
    """Update the learning rates"""
    original_base_lr = stages[0]["learning_rate"]
    scale = base_learning_rate / original_base_lr

    return [
        {
            **stage,
            "learning_rate": stage["learning_rate"] * scale,
        }
        for stage in stages
    ]