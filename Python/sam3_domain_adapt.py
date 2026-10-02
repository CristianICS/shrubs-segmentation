"""
Phase 1 domain adaptation: frozen-backbone SAM 3 decoder fine-tuning.

Trains only SAM 3's transformer/detection/segmentation head on JuniperMapper
(both the SAM 3 vision and language backbones stay frozen, via
`shrubs_sam3.training.FrozenBackboneTrainer`), so the trainable parameter
count stays small relative to the dataset. This is a thin CLI wrapper:
it converts JuniperMapper's grouped polygons to per-instance COCO with
`prepare_juniper_coco`, then drives the upstream `sam3.train.trainer.Trainer`
machinery through the Hydra config in `bin/shrubs_sam3/configs/
juniper_frozen_backbone.yaml`, the same way SAM 3's own `sam3/train/train.py`
launcher does for a single local process.

After every epoch the model is validated on the JuniperMapper Photo
Interpretation Val split, and the epoch with the highest COCO mask AP50
(ties broken by mask mAP50:95, then the earliest epoch) is saved as
`<output>/checkpoints/model_best.pt`, with its scores in `model_best.json`.
This is the same selection criterion as the Detectron2 and Matterport
Mask R-CNN models.

Example usage (inside the `sam3_shrubs` environment):

    python Python/sam3_domain_adapt.py --checkpoint /path/to/sam3.pt

Every other path and option has a project default; run with `--help` to
list them.

`--checkpoint` points to the official SAM 3 release checkpoint in your local
folder. If omitted, the gated checkpoint is downloaded from Hugging Face instead
(requires `hf auth login`).
"""
from __future__ import annotations

import argparse
import os
import socket
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BIN_DIR = PROJECT_ROOT / "bin"
if str(BIN_DIR) not in sys.path:
    sys.path.insert(0, str(BIN_DIR))

# Catch a wrong environment
try:
    from shrubs_sam3 import prepare_juniper_coco  # noqa: E402
except ImportError as e:
    raise ImportError(
        "SAM 3 dependencies are not installed in the active Python environment. "
        "Activate the 'sam3_shrubs' environment (WSL2) before running this script."
    ) from e

CONFIG_DIR = BIN_DIR / "shrubs_sam3" / "configs"
CONFIG_NAME = "juniper_frozen_backbone"

JM_DIR = PROJECT_ROOT / "data" / "JuniperMapper"
DEFAULT_TRAIN_JSON = JM_DIR / "Photo_Interpretation_Data/Train/Annotations/Train.json"
DEFAULT_TRAIN_IMAGES = JM_DIR / "Photo_Interpretation_Data/Train/Images"
DEFAULT_VAL_JSON = JM_DIR / "Photo_Interpretation_Data/Val/Annotations/Val.json"
DEFAULT_VAL_IMAGES = JM_DIR / "Photo_Interpretation_Data/Val/Images"
DEFAULT_OUTPUT = PROJECT_ROOT / "data/logs/sam3/domain_adapt"


def existing_file(value: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"file does not exist: {path}")
    return path


def existing_dir(value: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_dir():
        raise argparse.ArgumentTypeError(f"directory does not exist: {path}")
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Phase 1 domain adaptation: fine-tune SAM 3's decoder on JuniperMapper "
            "with both backbones frozen."
        )
    )
    parser.add_argument(
        "--checkpoint", type=existing_file, default=None,
        help="Local official SAM 3 release checkpoint to start training from "
             "(same file sam3_deploy.py's --checkpoint takes). Omit to "
             "download the gated release checkpoint from Hugging Face instead.",
    )
    parser.add_argument(
        "--train-json", type=existing_file, default=DEFAULT_TRAIN_JSON,
        help="JuniperMapper COCO annotations for training (default: %(default)s).",
    )
    parser.add_argument(
        "--train-images", type=existing_dir, default=DEFAULT_TRAIN_IMAGES,
        help="Training image folder (default: %(default)s).",
    )
    parser.add_argument(
        "--val-json", type=existing_file, default=DEFAULT_VAL_JSON,
        help="JuniperMapper COCO annotations for validation (default: %(default)s).",
    )
    parser.add_argument(
        "--val-images", type=existing_dir, default=DEFAULT_VAL_IMAGES,
        help="Validation image folder (default: %(default)s).",
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT,
        help="Experiment directory: converted annotations, checkpoints, logs "
             "(default: %(default)s). Training resumes if it already holds "
             "checkpoints/checkpoint.pt.",
    )
    parser.add_argument(
        "--prompt", default="shrub",
        help="Category name written into the converted COCO files and matched "
             "against by SAM 3's text-conditioned decoder (default: %(default)s).",
    )
    parser.add_argument(
        "--bpe-path", type=existing_file, default=None,
        help="Path to SAM 3's BPE vocab file. Defaults to the file installed "
             "alongside the sam3 package (sam3/assets/bpe_simple_vocab_16e6.txt.gz).",
    )
    parser.add_argument(
        "--max-epochs", type=int, default=30,
        help="Number of training epochs (default: %(default)s).",
    )
    parser.add_argument(
        "--val-epoch-freq", type=int, default=1,
        help="Validate (and update model_best.pt) every N epochs (default: %(default)s).",
    )
    parser.add_argument(
        "--checkpoint-freq", type=int, default=5,
        help="Save a numbered checkpoint every N epochs, 0 disables (default: %(default)s).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=2,
        help="Training batch size (default: %(default)s).",
    )
    parser.add_argument(
        "--num-workers", type=int, default=4,
        help="Dataloader worker processes. Set to 0 if Windows multiprocessing "
             "causes issues (default: %(default)s).",
    )
    parser.add_argument(
        "--device", choices=("cuda", "cpu"), default="cuda",
        help="Training device (default: %(default)s).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Passed straight through to the trainer for run-to-run "
             "reproducibility (default: %(default)s).",
    )
    return parser


def check_inputs(args: argparse.Namespace) -> None:
    """Fail early, listing every missing training input at once.

    argparse only applies `existing_file`/`existing_dir` to values typed on the
    command line, not to the Path defaults, so those are checked here.
    """
    inputs = (
        ("--train-json", args.train_json),
        ("--train-images", args.train_images),
        ("--val-json", args.val_json),
        ("--val-images", args.val_images),
    )

    # Only relevant when the inputs point inside the JuniperMapper database
    if not JM_DIR.exists() and any(Path(path).resolve().is_relative_to(JM_DIR) for _, path in inputs):
        raise FileNotFoundError(
            f"JuniperMapper database not found at '{JM_DIR}'. "
            "Place a valid database copy inside the 'data' directory."
        )

    missing = [f"  - {name}: {path}" for name, path in inputs if not Path(path).exists()]
    if missing:
        raise FileNotFoundError(
            "The following training inputs are missing:\n" + "\n".join(missing)
        )


def _default_bpe_path() -> Path:
    import sam3

    bpe_path = Path(sam3.__file__).resolve().parent / "assets" / "bpe_simple_vocab_16e6.txt.gz"
    if not bpe_path.is_file():
        raise FileNotFoundError(
            f"Could not find SAM 3's BPE vocab at {bpe_path}; pass --bpe-path explicitly."
        )
    return bpe_path


def _to_override(path: Path) -> str:
    """Quote a Windows path as a Hydra override value.

    Forward slashes avoid backslash-escaping issues in Hydra's override
    grammar; the surrounding quotes stop the drive letter's `:` from being
    parsed as part of that grammar instead of the path.
    """
    return "'" + str(path).replace("\\", "/") + "'"


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    check_inputs(args)
    bpe_path = args.bpe_path or _default_bpe_path()

    output = args.output.expanduser().resolve()
    if (output / "checkpoints" / "checkpoint.pt").is_file():
        print(
            f"Found an existing run in '{output}': training resumes from "
            "checkpoints/checkpoint.pt. Pass a new --output to start from scratch."
        )
    elif args.checkpoint is None:
        print("No --checkpoint given: downloading the gated SAM 3 checkpoint from Hugging Face.")

    annotations_dir = output / "annotations"
    train_json = annotations_dir / "train.json"
    val_json = annotations_dir / "val.json"

    print("Converting JuniperMapper annotations to per-instance COCO...")
    prepare_juniper_coco(args.train_json, train_json, prompt=args.prompt)
    prepare_juniper_coco(args.val_json, val_json, prompt=args.prompt)

    overrides = [
        f"paths.train_ann_file={_to_override(train_json)}",
        f"paths.train_img_folder={_to_override(args.train_images)}",
        f"paths.val_ann_file={_to_override(val_json)}",
        f"paths.val_img_folder={_to_override(args.val_images)}",
        f"paths.experiment_log_dir={_to_override(output)}",
        f"paths.bpe_path={_to_override(bpe_path)}",
        f"trainer.max_epochs={args.max_epochs}",
        f"trainer.val_epoch_freq={args.val_epoch_freq}",
        f"trainer.checkpoint.save_freq={args.checkpoint_freq}",
        f"trainer.accelerator={args.device}",
        f"trainer.seed_value={args.seed}",
        f"scratch.train_batch_size={args.batch_size}",
        f"scratch.num_train_workers={args.num_workers}",
    ]
    # nccl (the config default) needs CUDA and has no Windows build at all;
    # fall back to gloo for a CPU run or on native Windows (where gloo also
    # needs the libuv workaround documented in environment_sam3.yml). Linux
    # CUDA runs -- including WSL2, the recommended way to run this script --
    # keep the nccl default untouched.
    if args.device == "cpu" or sys.platform == "win32":
        overrides.append("trainer.distributed.backend=gloo")
    if args.checkpoint is not None:
        overrides.append(f"trainer.model.checkpoint_path={_to_override(args.checkpoint)}")
        overrides.append("trainer.model.load_from_HF=false")

    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from sam3.train.utils.train_utils import register_omegaconf_resolvers

    register_omegaconf_resolvers()
    initialize_config_dir(config_dir=str(CONFIG_DIR), version_base="1.2")
    cfg = compose(config_name=CONFIG_NAME, overrides=overrides)

    output.mkdir(parents=True, exist_ok=True)
    with (output / "config_resolved.yaml").open("w", encoding="utf-8") as destination:
        destination.write(OmegaConf.to_yaml(cfg, resolve=True))

    # Single local process, matching sam3.train.train's `single_proc_run` for
    # a one-GPU (or CPU) job: no submitit/SLURM launcher is wired in here,
    # since domain adaptation on JuniperMapper's size doesn't need a cluster.
    #
    # A literal loopback IP is used instead of the "localhost" hostname:
    # PyTorch's gloo backend resolves MASTER_ADDR to a network device via
    # makeDeviceForHostname, which fails on Windows for "localhost" specifically
    # (RuntimeError: makeDeviceForHostname(): unsupported gloo device) even
    # though the process group has only one member.
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(_free_port())
    os.environ["RANK"] = "0"
    os.environ["LOCAL_RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"

    trainer = instantiate(cfg.trainer, _recursive_=False)
    trainer.run()

    checkpoint_dir = output / "checkpoints"
    print(f"Finished. Checkpoints written to: {checkpoint_dir}")
    print(
        "  model_best.pt       -- highest mask AP50 on the Photo Interpretation "
        "Val split (the checkpoint to deploy; scores in model_best.json)\n"
        "  checkpoint.pt       -- latest epoch\n"
        f"  checkpoint_<N>.pt   -- every {args.checkpoint_freq} epochs"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
