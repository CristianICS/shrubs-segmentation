"""Phase 1 domain adaptation: frozen-backbone SAM 3 decoder fine-tuning.

Trains only SAM 3's transformer/detection/segmentation head on JuniperMapper
(both the SAM 3 vision and language backbones stay frozen, via
`shrubs_sam3.training.FrozenBackboneTrainer`), so the trainable parameter
count stays small relative to the dataset -- the conservative recipe from
`SAM3_upgrade.md`'s Phase 1. This is a thin CLI wrapper: it converts
JuniperMapper's grouped polygons to per-instance COCO with
`prepare_juniper_coco`, then drives the upstream `sam3.train.trainer.Trainer`
machinery through the Hydra config in `bin/shrubs_sam3/configs/
juniper_frozen_backbone.yaml`, the same way SAM 3's own `sam3/train/train.py`
launcher does for a single local process.

The zero-shot CLI (`sam3_shrubs.py`) and its default behavior are untouched.
Once training finishes, compare checkpoints with the existing evaluation
harness, since this script does not itself pick a "best" mask model:

    python Python/sam3_evaluate.py \\
      --dataset shrubs_val ... \\
      --checkpoint results/sam3_domain_adapt/checkpoints/checkpoint_10.pt

Example usage:

    python Python/sam3_domain_adapt.py \\
      --output results/sam3_domain_adapt --max-epochs 30 \\
      --checkpoint models/sam3/sam3.pt

`--checkpoint` is optional: it starts training from an already-downloaded
local release checkpoint (the same file `sam3_shrubs.py --checkpoint` and
`sam3_evaluate.py --checkpoint` take) instead of re-downloading the gated
checkpoint from Hugging Face.
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

from shrubs_sam3 import prepare_juniper_coco  # noqa: E402

CONFIG_DIR = BIN_DIR / "shrubs_sam3" / "configs"
CONFIG_NAME = "juniper_frozen_backbone"

DEFAULT_TRAIN_JSON = PROJECT_ROOT / "data/JuniperMapper/Photo_Interpretation_Data/Train/Annotations/Train.json"
DEFAULT_TRAIN_IMAGES = PROJECT_ROOT / "data/JuniperMapper/Photo_Interpretation_Data/Train/Images"
DEFAULT_VAL_JSON = PROJECT_ROOT / "data/JuniperMapper/Photo_Interpretation_Data/Val/Annotations/Val.json"
DEFAULT_VAL_IMAGES = PROJECT_ROOT / "data/JuniperMapper/Photo_Interpretation_Data/Val/Images"


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
    parser.add_argument("--train-json", type=existing_file, default=DEFAULT_TRAIN_JSON)
    parser.add_argument("--train-images", type=existing_dir, default=DEFAULT_TRAIN_IMAGES)
    parser.add_argument("--val-json", type=existing_file, default=DEFAULT_VAL_JSON)
    parser.add_argument("--val-images", type=existing_dir, default=DEFAULT_VAL_IMAGES)
    parser.add_argument(
        "--output", type=Path, required=True,
        help="Experiment directory: converted annotations, checkpoints, logs.",
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
        "--checkpoint", type=existing_file, default=None,
        help="Optional local official SAM 3 release checkpoint to start training "
             "from (same file sam3_shrubs.py's --checkpoint takes). Omit to "
             "download the gated release checkpoint from Hugging Face instead.",
    )
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--val-epoch-freq", type=int, default=1)
    parser.add_argument("--checkpoint-freq", type=int, default=5,
                         help="Save a numbered checkpoint every N epochs (0 disables).")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument(
        "--num-workers", type=int, default=4,
        help="Dataloader worker processes. Set to 0 if Windows multiprocessing "
             "causes issues.",
    )
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Passed straight through to the trainer for run-to-run reproducibility.",
    )
    return parser


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
    bpe_path = args.bpe_path or _default_bpe_path()

    output = args.output.expanduser().resolve()
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
        "  checkpoint.pt                          -- latest epoch\n"
        f"  checkpoint_<N>.pt                       -- every {args.checkpoint_freq} epochs\n"
        "  val_shrubs_val_detection_<metric>.pt   -- saved whenever that bbox "
        "metric (e.g. AP, AP50) improved during validation\n"
        "None of these is picked automatically as the mask model to deploy -- "
        "that in-loop meter only tracks box AP, not mask quality. Compare "
        "candidates with Python/sam3_evaluate.py's pixel/instance-IoU/S-IoU "
        "metrics before picking one, e.g.:\n"
        f"  python Python/sam3_evaluate.py --checkpoint {checkpoint_dir / 'checkpoint.pt'} "
        f"--prompt {args.prompt} --dataset shrubs_val <val.json> <val images>"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
