"""Compare SAM 3 text prompts on the JuniperMapper Photo Interpretation Val split.

Scores the domain-adapted checkpoint (`model_best.pt`) once per prompt with
exactly the in-training validation of `sam3_domain_adapt.py`: same Hydra
config, val transforms, bfloat16 AMP and COCO mask/box evaluators, run through
the upstream trainer's `mode=val`. As during training, the prompt is the
category name written into the converted val.json, so the `shrub` row
reproduces the scores stored in `model_best.json`.

Run inside WSL2 (conda env `sam3_shrubs`), like the other SAM 3 scripts:

    python Python/sam3_prompt_study.py

Each prompt's COCO dumps and logs go under `--work-dir/<prompt>`; prompts that
already have a `logs/val_stats.json` there are not re-run (pass --overwrite to
re-score them). The summary CSV has one row per prompt.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BIN_DIR = PROJECT_ROOT / "bin"
if str(BIN_DIR) not in sys.path:
    sys.path.insert(0, str(BIN_DIR))

from sam3_domain_adapt import (  # noqa: E402
    CONFIG_DIR, CONFIG_NAME, DEFAULT_VAL_IMAGES, DEFAULT_VAL_JSON, JM_DIR,
    _default_bpe_path, _free_port, _to_override, existing_dir, existing_file,
)
from shrubs_sam3 import prepare_juniper_coco  # noqa: E402

DEFAULT_CHECKPOINT = PROJECT_ROOT / "data/logs/sam3/domain_adapt/checkpoints/model_best.pt"
DEFAULT_PROMPTS = (
    "shrub", "bush", "brown shrub", "burned shrub", "dry shrub", "low shrub",
    "mediterranean shrub", "small bush", "small shrub", "woody shrub",
)
METER_PREFIX = "Meters_train/val_shrubs_val/"
METRICS = {
    "mask_AP50": "segmentation/coco_eval_segm_AP_50",
    "mask_mAP_50_95": "segmentation/coco_eval_segm_AP",
    "mask_AP75": "segmentation/coco_eval_segm_AP_75",
    "mask_AR100": "segmentation/coco_eval_segm_AR_maxDets@100",
    "box_AP50": "detection/coco_eval_bbox_AP_50",
    "box_mAP_50_95": "detection/coco_eval_bbox_AP",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Score SAM 3 text prompts on the JuniperMapper PI Val split."
    )
    parser.add_argument(
        "--checkpoint", type=existing_file, default=DEFAULT_CHECKPOINT,
        help="Fine-tuned SAM 3 checkpoint to score (default: %(default)s).",
    )
    parser.add_argument("--prompts", nargs="+", default=list(DEFAULT_PROMPTS))
    parser.add_argument("--val-json", type=existing_file, default=DEFAULT_VAL_JSON)
    parser.add_argument("--val-images", type=existing_dir, default=DEFAULT_VAL_IMAGES)
    parser.add_argument(
        "--work-dir", type=Path, default=PROJECT_ROOT / "data/logs/sam3/prompt_study",
        help="Per-prompt annotations, COCO dumps and logs (default: %(default)s).",
    )
    parser.add_argument(
        "--output", type=Path,
        default=PROJECT_ROOT / "results/evaluation/model_sam3/prompt_study_pi_val.csv",
    )
    parser.add_argument("--bpe-path", type=existing_file, default=None)
    parser.add_argument("--overwrite", action="store_true", help="Re-score prompts that already have results.")
    return parser


def check_inputs(args: argparse.Namespace) -> None:
    """Fail early if the checkpoint or the validation split is missing.

    argparse only applies `existing_file`/`existing_dir` to values typed on the
    command line, not to the Path defaults, so those are checked here. A
    missing checkpoint would otherwise only fail deep inside the trainer,
    because `_link_checkpoint` happily symlinks to a file that doesn't exist.
    """
    if not Path(args.checkpoint).is_file():
        raise FileNotFoundError(
            f"Fine-tuned SAM 3 checkpoint not found at '{args.checkpoint}'. "
            "Run 'sam3_domain_adapt.py' first, or pass the trained "
            "'model_best.pt' with --checkpoint."
        )

    inputs = (("--val-json", args.val_json), ("--val-images", args.val_images))

    # Only relevant when the inputs point inside the JuniperMapper database
    if not JM_DIR.exists() and any(Path(path).resolve().is_relative_to(JM_DIR) for _, path in inputs):
        raise FileNotFoundError(
            f"JuniperMapper database not found at '{JM_DIR}'. "
            "Place a valid database copy inside the 'data' directory."
        )

    missing = [f"  - {name}: {path}" for name, path in inputs if not Path(path).exists()]
    if missing:
        raise FileNotFoundError(
            "The following validation inputs are missing:\n" + "\n".join(missing)
        )


def _link_checkpoint(checkpoint: Path, save_dir: Path) -> None:
    """Expose `checkpoint` as `<save_dir>/checkpoint.pt`, the file the trainer
    resumes from. A symlink avoids copying the ~3.6 GB file; fall back to a
    copy where symlinks aren't allowed (e.g. native Windows without
    developer mode)."""
    save_dir.mkdir(parents=True, exist_ok=True)
    target = save_dir / "checkpoint.pt"
    if target.is_symlink() or target.exists():
        if target.resolve() == checkpoint.resolve():
            return
        target.unlink()
    try:
        target.symlink_to(checkpoint)
    except OSError:
        shutil.copyfile(checkpoint, target)


def score_prompt(prompt: str, args: argparse.Namespace, bpe_path: Path) -> dict[str, float]:
    import torch
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from hydra.utils import instantiate

    output = args.work_dir.resolve() / prompt.replace(" ", "_")
    stats_path = output / "logs" / "val_stats.json"
    if args.overwrite or not stats_path.is_file():
        if stats_path.is_file():
            stats_path.unlink()
        val_json = output / "annotations" / "val.json"
        prepare_juniper_coco(args.val_json, val_json, prompt=prompt)
        save_dir = args.work_dir.resolve() / "checkpoint"
        _link_checkpoint(args.checkpoint, save_dir)

        overrides = [
            # The train split is never built in val mode; point it at val.
            f"paths.train_ann_file={_to_override(val_json)}",
            f"paths.train_img_folder={_to_override(args.val_images)}",
            f"paths.val_ann_file={_to_override(val_json)}",
            f"paths.val_img_folder={_to_override(args.val_images)}",
            f"paths.experiment_log_dir={_to_override(output)}",
            f"paths.bpe_path={_to_override(bpe_path)}",
            "trainer.mode=val",
            "trainer.skip_saving_ckpts=true",
            f"trainer.checkpoint.save_dir={_to_override(save_dir)}",
            # Every weight comes from the fine-tuned checkpoint, so the
            # release checkpoint is never needed.
            "trainer.model.load_from_HF=false",
        ]
        if sys.platform == "win32":
            overrides.append("trainer.distributed.backend=gloo")

        GlobalHydra.instance().clear()
        initialize_config_dir(config_dir=str(CONFIG_DIR), version_base="1.2")
        cfg = compose(config_name=CONFIG_NAME, overrides=overrides)

        os.environ["MASTER_PORT"] = str(_free_port())
        trainer = instantiate(cfg.trainer, _recursive_=False)
        trainer.run()
        del trainer
        # The trainer initializes the default process group, which can't be
        # initialized twice in one process.
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        torch.cuda.empty_cache()

    lines = [line for line in stats_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    meters = json.loads(lines[-1])
    return {name: round(meters[METER_PREFIX + key], 4) for name, key in METRICS.items()}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    check_inputs(args)
    bpe_path = args.bpe_path or _default_bpe_path()

    import torch
    from sam3.train.utils.train_utils import register_omegaconf_resolvers
    from shrubs_sam3.utils import _NUMPY_SCALAR_GLOBALS

    # Fine-tuned checkpoints store their best-epoch meters as numpy scalars.
    torch.serialization.add_safe_globals(_NUMPY_SCALAR_GLOBALS)
    register_omegaconf_resolvers()
    os.environ.update(MASTER_ADDR="127.0.0.1", RANK="0", LOCAL_RANK="0", WORLD_SIZE="1")

    rows = []
    for prompt in args.prompts:
        row = {"prompt": prompt, **score_prompt(prompt, args, bpe_path)}
        print(json.dumps(row), flush=True)
        rows.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Summary written to: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
