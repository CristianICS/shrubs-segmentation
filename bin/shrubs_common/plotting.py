"""
F1-versus-confidence plotting and the shared evaluation report.

Kept out of `metrics.py` so importers that only need the numeric metrics (for
example SAM 3's in-tile NMS at inference time) do not pull in matplotlib.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Mapping

from matplotlib import pyplot as plt

from .evaluation import EvaluationSettings
from .metrics import (
    PredictionCache, confidence_thresholds, evaluate_instance_iou, evaluate_instance_siou,
    evaluate_pixel_metrics, validate_probability, write_sweep,
)


def plot_performance(
    csv_files: Mapping[float, Mapping[str, Path]],
    out_dir: Path,
    show: bool,
    file_prefix: str = "evaluation",
) -> None:
    """Save one F1-vs-confidence figure per evaluation threshold.

    Figures are written as `<file_prefix>_threshold_<pct>.png`, so plots built
    from different IoU definitions can share one folder. Without `show`, use the non-GUI Agg backend so saving never depends on a
    Qt/Tk toolkit. With `show`, keep matplotlib's default backend so each
    figure opens in a window (matplotlib falls back to Agg with a warning
    when no display is available).
    """
    if not show:
        plt.switch_backend("Agg")
    out_dir.mkdir(parents=True, exist_ok=True)
    for evaluation_threshold, metric_files in csv_files.items():
        fig, ax = plt.subplots(figsize=(7, 5))
        for label, csv_path in metric_files.items():
            thresholds_pct, f1_pct = [], []
            with csv_path.open(newline="", encoding="utf-8") as stream:
                for row in csv.DictReader(stream):
                    thresholds_pct.append(float(row["confidence_threshold"]) * 100)
                    f1_pct.append(float(row["f1"]) * 100)
            ax.plot(thresholds_pct, f1_pct, marker="o", markersize=4, linewidth=1.5, label=label)
        percent = round(evaluation_threshold * 100)
        ax.set_title(f"Evaluation threshold = {percent}%")
        ax.set_xlabel("Prediction-confidence threshold (%)")
        ax.set_ylabel("F1-score (%)")
        ax.set_xlim(0, 100); ax.set_ylim(0, 100)
        ax.set_xticks(range(0, 101, 20)); ax.set_yticks(range(0, 101, 20))
        ax.grid(alpha=0.25); ax.legend(title="Evaluation metric"); fig.tight_layout()
        output = out_dir / f"{file_prefix}_threshold_{percent}.png"
        fig.savefig(output, dpi=300, bbox_inches="tight")
        if show: plt.show()
        plt.close(fig)
        print(f"Figure saved to: {output}")


def run_evaluation_report(
    ground_truths: PredictionCache,
    predictions: PredictionCache,
    output_dir: Path,
    settings: EvaluationSettings,
) -> None:
    """Compute and save pixel/IoU/S-IoU metrics and F1-vs-confidence plots.

    Shared by `detectron2_evaluate.py` and `sam3_evaluate.py`, which both pass
    the same `EvaluationSettings` from `shrubs_common.evaluation`.

    Every file goes flat into `output_dir`. The paper IoU definition writes
    `iou_paper_threshold_<pct>.csv` and `evaluation_paper_threshold_<pct>.png`
    so its results never overwrite the standard `iou_threshold_<pct>.csv` and
    `evaluation_threshold_<pct>.png`. S-IoU does not depend on the IoU
    definition, so `siou_threshold_<pct>.csv` is shared by both.
    """
    is_paper = settings.iou_metric_definition == "paper"
    iou_prefix = "iou_paper" if is_paper else "iou"
    iou_label = "IoU (paper)" if is_paper else "IoU"
    plot_prefix = "evaluation_paper" if is_paper else "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)

    pixel = evaluate_pixel_metrics(
        ground_truths, predictions, class_id=settings.class_id,
        confidence_threshold=settings.pixel_confidence,
    )
    print(
        f"Pixel metrics at confidence {settings.pixel_confidence:.2f}: "
        f"P={pixel['precision']:.4f}, R={pixel['recall']:.4f}, F1={pixel['f1']:.4f}, "
        f"TP={pixel['tp_pixels']}, FP={pixel['fp_pixels']}, FN={pixel['fn_pixels']}"
    )

    thresholds = confidence_thresholds(settings.confidence_thresholds)
    plot_inputs: dict[float, dict[str, Path]] = {}
    for evaluation_threshold in settings.evaluation_thresholds:
        evaluation_threshold = validate_probability(
            evaluation_threshold, "evaluation threshold", allow_zero=False
        )
        percent = round(evaluation_threshold * 100)
        iou_csv = output_dir / f"{iou_prefix}_threshold_{percent}.csv"
        siou_csv = output_dir / f"siou_threshold_{percent}.csv"

        write_sweep(
            iou_csv, metric_name="IoU", evaluation_threshold=evaluation_threshold,
            thresholds=thresholds,
            evaluator=lambda confidence, t=evaluation_threshold: evaluate_instance_iou(
                ground_truths, predictions, class_id=settings.class_id,
                confidence_threshold=confidence, iou_threshold=t,
                metric_definition=settings.iou_metric_definition,
            ),
        )
        write_sweep(
            siou_csv, metric_name="S-IoU", evaluation_threshold=evaluation_threshold,
            thresholds=thresholds,
            evaluator=lambda confidence, t=evaluation_threshold: evaluate_instance_siou(
                ground_truths, predictions, class_id=settings.class_id,
                confidence_threshold=confidence, siou_threshold=t,
                minimum_intersection_pixels=settings.minimum_intersection_pixels,
            ),
        )
        plot_inputs[evaluation_threshold] = {iou_label: iou_csv, "S-IoU": siou_csv}

    plot_performance(plot_inputs, output_dir, settings.show_plots, file_prefix=plot_prefix)
