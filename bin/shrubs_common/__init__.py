"""Model-agnostic evaluation metrics shared by the SAM 3 and Detectron2 pipelines."""

from .metrics import (
    METRIC_DEFINITIONS,
    Prediction,
    PredictionCache,
    best_iou_matches,
    compute_s_iou_matches,
    confidence_thresholds,
    evaluate_instance_iou,
    evaluate_instance_siou,
    evaluate_pixel_metrics,
    filter_instances,
    greedy_nms,
    harmonic_mean,
    mask_iou_matrix,
    one_to_one_matches,
    precision_recall_f1,
    run_predictor,
    validate_probability,
    write_sweep,
)
from .polygons import masks_to_features, write_window_masks

try:
    # rasterio is unavailable in some environments (e.g. Matterport's
    # mrcnn_tf2 conda env, which has a Windows DLL conflict between its old
    # TensorFlow/Keras stack and GDAL). masks_to_features/write_window_masks
    # above have no rasterio dependency, so they stay usable even then.
    from .tiling import iter_windows, read_rgb_window, starts
except ImportError:
    iter_windows = read_rgb_window = starts = None

__all__ = [
    "METRIC_DEFINITIONS",
    "Prediction",
    "PredictionCache",
    "best_iou_matches",
    "compute_s_iou_matches",
    "confidence_thresholds",
    "evaluate_instance_iou",
    "evaluate_instance_siou",
    "evaluate_pixel_metrics",
    "filter_instances",
    "greedy_nms",
    "harmonic_mean",
    "iter_windows",
    "mask_iou_matrix",
    "masks_to_features",
    "one_to_one_matches",
    "precision_recall_f1",
    "read_rgb_window",
    "run_predictor",
    "starts",
    "validate_probability",
    "write_sweep",
    "write_window_masks",
]
