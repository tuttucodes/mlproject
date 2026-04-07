# backend/app/api/comparison.py
"""Novel Feature #3: Multi-case comparative analysis with clinical metrics."""

import numpy as np
import logging
from app.utils.preprocessing import load_nifti

logger = logging.getLogger(__name__)

REGION_NAMES = {1: "Necrotic tumor", 2: "Peritumoral edema", 4: "Enhancing tumor"}


def dice_coefficient(pred: np.ndarray, target: np.ndarray) -> float:
    intersection = np.logical_and(pred, target).sum()
    return float(2 * intersection / (pred.sum() + target.sum() + 1e-8))


def jaccard_index(pred: np.ndarray, target: np.ndarray) -> float:
    intersection = np.logical_and(pred, target).sum()
    union = np.logical_or(pred, target).sum()
    return float(intersection / (union + 1e-8))


def hausdorff_distance(pred: np.ndarray, target: np.ndarray) -> float:
    """Compute 95th percentile Hausdorff distance."""
    from scipy.ndimage import distance_transform_edt

    if pred.sum() == 0 or target.sum() == 0:
        return float("inf")

    dt_pred = distance_transform_edt(~pred)
    dt_target = distance_transform_edt(~target)

    # Surface distances
    pred_surface = pred & ~np.roll(pred, 1, axis=0)  # Simplified surface
    target_surface = target & ~np.roll(target, 1, axis=0)

    if pred_surface.sum() == 0 or target_surface.sum() == 0:
        return float("inf")

    d1 = dt_target[pred_surface]
    d2 = dt_pred[target_surface]

    hd95 = max(np.percentile(d1, 95), np.percentile(d2, 95))
    return float(hd95)


def volume_similarity(vol1: float, vol2: float) -> float:
    """Volume similarity metric (1 = identical volumes)."""
    return 1.0 - abs(vol1 - vol2) / (vol1 + vol2 + 1e-8)


def run_comparison(seg1_path: str, seg2_path: str) -> dict:
    """Compare two segmentation masks."""
    seg1_data, _ = load_nifti(seg1_path)
    seg2_data, _ = load_nifti(seg2_path)

    seg1 = seg1_data.astype(np.int32)
    seg2 = seg2_data.astype(np.int32)

    # Handle shape mismatch
    if seg1.shape != seg2.shape:
        min_shape = tuple(min(s1, s2) for s1, s2 in zip(seg1.shape, seg2.shape))
        seg1 = seg1[: min_shape[0], : min_shape[1], : min_shape[2]]
        seg2 = seg2[: min_shape[0], : min_shape[1], : min_shape[2]]

    # Overall metrics (whole tumor)
    wt1 = seg1 > 0
    wt2 = seg2 > 0
    overall_dice = dice_coefficient(wt1, wt2)
    overall_jaccard = jaccard_index(wt1, wt2)
    overall_hausdorff = hausdorff_distance(wt1, wt2)

    # Per-region metrics
    region_comparisons = []
    for label, name in REGION_NAMES.items():
        r1 = seg1 == label
        r2 = seg2 == label

        v1 = float(r1.sum()) / 1000.0
        v2 = float(r2.sum()) / 1000.0

        region_comparisons.append({
            "region_name": name,
            "dice_coefficient": round(dice_coefficient(r1, r2), 4),
            "jaccard_index": round(jaccard_index(r1, r2), 4),
            "hausdorff_distance_mm": round(hausdorff_distance(r1, r2), 2),
            "volume_similarity": round(volume_similarity(v1, v2), 4),
            "volume_diff_cm3": round(v2 - v1, 4),
        })

    # Volume change
    total_v1 = float(wt1.sum()) / 1000.0
    total_v2 = float(wt2.sum()) / 1000.0
    vol_change = total_v2 - total_v1
    vol_change_pct = (vol_change / (total_v1 + 1e-8)) * 100

    # Progression status
    if vol_change_pct > 25:
        progression = "Progressing (significant growth)"
    elif vol_change_pct > 10:
        progression = "Progressing (moderate growth)"
    elif vol_change_pct < -25:
        progression = "Regressing (significant shrinkage)"
    elif vol_change_pct < -10:
        progression = "Regressing (moderate shrinkage)"
    else:
        progression = "Stable"

    return {
        "overall_dice": round(overall_dice, 4),
        "overall_jaccard": round(overall_jaccard, 4),
        "overall_hausdorff_mm": round(overall_hausdorff, 2),
        "region_comparisons": region_comparisons,
        "volume_change_cm3": round(vol_change, 4),
        "volume_change_percent": round(vol_change_pct, 2),
        "progression_status": progression,
        "diff_heatmap_file": None,
    }
