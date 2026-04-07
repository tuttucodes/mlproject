# backend/app/api/uncertainty.py
"""Novel Feature #2: MC-Dropout uncertainty quantification."""

import time
import numpy as np
import logging
from pathlib import Path
from typing import Dict

from app.utils.preprocessing import preprocess_nifti, save_nifti

logger = logging.getLogger(__name__)


def run_uncertainty(
    modality_paths: Dict[str, str],
    result_dir: str,
    model_manager,
    n_iterations: int = 20,
) -> dict:
    """
    MC-Dropout Uncertainty Quantification:
    1. Run inference N times with dropout enabled
    2. Compute voxel-wise prediction variance
    3. Generate uncertainty heatmap
    4. Identify high-uncertainty regions
    """
    start = time.time()

    volume, affine, metadata = preprocess_nifti(modality_paths)

    # Collect predictions from multiple forward passes
    predictions = []
    for i in range(n_iterations):
        probs = model_manager.segmentation.predict_probabilities(volume)
        predictions.append(probs)
        if (i + 1) % 5 == 0:
            logger.info(f"MC-Dropout iteration {i + 1}/{n_iterations}")

    # Stack: (N, C, D, H, W)
    all_probs = np.stack(predictions, axis=0)

    # Mean prediction
    mean_probs = all_probs.mean(axis=0)  # (C, D, H, W)

    # Predictive entropy (uncertainty)
    entropy = -np.sum(mean_probs * np.log(mean_probs + 1e-8), axis=0)  # (D, H, W)

    # Mutual information (epistemic uncertainty)
    expected_entropy = -np.mean(
        np.sum(all_probs * np.log(all_probs + 1e-8), axis=1), axis=0
    )
    mutual_info = entropy - expected_entropy  # (D, H, W)

    # Prediction variance
    pred_variance = all_probs.var(axis=0).mean(axis=0)  # (D, H, W)

    # Normalize uncertainty to [0, 1]
    if entropy.max() > 0:
        uncertainty_map = entropy / (entropy.max() + 1e-8)
    else:
        uncertainty_map = entropy

    # Statistics
    mean_unc = float(uncertainty_map.mean())
    max_unc = float(uncertainty_map.max())

    # High uncertainty regions (> 0.5 threshold)
    high_unc_mask = uncertainty_map > 0.5
    high_unc_volume = float(high_unc_mask.sum()) / 1000.0  # cm³
    total_nonzero = (uncertainty_map > 0).sum()
    high_unc_pct = float(high_unc_mask.sum() / (total_nonzero + 1e-8) * 100)

    # Save uncertainty heatmap
    result_path = str(Path(result_dir) / "uncertainty_map.nii.gz")
    save_nifti(uncertainty_map, affine, result_path)

    job_id = Path(result_dir).name
    elapsed = time.time() - start

    # Clinical interpretation
    if mean_unc < 0.15:
        interpretation = "Low overall uncertainty — model is highly confident in segmentation boundaries."
    elif mean_unc < 0.35:
        interpretation = "Moderate uncertainty — some ambiguous regions present, particularly at tumor margins. Radiologist review recommended for boundary regions."
    else:
        interpretation = "High uncertainty — significant ambiguity in segmentation. Manual review strongly recommended. Consider acquiring higher-quality MRI or additional sequences."

    return {
        "mean_uncertainty": round(mean_unc, 4),
        "max_uncertainty": round(max_unc, 4),
        "high_uncertainty_volume_cm3": round(high_unc_volume, 4),
        "high_uncertainty_percentage": round(high_unc_pct, 2),
        "uncertainty_heatmap_file": result_path,
        "n_mc_iterations": n_iterations,
        "voxel_wise_stats": {
            "mean_entropy": round(float(entropy.mean()), 4),
            "max_entropy": round(float(entropy.max()), 4),
            "mean_mutual_info": round(float(mutual_info.mean()), 6),
            "mean_pred_variance": round(float(pred_variance.mean()), 6),
            "inference_time_seconds": round(elapsed, 2),
        },
        "download_url": f"/api/results/{job_id}/uncertainty_map.nii.gz",
        "clinical_interpretation": interpretation,
    }
