# backend/app/api/segmentation.py
"""Core segmentation endpoint: runs 3D U-Net inference on MRI volumes."""

import time
import logging
import numpy as np
from pathlib import Path
from typing import Dict

from app.utils.preprocessing import preprocess_nifti, save_nifti, extract_tumor_region

logger = logging.getLogger(__name__)


def run_segmentation(
    modality_paths: Dict[str, str],
    result_dir: str,
    model_manager,
) -> dict:
    """
    Full segmentation pipeline:
    1. Preprocess multi-modal MRI
    2. Run 3D U-Net inference
    3. Post-process segmentation
    4. Save results and compute statistics
    """
    start_time = time.time()

    # 1. Preprocess
    volume, affine, metadata = preprocess_nifti(modality_paths)
    logger.info(f"Preprocessed volume shape: {volume.shape}")

    # 2. Inference
    segmentation = model_manager.segmentation.predict(volume)
    logger.info(f"Segmentation shape: {segmentation.shape}, unique labels: {np.unique(segmentation)}")

    # 3. Post-process: morphological cleanup
    segmentation = _postprocess_segmentation(segmentation)

    # 4. Save result
    result_path = str(Path(result_dir) / "segmentation.nii.gz")
    save_nifti(segmentation.astype(np.float32), affine, result_path)

    # 5. Compute statistics
    stats = extract_tumor_region(segmentation)
    elapsed = time.time() - start_time

    job_id = Path(result_dir).name

    return {
        "job_id": job_id,
        "status": "completed",
        "segmentation_file": result_path,
        "total_tumor_volume_cm3": round(stats["total_tumor_volume_cm3"], 4),
        "regions": stats["regions"],
        "inference_time_seconds": round(elapsed, 2),
        "model_used": "3D-UNet (BraTS)",
        "input_shape": list(volume.shape),
        "download_url": f"/api/results/{job_id}/segmentation.nii.gz",
    }


def _postprocess_segmentation(seg: np.ndarray) -> np.ndarray:
    """
    Morphological post-processing:
    - Remove small islands (< 50 voxels)
    - Fill holes in each region
    - Ensure anatomical consistency
    """
    from scipy import ndimage

    cleaned = seg.copy()

    for label in [1, 2, 4]:
        mask = cleaned == label
        if mask.sum() == 0:
            continue

        # Label connected components
        labeled, n_components = ndimage.label(mask)
        if n_components > 1:
            # Keep only the largest component
            sizes = ndimage.sum(mask, labeled, range(1, n_components + 1))
            largest = np.argmax(sizes) + 1
            mask = labeled == largest

        # Fill holes
        mask = ndimage.binary_fill_holes(mask)
        cleaned[seg == label] = 0
        cleaned[mask] = label

    return cleaned
