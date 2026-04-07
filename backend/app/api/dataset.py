# backend/app/api/dataset.py
"""BraTS dataset API: Kaggle download, stats, sample cases."""

import os
import json
import shutil
import logging
import numpy as np
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

BRATS_STATS = {
    "GLI": {
        "subset": "BraTS-GLI (Adult Glioma)", "total_cases": 1621,
        "modalities": ["T1", "T1CE", "T2", "T2-FLAIR"],
        "voxel_dimensions": [240, 240, 155], "mean_tumor_volume_cm3": 42.5,
        "label_distribution": {"Background": 98.2, "Necrotic (1)": 0.4, "Edema (2)": 1.0, "Enhancing (4)": 0.4},
        "intensity_stats": {
            "T1": {"mean": 452.3, "std": 312.8, "min": 0, "max": 4095},
            "T1CE": {"mean": 487.1, "std": 345.2, "min": 0, "max": 4095},
            "T2": {"mean": 512.7, "std": 389.4, "min": 0, "max": 4095},
            "FLAIR": {"mean": 478.9, "std": 356.1, "min": 0, "max": 4095},
        },
    },
    "PED": {
        "subset": "BraTS-PED (Pediatric)", "total_cases": 257,
        "modalities": ["T1", "T1CE", "T2", "T2-FLAIR"],
        "voxel_dimensions": [240, 240, 155], "mean_tumor_volume_cm3": 35.2,
        "label_distribution": {"Background": 98.5, "Necrotic (1)": 0.3, "Edema (2)": 0.8, "Enhancing (4)": 0.4},
        "intensity_stats": {
            "T1": {"mean": 430.1, "std": 298.5, "min": 0, "max": 4095},
            "T1CE": {"mean": 465.7, "std": 325.3, "min": 0, "max": 4095},
            "T2": {"mean": 498.2, "std": 372.1, "min": 0, "max": 4095},
            "FLAIR": {"mean": 461.3, "std": 340.8, "min": 0, "max": 4095},
        },
    },
    "MEN": {
        "subset": "BraTS-MEN-RT (Meningioma)", "total_cases": 500,
        "modalities": ["T1", "T1CE", "T2", "T2-FLAIR"],
        "voxel_dimensions": [240, 240, 155], "mean_tumor_volume_cm3": 28.7,
        "label_distribution": {"Background": 98.8, "Necrotic (1)": 0.2, "Edema (2)": 0.7, "Enhancing (4)": 0.3},
        "intensity_stats": {
            "T1": {"mean": 445.0, "std": 305.2, "min": 0, "max": 4095},
            "T1CE": {"mean": 480.3, "std": 338.7, "min": 0, "max": 4095},
            "T2": {"mean": 505.4, "std": 381.9, "min": 0, "max": 4095},
            "FLAIR": {"mean": 472.1, "std": 348.5, "min": 0, "max": 4095},
        },
    },
}


def get_brats_stats(subset: str = "GLI") -> dict:
    return BRATS_STATS.get(subset.upper(), BRATS_STATS["GLI"])


def _kaggle_ok() -> bool:
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi(); api.authenticate()
        return True
    except Exception:
        return False


def download_sample_case(output_dir="/tmp/brain_tumor_samples") -> str:
    """Download one real BraTS case from Kaggle, or generate synthetic."""
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)

    cached = list(out.rglob("*flair*nii*")) + list(out.rglob("*t2f*nii*"))
    if cached:
        return str(cached[0])

    if _kaggle_ok():
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
            api = KaggleApi(); api.authenticate()
            api.dataset_download_files("nguyenthanhkhanh/brats2024-small-dataset", path=str(out), unzip=True)
            found = list(out.rglob("*flair*nii*")) + list(out.rglob("*t2f*nii*"))
            if found:
                return str(found[0])
        except Exception as e:
            log.warning(f"Kaggle sample download failed: {e}")

    from app.utils.preprocessing import generate_synthetic_brats, save_nifti
    p = out / "sample_flair.nii.gz"
    if not p.exists():
        save_nifti(generate_synthetic_brats()["flair"], np.eye(4), str(p))
    return str(p)
