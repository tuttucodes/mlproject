# backend/app/utils/preprocessing.py
"""
NIfTI preprocessing: load, normalize, crop, augment.
Handles all BraTS versions (2020-2024) transparently.
"""

import numpy as np
import logging
from pathlib import Path
from typing import Dict, Tuple

log = logging.getLogger(__name__)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  NIfTI I/O
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def validate_nifti(filepath: str) -> bool:
    try:
        import nibabel as nib
        img = nib.load(filepath)
        data = img.get_fdata()
        return data.ndim >= 3
    except Exception as e:
        log.error(f"Invalid NIfTI {filepath}: {e}")
        return False


def load_nifti(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
    import nibabel as nib
    img = nib.load(filepath)
    return img.get_fdata().astype(np.float32), img.affine


def save_nifti(data: np.ndarray, affine: np.ndarray, filepath: str):
    import nibabel as nib
    nib.save(nib.Nifti1Image(data.astype(np.float32), affine), filepath)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Modality auto-detection (all BraTS versions)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def detect_modalities(case_dir: str) -> Dict[str, str]:
    """
    Auto-detect modality files in any BraTS case directory.
    Returns: {'t1': path, 't1ce': path, 't2': path, 'flair': path, 'seg': path}

    Handles naming from BraTS 2020 (_t1, _t1ce, _t2, _flair)
    and BraTS 2023/24 (-t1n, -t1c, -t2w, -t2f).
    """
    d = Path(case_dir)
    niftis = sorted(d.glob("*.nii*"))
    result = {"t1": None, "t1ce": None, "t2": None, "flair": None, "seg": None}

    for f in niftis:
        n = f.name.lower()
        if "seg" in n:
            result["seg"] = str(f)
        elif any(n.endswith(x) for x in ["_t1ce.nii.gz", "_t1ce.nii", "-t1c.nii.gz", "-t1c.nii", "_t1gd.nii.gz"]):
            result["t1ce"] = str(f)
        elif any(n.endswith(x) for x in ["_flair.nii.gz", "_flair.nii", "-t2f.nii.gz", "-t2f.nii"]):
            result["flair"] = str(f)
        elif any(n.endswith(x) for x in ["_t1.nii.gz", "_t1.nii", "-t1n.nii.gz", "-t1n.nii"]):
            result["t1"] = str(f)
        elif any(n.endswith(x) for x in ["_t2.nii.gz", "_t2.nii", "-t2w.nii.gz", "-t2w.nii"]):
            result["t2"] = str(f)

    return result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Preprocessing pipeline
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _center_crop_pad(volume: np.ndarray, target: Tuple[int, int, int]) -> np.ndarray:
    """Center crop or zero-pad a 3D volume to target shape."""
    result = np.zeros(target, dtype=volume.dtype)
    slices_src, slices_dst = [], []
    for i in range(3):
        s, t = volume.shape[i], target[i]
        if s >= t:
            start = (s - t) // 2
            slices_src.append(slice(start, start + t))
            slices_dst.append(slice(0, t))
        else:
            pad = (t - s) // 2
            slices_src.append(slice(0, s))
            slices_dst.append(slice(pad, pad + s))
    result[slices_dst[0], slices_dst[1], slices_dst[2]] = \
        volume[slices_src[0], slices_src[1], slices_src[2]]
    return result


def _center_crop_pad_4d(volume: np.ndarray, target: Tuple[int, int, int]) -> np.ndarray:
    """Center crop/pad a (C, D, H, W) volume."""
    out = np.zeros((volume.shape[0], *target), dtype=volume.dtype)
    for c in range(volume.shape[0]):
        out[c] = _center_crop_pad(volume[c], target)
    return out


def preprocess_nifti(
    modality_paths: Dict[str, str],
    target_shape: Tuple[int, int, int] = (128, 128, 128),
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Full preprocessing:
    1. Load available modalities
    2. Z-score normalize (non-zero brain voxels)
    3. Center crop/pad to target_shape
    4. Stack into (4, D, H, W)
    """
    order = ["t1", "t1ce", "t2", "flair"]
    volumes = []
    affine = None
    meta = {"modalities_loaded": [], "original_shapes": {}}

    for mod in order:
        path = modality_paths.get(mod)
        if path and Path(path).exists():
            data, aff = load_nifti(path)
            if affine is None:
                affine = aff
            meta["original_shapes"][mod] = list(data.shape)
            meta["modalities_loaded"].append(mod)
            # Z-score on non-zero
            mask = data > 0
            if mask.sum() > 0:
                data[mask] = (data[mask] - data[mask].mean()) / (data[mask].std() + 1e-8)
            volumes.append(data)
        else:
            if volumes:
                volumes.append(np.zeros_like(volumes[0]))
            else:
                volumes.append(np.zeros(target_shape, dtype=np.float32))

    stacked = np.stack(volumes, axis=0)
    stacked = _center_crop_pad_4d(stacked, target_shape)
    if affine is None:
        affine = np.eye(4)
    meta["preprocessed_shape"] = list(stacked.shape)
    return stacked, affine, meta


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Tumor region statistics
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def extract_tumor_region(segmentation: np.ndarray) -> Dict:
    regions = {1: "Necrotic/Non-enhancing tumor", 2: "Peritumoral edema", 4: "GD-enhancing tumor"}
    total = 0
    stats = []
    for label, name in regions.items():
        count = int((segmentation == label).sum())
        total += count
        stats.append({
            "label": label, "name": name,
            "volume_mm3": round(float(count), 2),
            "volume_cm3": round(count / 1000.0, 4),
            "voxel_count": count,
        })
    for s in stats:
        s["percentage"] = round(s["voxel_count"] / total * 100, 2) if total > 0 else 0
    return {
        "total_tumor_volume_mm3": float(total),
        "total_tumor_volume_cm3": total / 1000.0,
        "regions": stats,
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Synthetic data (fallback for demo)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def generate_synthetic_brats(shape=(128, 128, 128), seed=42) -> Dict[str, np.ndarray]:
    np.random.seed(seed)
    center = np.array(shape) // 2
    coords = np.mgrid[:shape[0], :shape[1], :shape[2]]
    dist = np.sqrt(sum((c - ct) ** 2 for c, ct in zip(coords, center)))
    brain = dist < min(shape) * 0.4
    base = np.random.randn(*shape).astype(np.float32) * 0.3
    volumes = {}
    for mod, shift in [("t1", 0.8), ("t1ce", 1.0), ("t2", 0.9), ("flair", 0.7)]:
        v = base.copy() + shift
        v[~brain] = 0
        v += np.random.randn(*shape) * 0.1
        v[~brain] = 0
        volumes[mod] = v
    seg = np.zeros(shape, dtype=np.int32)
    tc = center + np.array([5, -3, 2])
    td = np.sqrt(sum((c - ct) ** 2 for c, ct in zip(coords, tc)))
    seg[td < 8] = 4
    seg[(td < 14) & (seg == 0)] = 1
    seg[(td < 22) & (seg == 0)] = 2
    volumes["seg"] = seg
    return volumes
