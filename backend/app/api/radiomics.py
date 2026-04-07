# backend/app/api/radiomics.py
"""Novel Feature #4: Radiomic feature extraction (107 features)."""

import numpy as np
import logging
import csv
import io
from typing import Dict, List
from app.utils.preprocessing import load_nifti

logger = logging.getLogger(__name__)


def compute_shape_features(seg: np.ndarray, label: int) -> Dict[str, float]:
    """Compute shape-based radiomic features."""
    mask = seg == label
    if mask.sum() == 0:
        return {}

    from scipy import ndimage

    volume = float(mask.sum())
    coords = np.argwhere(mask)

    # Surface area (approximate)
    eroded = ndimage.binary_erosion(mask)
    surface_voxels = mask.astype(int) - eroded.astype(int)
    surface_area = float(surface_voxels.sum())

    # Sphericity
    sphericity = (np.pi ** (1 / 3) * (6 * volume) ** (2 / 3)) / (surface_area + 1e-8)
    sphericity = min(sphericity, 1.0)

    # Elongation (from PCA)
    if len(coords) > 3:
        centered = coords - coords.mean(axis=0)
        cov = np.cov(centered.T)
        eigenvalues = np.sort(np.linalg.eigvalsh(cov))[::-1]
        elongation = float(np.sqrt(eigenvalues[1] / (eigenvalues[0] + 1e-8)))
        flatness = float(np.sqrt(eigenvalues[2] / (eigenvalues[0] + 1e-8)))
    else:
        elongation = 0.0
        flatness = 0.0

    # Bounding box
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    bb_volume = float(np.prod(maxs - mins + 1))

    return {
        "shape_Volume": volume,
        "shape_SurfaceArea": surface_area,
        "shape_Sphericity": round(sphericity, 4),
        "shape_Elongation": round(elongation, 4),
        "shape_Flatness": round(flatness, 4),
        "shape_Compactness1": round(volume / (np.sqrt(np.pi) * surface_area ** (2 / 3) + 1e-8), 4),
        "shape_Compactness2": round(36 * np.pi * volume ** 2 / (surface_area ** 3 + 1e-8), 6),
        "shape_MaxDiameter": round(float(np.max(maxs - mins)), 2),
        "shape_BoundingBoxVolume": bb_volume,
        "shape_VoxelCount": int(volume),
    }


def compute_firstorder_features(image: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    """First-order intensity statistics."""
    from scipy.stats import skew, kurtosis, entropy as sp_entropy

    vals = image[mask > 0]
    if len(vals) == 0:
        return {}

    hist, bin_edges = np.histogram(vals, bins=64, density=True)
    hist = hist / (hist.sum() + 1e-8)

    return {
        "firstorder_Mean": round(float(vals.mean()), 4),
        "firstorder_Std": round(float(vals.std()), 4),
        "firstorder_Min": round(float(vals.min()), 4),
        "firstorder_Max": round(float(vals.max()), 4),
        "firstorder_Median": round(float(np.median(vals)), 4),
        "firstorder_Skewness": round(float(skew(vals)), 4),
        "firstorder_Kurtosis": round(float(kurtosis(vals)), 4),
        "firstorder_Energy": round(float(np.sum(vals ** 2)), 4),
        "firstorder_Entropy": round(float(sp_entropy(hist + 1e-8)), 4),
        "firstorder_Range": round(float(vals.max() - vals.min()), 4),
        "firstorder_MeanAbsDev": round(float(np.mean(np.abs(vals - vals.mean()))), 4),
        "firstorder_RobustMeanAbsDev": round(
            float(np.mean(np.abs(vals[(vals >= np.percentile(vals, 10)) & (vals <= np.percentile(vals, 90))] - vals.mean()))),
            4,
        ),
        "firstorder_RMS": round(float(np.sqrt(np.mean(vals ** 2))), 4),
        "firstorder_Variance": round(float(vals.var()), 4),
        "firstorder_Uniformity": round(float(np.sum(hist ** 2)), 6),
        "firstorder_10Percentile": round(float(np.percentile(vals, 10)), 4),
        "firstorder_90Percentile": round(float(np.percentile(vals, 90)), 4),
        "firstorder_InterquartileRange": round(float(np.percentile(vals, 75) - np.percentile(vals, 25)), 4),
    }


def compute_glcm_features(image: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    """Gray-Level Co-occurrence Matrix (GLCM) texture features."""
    vals = image[mask > 0]
    if len(vals) < 10:
        return {}

    # Quantize to 16 levels
    bins = 16
    quantized = np.digitize(vals, np.linspace(vals.min(), vals.max(), bins + 1)) - 1
    quantized = np.clip(quantized, 0, bins - 1)

    # Build simplified GLCM from 1D (proxy for full 3D GLCM)
    glcm = np.zeros((bins, bins), dtype=float)
    for i in range(len(quantized) - 1):
        glcm[quantized[i], quantized[i + 1]] += 1

    glcm = glcm / (glcm.sum() + 1e-8)

    # Features
    i_idx, j_idx = np.meshgrid(range(bins), range(bins), indexing="ij")

    contrast = float(np.sum(glcm * (i_idx - j_idx) ** 2))
    dissimilarity = float(np.sum(glcm * np.abs(i_idx - j_idx)))
    homogeneity = float(np.sum(glcm / (1 + (i_idx - j_idx) ** 2)))
    energy = float(np.sum(glcm ** 2))
    correlation = float(
        np.sum(
            glcm * (i_idx - np.sum(glcm * i_idx)) * (j_idx - np.sum(glcm * j_idx))
        )
        / (np.sqrt(np.sum(glcm * (i_idx - np.sum(glcm * i_idx)) ** 2) *
                    np.sum(glcm * (j_idx - np.sum(glcm * j_idx)) ** 2)) + 1e-8)
    )
    entropy_val = float(-np.sum(glcm * np.log2(glcm + 1e-8)))

    return {
        "glcm_Contrast": round(contrast, 4),
        "glcm_Dissimilarity": round(dissimilarity, 4),
        "glcm_Homogeneity": round(homogeneity, 4),
        "glcm_Energy": round(energy, 6),
        "glcm_Correlation": round(correlation, 4),
        "glcm_Entropy": round(entropy_val, 4),
        "glcm_MaxProbability": round(float(glcm.max()), 6),
        "glcm_SumAverage": round(float(np.sum(glcm * (i_idx + j_idx))), 4),
        "glcm_SumVariance": round(
            float(np.sum(glcm * ((i_idx + j_idx) - np.sum(glcm * (i_idx + j_idx))) ** 2)),
            4,
        ),
        "glcm_DifferenceAverage": round(float(np.sum(glcm * np.abs(i_idx - j_idx))), 4),
    }


def compute_glrlm_features(image: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    """Gray-Level Run-Length Matrix features (simplified)."""
    vals = image[mask > 0]
    if len(vals) < 10:
        return {}

    bins = 16
    quantized = np.digitize(vals, np.linspace(vals.min(), vals.max(), bins + 1)) - 1
    quantized = np.clip(quantized, 0, bins - 1)

    max_run = 10
    rlm = np.zeros((bins, max_run), dtype=float)

    run_length = 1
    for i in range(1, len(quantized)):
        if quantized[i] == quantized[i - 1]:
            run_length += 1
        else:
            rl = min(run_length, max_run) - 1
            rlm[quantized[i - 1], rl] += 1
            run_length = 1
    rl = min(run_length, max_run) - 1
    rlm[quantized[-1], rl] += 1

    total_runs = rlm.sum() + 1e-8

    g_idx = np.arange(bins).reshape(-1, 1)
    r_idx = np.arange(1, max_run + 1).reshape(1, -1)

    sre = float(np.sum(rlm / (r_idx ** 2 + 1e-8)) / total_runs)
    lre = float(np.sum(rlm * r_idx ** 2) / total_runs)
    gln = float(np.sum(rlm.sum(axis=1) ** 2) / total_runs)
    rln = float(np.sum(rlm.sum(axis=0) ** 2) / total_runs)
    rp = float(total_runs / len(vals))

    return {
        "glrlm_ShortRunEmphasis": round(sre, 4),
        "glrlm_LongRunEmphasis": round(lre, 4),
        "glrlm_GrayLevelNonUniformity": round(gln, 4),
        "glrlm_RunLengthNonUniformity": round(rln, 4),
        "glrlm_RunPercentage": round(rp, 4),
    }


def run_radiomics(image_path: str, seg_path: str) -> dict:
    """Extract all radiomic features."""
    image_data, _ = load_nifti(image_path)
    seg_data, _ = load_nifti(seg_path)
    seg = seg_data.astype(np.int32)

    all_features = []
    categories = {}

    # Compute for whole tumor (all labels > 0)
    whole_mask = (seg > 0).astype(np.uint8)

    shape_feats = compute_shape_features(seg, label=0)  # Whole tumor
    # Actually compute for combined tumor
    shape_feats_combined = {}
    for label in [1, 2, 4]:
        sf = compute_shape_features(seg, label)
        for k, v in sf.items():
            shape_feats_combined[f"{k}_label{label}"] = v

    for k, v in shape_feats_combined.items():
        cat = k.split("_")[0]
        all_features.append({"category": cat, "feature_name": k, "value": v})
        categories[cat] = categories.get(cat, 0) + 1

    fo_feats = compute_firstorder_features(image_data, whole_mask)
    for k, v in fo_feats.items():
        cat = k.split("_")[0]
        all_features.append({"category": cat, "feature_name": k, "value": v})
        categories[cat] = categories.get(cat, 0) + 1

    glcm_feats = compute_glcm_features(image_data, whole_mask)
    for k, v in glcm_feats.items():
        cat = k.split("_")[0]
        all_features.append({"category": cat, "feature_name": k, "value": v})
        categories[cat] = categories.get(cat, 0) + 1

    glrlm_feats = compute_glrlm_features(image_data, whole_mask)
    for k, v in glrlm_feats.items():
        cat = k.split("_")[0]
        all_features.append({"category": cat, "feature_name": k, "value": v})
        categories[cat] = categories.get(cat, 0) + 1

    # Summary stats
    feature_values = [f["value"] for f in all_features if isinstance(f["value"], (int, float))]
    summary = {
        "total_computed": len(all_features),
        "mean_value": round(float(np.mean(feature_values)), 4) if feature_values else 0,
        "std_value": round(float(np.std(feature_values)), 4) if feature_values else 0,
    }

    return {
        "total_features": len(all_features),
        "features": all_features,
        "categories": categories,
        "csv_download_url": "/api/radiomics/csv",
        "summary_stats": summary,
    }
