# backend/app/api/grading.py
"""Novel Feature #1: Automated WHO tumor grading from segmentation mask."""

import numpy as np
import logging
from app.utils.preprocessing import load_nifti

logger = logging.getLogger(__name__)

GRADE_DESCRIPTIONS = {
    "Grade I": "Pilocytic astrocytoma — slow-growing, well-defined margins",
    "Grade II": "Diffuse astrocytoma — slow-growing but infiltrative",
    "Grade III": "Anaplastic astrocytoma — malignant, rapid growth",
    "Grade IV": "Glioblastoma multiforme (GBM) — most aggressive, poor prognosis",
}

WHO_CLASSIFICATION = {
    "Grade I": "Low-grade glioma (WHO Grade I)",
    "Grade II": "Low-grade glioma (WHO Grade II)",
    "Grade III": "High-grade glioma (WHO Grade III)",
    "Grade IV": "High-grade glioma (WHO Grade IV - GBM)",
}


def extract_grading_features(segmentation: np.ndarray) -> dict:
    """Extract morphological & intensity features for grading."""
    from scipy import ndimage

    total_tumor = (segmentation > 0).sum()
    if total_tumor == 0:
        return {f: 0.0 for f in [
            "volume_cm3", "surface_area_mm2", "sphericity", "compactness",
            "mean_intensity", "std_intensity", "skewness", "kurtosis",
            "enhancing_ratio", "necrotic_ratio", "edema_ratio",
            "et_to_ncr_ratio", "tumor_location_x", "tumor_location_y",
            "tumor_location_z",
        ]}

    # Volume
    volume_mm3 = float(total_tumor)
    volume_cm3 = volume_mm3 / 1000.0

    # Surface area (approximate using gradient)
    tumor_mask = (segmentation > 0).astype(float)
    gradient = np.gradient(tumor_mask)
    surface = np.sqrt(sum(g**2 for g in gradient))
    surface_area = float(surface.sum())

    # Sphericity
    if volume_mm3 > 0:
        sphericity = (np.pi ** (1/3) * (6 * volume_mm3) ** (2/3)) / (surface_area + 1e-8)
        sphericity = min(sphericity, 1.0)
    else:
        sphericity = 0.0

    compactness = volume_mm3 / (surface_area ** 1.5 + 1e-8)

    # Region ratios
    enhancing = (segmentation == 4).sum()
    necrotic = (segmentation == 1).sum()
    edema = (segmentation == 2).sum()

    enhancing_ratio = enhancing / total_tumor if total_tumor > 0 else 0
    necrotic_ratio = necrotic / total_tumor if total_tumor > 0 else 0
    edema_ratio = edema / total_tumor if total_tumor > 0 else 0
    et_to_ncr = enhancing / (necrotic + 1) if necrotic > 0 else enhancing

    # Tumor centroid location (normalized)
    coords = np.argwhere(segmentation > 0)
    if len(coords) > 0:
        centroid = coords.mean(axis=0)
        loc_x = centroid[0] / segmentation.shape[0]
        loc_y = centroid[1] / segmentation.shape[1]
        loc_z = centroid[2] / segmentation.shape[2]
    else:
        loc_x = loc_y = loc_z = 0.5

    # Intensity statistics (use segmentation values as proxy)
    tumor_vals = segmentation[segmentation > 0].astype(float)
    from scipy.stats import skew, kurtosis as kurt
    mean_int = float(tumor_vals.mean())
    std_int = float(tumor_vals.std())
    skewness = float(skew(tumor_vals)) if len(tumor_vals) > 1 else 0
    kurt_val = float(kurt(tumor_vals)) if len(tumor_vals) > 1 else 0

    return {
        "volume_cm3": round(volume_cm3, 4),
        "surface_area_mm2": round(surface_area, 2),
        "sphericity": round(sphericity, 4),
        "compactness": round(compactness, 6),
        "mean_intensity": round(mean_int, 4),
        "std_intensity": round(std_int, 4),
        "skewness": round(skewness, 4),
        "kurtosis": round(kurt_val, 4),
        "enhancing_ratio": round(enhancing_ratio, 4),
        "necrotic_ratio": round(necrotic_ratio, 4),
        "edema_ratio": round(edema_ratio, 4),
        "et_to_ncr_ratio": round(et_to_ncr, 4),
        "tumor_location_x": round(loc_x, 4),
        "tumor_location_y": round(loc_y, 4),
        "tumor_location_z": round(loc_z, 4),
    }


def run_grading(seg_path: str, model_manager) -> dict:
    """Run tumor grading pipeline."""
    seg_data, _ = load_nifti(seg_path)
    seg = seg_data.astype(np.int32)

    features = extract_grading_features(seg)
    prediction = model_manager.grading.predict(features)

    grade = prediction["predicted_grade"]
    risk = "High" if grade in ("Grade III", "Grade IV") else (
        "Moderate" if grade == "Grade II" else "Low"
    )

    return {
        "predicted_grade": grade,
        "who_classification": WHO_CLASSIFICATION.get(grade, "Unknown"),
        "confidence": round(prediction["confidence"], 4),
        "grade_probabilities": prediction["probabilities"],
        "risk_stratification": risk,
        "features_used": features,
        "clinical_notes": GRADE_DESCRIPTIONS.get(grade, ""),
    }
