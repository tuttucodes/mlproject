# backend/app/api/qa.py
"""Novel Feature #5: Segmentation Quality Assurance (QA)."""

import numpy as np
import logging
from typing import List, Dict
from app.utils.preprocessing import load_nifti

logger = logging.getLogger(__name__)


def check_connectivity(seg: np.ndarray, label: int) -> Dict:
    """Check if region has a single connected component."""
    from scipy import ndimage

    mask = seg == label
    if mask.sum() == 0:
        return {"status": "PASS", "message": f"Label {label}: not present", "n_components": 0}

    labeled, n_components = ndimage.label(mask)
    if n_components == 1:
        return {"status": "PASS", "message": f"Label {label}: single connected component", "n_components": 1}
    elif n_components <= 3:
        return {
            "status": "REVIEW",
            "message": f"Label {label}: {n_components} components — minor fragmentation",
            "n_components": n_components,
        }
    else:
        return {
            "status": "FAIL",
            "message": f"Label {label}: {n_components} components — significant fragmentation",
            "n_components": n_components,
        }


def check_volume_plausibility(seg: np.ndarray) -> Dict:
    """Check if tumor volume is within plausible range."""
    total_voxels = (seg > 0).sum()
    volume_cm3 = total_voxels / 1000.0  # 1mm³ voxels

    if volume_cm3 < 0.1:
        return {
            "status": "FAIL",
            "message": f"Tumor volume {volume_cm3:.2f} cm³ is suspiciously small (<0.1 cm³)",
            "volume_cm3": volume_cm3,
        }
    elif volume_cm3 > 500:
        return {
            "status": "FAIL",
            "message": f"Tumor volume {volume_cm3:.2f} cm³ exceeds plausible maximum (>500 cm³)",
            "volume_cm3": volume_cm3,
        }
    elif volume_cm3 > 200:
        return {
            "status": "REVIEW",
            "message": f"Tumor volume {volume_cm3:.2f} cm³ is large — verify segmentation",
            "volume_cm3": volume_cm3,
        }
    else:
        return {
            "status": "PASS",
            "message": f"Tumor volume {volume_cm3:.2f} cm³ within normal range",
            "volume_cm3": volume_cm3,
        }


def check_holes(seg: np.ndarray, label: int) -> Dict:
    """Detect internal holes in segmentation region."""
    from scipy import ndimage

    mask = seg == label
    if mask.sum() == 0:
        return {"status": "PASS", "message": f"Label {label}: not present"}

    filled = ndimage.binary_fill_holes(mask)
    holes = filled.astype(int) - mask.astype(int)
    n_hole_voxels = holes.sum()

    if n_hole_voxels == 0:
        return {"status": "PASS", "message": f"Label {label}: no internal holes"}
    elif n_hole_voxels < 100:
        return {"status": "REVIEW", "message": f"Label {label}: {n_hole_voxels} hole voxels — minor"}
    else:
        return {"status": "FAIL", "message": f"Label {label}: {n_hole_voxels} hole voxels — significant"}


def check_label_hierarchy(seg: np.ndarray) -> Dict:
    """
    Check BraTS label hierarchy consistency:
    - Enhancing (4) should be surrounded by necrotic (1) or edema (2)
    - Necrotic (1) should be inside edema (2) boundary
    """
    from scipy import ndimage

    enhancing = seg == 4
    necrotic = seg == 1
    edema = seg == 2

    if enhancing.sum() == 0:
        return {"status": "PASS", "message": "No enhancing tumor — hierarchy check skipped"}

    # Dilate enhancing by 1 voxel and check if surrounded by tumor
    dilated_et = ndimage.binary_dilation(enhancing, iterations=2)
    surrounding = dilated_et & ~enhancing
    tumor_around_et = (seg[surrounding] > 0).mean() if surrounding.sum() > 0 else 1.0

    if tumor_around_et > 0.8:
        return {"status": "PASS", "message": f"Enhancing tumor properly surrounded ({tumor_around_et:.0%})"}
    elif tumor_around_et > 0.5:
        return {"status": "REVIEW", "message": f"Enhancing tumor partially exposed ({tumor_around_et:.0%})"}
    else:
        return {"status": "FAIL", "message": f"Enhancing tumor largely uncontained ({tumor_around_et:.0%})"}


def check_symmetry(seg: np.ndarray) -> Dict:
    """Check if tumor is unrealistically symmetric (potential artifact)."""
    tumor = (seg > 0).astype(float)
    if tumor.sum() == 0:
        return {"status": "PASS", "message": "No tumor — symmetry check skipped"}

    # Compare left-right symmetry
    mid = tumor.shape[2] // 2
    left = tumor[:, :, :mid]
    right = np.flip(tumor[:, :, mid:2*mid], axis=2)

    if left.shape == right.shape:
        correlation = np.corrcoef(left.flatten(), right.flatten())[0, 1]
        if correlation > 0.95:
            return {"status": "REVIEW", "message": f"Suspiciously symmetric (r={correlation:.3f}) — possible artifact"}
    return {"status": "PASS", "message": "Normal asymmetry"}


def run_qa(seg_path: str) -> dict:
    """Run all QA checks on segmentation."""
    seg_data, _ = load_nifti(seg_path)
    seg = seg_data.astype(np.int32)

    checks = []

    # 1. Connectivity per region
    for label, name in [(1, "Necrotic"), (2, "Edema"), (4, "Enhancing")]:
        result = check_connectivity(seg, label)
        checks.append({
            "check_name": f"Connectivity ({name})",
            "status": result["status"],
            "message": result["message"],
            "details": {"n_components": result.get("n_components", 0)},
        })

    # 2. Volume plausibility
    vol_result = check_volume_plausibility(seg)
    checks.append({
        "check_name": "Volume Plausibility",
        "status": vol_result["status"],
        "message": vol_result["message"],
        "details": {"volume_cm3": vol_result.get("volume_cm3", 0)},
    })

    # 3. Hole detection
    for label, name in [(1, "Necrotic"), (2, "Edema"), (4, "Enhancing")]:
        result = check_holes(seg, label)
        checks.append({
            "check_name": f"Hole Detection ({name})",
            "status": result["status"],
            "message": result["message"],
        })

    # 4. Label hierarchy
    hier_result = check_label_hierarchy(seg)
    checks.append({
        "check_name": "Label Hierarchy",
        "status": hier_result["status"],
        "message": hier_result["message"],
    })

    # 5. Symmetry check
    sym_result = check_symmetry(seg)
    checks.append({
        "check_name": "Symmetry Check",
        "status": sym_result["status"],
        "message": sym_result["message"],
    })

    # 6. Empty segmentation check
    if (seg > 0).sum() == 0:
        checks.append({
            "check_name": "Non-Empty Check",
            "status": "FAIL",
            "message": "Segmentation is completely empty",
        })
    else:
        checks.append({
            "check_name": "Non-Empty Check",
            "status": "PASS",
            "message": f"Segmentation contains {np.unique(seg[seg > 0]).tolist()} labels",
        })

    # Overall status
    statuses = [c["status"] for c in checks]
    if "FAIL" in statuses:
        overall = "FAIL"
    elif "REVIEW" in statuses:
        overall = "REVIEW"
    else:
        overall = "PASS"

    # Quality score
    pass_count = statuses.count("PASS")
    total = len(statuses)
    quality_score = (pass_count / total) * 100

    # Recommendations
    recommendations = []
    if overall == "FAIL":
        recommendations.append("Re-run segmentation with different preprocessing parameters")
        recommendations.append("Check input MRI quality and modality alignment")
    if any("fragmentation" in c["message"] for c in checks):
        recommendations.append("Apply morphological closing to reduce fragmentation")
    if any("hole" in c["message"].lower() and c["status"] != "PASS" for c in checks):
        recommendations.append("Apply hole-filling post-processing")
    if not recommendations:
        recommendations.append("Segmentation passes all quality checks")

    return {
        "overall_status": overall,
        "checks": checks,
        "recommendations": recommendations,
        "segmentation_quality_score": round(quality_score, 1),
    }
