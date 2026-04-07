# backend/app/schemas/schemas.py
"""Pydantic models for all API request/response schemas."""

from pydantic import BaseModel, Field
from typing import Dict, List, Optional


class TumorRegionStats(BaseModel):
    label: int
    name: str
    volume_mm3: float
    volume_cm3: float
    voxel_count: int
    percentage: float


class SegmentationResponse(BaseModel):
    job_id: str
    status: str = "completed"
    segmentation_file: str
    total_tumor_volume_cm3: float
    regions: List[TumorRegionStats]
    inference_time_seconds: float
    model_used: str
    input_shape: List[int]
    download_url: str


class GradingResponse(BaseModel):
    predicted_grade: str  # "Grade I", "Grade II", etc.
    who_classification: str  # "Low-grade glioma", "High-grade glioma"
    confidence: float
    grade_probabilities: Dict[str, float]
    risk_stratification: str  # "Low", "Moderate", "High"
    features_used: Dict[str, float]
    clinical_notes: str


class UncertaintyResponse(BaseModel):
    mean_uncertainty: float
    max_uncertainty: float
    high_uncertainty_volume_cm3: float
    high_uncertainty_percentage: float
    uncertainty_heatmap_file: str
    n_mc_iterations: int
    voxel_wise_stats: Dict[str, float]
    download_url: str
    clinical_interpretation: str


class RegionComparison(BaseModel):
    region_name: str
    dice_coefficient: float
    jaccard_index: float
    hausdorff_distance_mm: float
    volume_similarity: float
    volume_diff_cm3: float


class ComparisonResponse(BaseModel):
    overall_dice: float
    overall_jaccard: float
    overall_hausdorff_mm: float
    region_comparisons: List[RegionComparison]
    volume_change_cm3: float
    volume_change_percent: float
    progression_status: str  # "Stable", "Progressing", "Regressing"
    diff_heatmap_file: Optional[str] = None


class RadiomicFeature(BaseModel):
    category: str
    feature_name: str
    value: float


class RadiomicsResponse(BaseModel):
    total_features: int
    features: List[RadiomicFeature]
    categories: Dict[str, int]
    csv_download_url: str
    summary_stats: Dict[str, float]


class QACheck(BaseModel):
    check_name: str
    status: str  # "PASS", "REVIEW", "FAIL"
    message: str
    details: Optional[Dict] = None


class QAResponse(BaseModel):
    overall_status: str  # "PASS", "REVIEW", "FAIL"
    checks: List[QACheck]
    recommendations: List[str]
    segmentation_quality_score: float  # 0-100


class DatasetStatsResponse(BaseModel):
    subset: str
    total_cases: int
    modalities: List[str]
    voxel_dimensions: List[int]
    mean_tumor_volume_cm3: float
    label_distribution: Dict[str, float]
    intensity_stats: Dict[str, Dict[str, float]]


class SurvivalResponse(BaseModel):
    predicted_os_days: float
    predicted_os_months: float
    confidence_interval_lower: float
    confidence_interval_upper: float
    risk_group: str
    survival_curve: List[Dict[str, float]]
    features_importance: Dict[str, float]


class AttentionMapResponse(BaseModel):
    attention_map_file: str
    download_url: str
    top_regions: List[Dict[str, float]]
    model_confidence: float
