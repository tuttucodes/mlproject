# backend/app/api/survival.py
"""Novel Feature #6: Survival prediction from tumor morphology."""

import numpy as np
from app.utils.preprocessing import load_nifti
from app.api.grading import extract_grading_features


def run_survival_prediction(seg_path: str, model_manager) -> dict:
    seg_data, _ = load_nifti(seg_path)
    seg = seg_data.astype(np.int32)
    features = extract_grading_features(seg)
    return model_manager.survival.predict(features)
