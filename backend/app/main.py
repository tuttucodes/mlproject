# backend/app/main.py
"""
Brain Tumor Segmentation API — ML Project by Rahul & Krishnaa for Dr. Valarmathi
Complete FastAPI backend with segmentation, grading, uncertainty, comparison, radiomics, QA
"""

import os
import uuid
import shutil
import logging
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from app.schemas.schemas import (
    SegmentationResponse,
    GradingResponse,
    UncertaintyResponse,
    ComparisonResponse,
    RadiomicsResponse,
    QAResponse,
    DatasetStatsResponse,
)
from app.api.segmentation import run_segmentation
from app.api.grading import run_grading
from app.api.uncertainty import run_uncertainty
from app.api.comparison import run_comparison
from app.api.radiomics import run_radiomics
from app.api.qa import run_qa
from app.api.dataset import get_brats_stats, download_sample_case
from app.utils.preprocessing import validate_nifti, preprocess_nifti
from app.models.model_manager import ModelManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

UPLOAD_DIR = Path("/tmp/brain_tumor_uploads")
RESULTS_DIR = Path("/tmp/brain_tumor_results")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="Brain Tumor Segmentation API — Rahul & Krishnaa for Dr. Valarmathi",
    description="End-to-end brain tumor segmentation with 5 novel ML features",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model_manager = ModelManager()


def cleanup_files(*paths):
    for p in paths:
        try:
            if p and Path(p).exists():
                if Path(p).is_dir():
                    shutil.rmtree(p)
                else:
                    os.remove(p)
        except Exception:
            pass


@app.on_event("startup")
async def startup():
    logger.info("Loading models...")
    model_manager.load_all()
    logger.info("Models ready.")


@app.get("/health")
async def health():
    return {"status": "ok", "models_loaded": model_manager.is_ready()}


# ─────────────────────────────────────────────────
# ENDPOINT 1: Segmentation
# ─────────────────────────────────────────────────
@app.post("/api/segment", response_model=SegmentationResponse)
async def segment(
    background_tasks: BackgroundTasks,
    t1: UploadFile = File(None),
    t1ce: UploadFile = File(None),
    t2: UploadFile = File(None),
    flair: UploadFile = File(None),
):
    """Run brain tumor segmentation on uploaded MRI modalities."""
    job_id = str(uuid.uuid4())
    job_dir = UPLOAD_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    result_dir = RESULTS_DIR / job_id
    result_dir.mkdir(parents=True, exist_ok=True)

    modality_paths = {}
    for name, upload in [("t1", t1), ("t1ce", t1ce), ("t2", t2), ("flair", flair)]:
        if upload:
            fpath = job_dir / f"{name}.nii.gz"
            with open(fpath, "wb") as f:
                content = await upload.read()
                f.write(content)
            if not validate_nifti(str(fpath)):
                raise HTTPException(400, f"Invalid NIfTI file for {name}")
            modality_paths[name] = str(fpath)

    if not modality_paths:
        raise HTTPException(400, "At least one MRI modality file required")

    try:
        result = run_segmentation(modality_paths, str(result_dir), model_manager)
        background_tasks.add_task(cleanup_files, str(job_dir))
        return result
    except Exception as e:
        logger.error(f"Segmentation failed: {e}")
        cleanup_files(str(job_dir), str(result_dir))
        raise HTTPException(500, f"Segmentation failed: {str(e)}")


# ─────────────────────────────────────────────────
# ENDPOINT 2: Tumor Grading (Novel Feature #1)
# ─────────────────────────────────────────────────
@app.post("/api/grade-tumor", response_model=GradingResponse)
async def grade_tumor(seg_file: UploadFile = File(...)):
    """Predict WHO tumor grade from segmentation mask."""
    job_id = str(uuid.uuid4())
    seg_path = UPLOAD_DIR / f"{job_id}_seg.nii.gz"
    with open(seg_path, "wb") as f:
        f.write(await seg_file.read())

    try:
        result = run_grading(str(seg_path), model_manager)
        return result
    except Exception as e:
        raise HTTPException(500, f"Grading failed: {str(e)}")
    finally:
        cleanup_files(str(seg_path))


# ─────────────────────────────────────────────────
# ENDPOINT 3: Uncertainty Quantification (Novel Feature #2)
# ─────────────────────────────────────────────────
@app.post("/api/uncertainty", response_model=UncertaintyResponse)
async def uncertainty(
    t1: UploadFile = File(None),
    t1ce: UploadFile = File(None),
    t2: UploadFile = File(None),
    flair: UploadFile = File(None),
    n_iterations: int = 20,
):
    """Run MC-Dropout uncertainty quantification."""
    job_id = str(uuid.uuid4())
    job_dir = UPLOAD_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    result_dir = RESULTS_DIR / job_id
    result_dir.mkdir(parents=True, exist_ok=True)

    modality_paths = {}
    for name, upload in [("t1", t1), ("t1ce", t1ce), ("t2", t2), ("flair", flair)]:
        if upload:
            fpath = job_dir / f"{name}.nii.gz"
            with open(fpath, "wb") as f:
                f.write(await upload.read())
            modality_paths[name] = str(fpath)

    try:
        result = run_uncertainty(
            modality_paths, str(result_dir), model_manager, n_iterations
        )
        return result
    except Exception as e:
        raise HTTPException(500, f"Uncertainty quantification failed: {str(e)}")
    finally:
        cleanup_files(str(job_dir))


# ─────────────────────────────────────────────────
# ENDPOINT 4: Comparative Analysis (Novel Feature #3)
# ─────────────────────────────────────────────────
@app.post("/api/compare", response_model=ComparisonResponse)
async def compare(
    seg1: UploadFile = File(...),
    seg2: UploadFile = File(...),
):
    """Compare two segmentation masks with Dice, Jaccard, Hausdorff metrics."""
    job_id = str(uuid.uuid4())
    p1 = UPLOAD_DIR / f"{job_id}_seg1.nii.gz"
    p2 = UPLOAD_DIR / f"{job_id}_seg2.nii.gz"

    with open(p1, "wb") as f:
        f.write(await seg1.read())
    with open(p2, "wb") as f:
        f.write(await seg2.read())

    try:
        result = run_comparison(str(p1), str(p2))
        return result
    except Exception as e:
        raise HTTPException(500, f"Comparison failed: {str(e)}")
    finally:
        cleanup_files(str(p1), str(p2))


# ─────────────────────────────────────────────────
# ENDPOINT 5: Radiomics Extraction (Novel Feature #4)
# ─────────────────────────────────────────────────
@app.post("/api/radiomics", response_model=RadiomicsResponse)
async def radiomics(
    image_file: UploadFile = File(...),
    seg_file: UploadFile = File(...),
):
    """Extract 107 radiomic features from segmentation."""
    job_id = str(uuid.uuid4())
    img_path = UPLOAD_DIR / f"{job_id}_img.nii.gz"
    seg_path = UPLOAD_DIR / f"{job_id}_seg.nii.gz"

    with open(img_path, "wb") as f:
        f.write(await image_file.read())
    with open(seg_path, "wb") as f:
        f.write(await seg_file.read())

    try:
        result = run_radiomics(str(img_path), str(seg_path))
        return result
    except Exception as e:
        raise HTTPException(500, f"Radiomics extraction failed: {str(e)}")
    finally:
        cleanup_files(str(img_path), str(seg_path))


# ─────────────────────────────────────────────────
# ENDPOINT 6: Quality Assurance (Novel Feature #5)
# ─────────────────────────────────────────────────
@app.post("/api/qa", response_model=QAResponse)
async def qa_check(seg_file: UploadFile = File(...)):
    """Run quality assurance checks on segmentation mask."""
    job_id = str(uuid.uuid4())
    seg_path = UPLOAD_DIR / f"{job_id}_seg.nii.gz"
    with open(seg_path, "wb") as f:
        f.write(await seg_file.read())

    try:
        result = run_qa(str(seg_path))
        return result
    except Exception as e:
        raise HTTPException(500, f"QA check failed: {str(e)}")
    finally:
        cleanup_files(str(seg_path))


# ─────────────────────────────────────────────────
# ENDPOINT 7: Dataset Statistics
# ─────────────────────────────────────────────────
@app.get("/api/brats-stats", response_model=DatasetStatsResponse)
async def brats_stats(subset: str = "GLI"):
    """Get BraTS dataset statistics for normalization."""
    return get_brats_stats(subset)


# ─────────────────────────────────────────────────
# ENDPOINT 8: Download Sample Case
# ─────────────────────────────────────────────────
@app.get("/api/sample-case")
async def sample_case():
    """Download a sample BraTS case for testing."""
    try:
        path = download_sample_case()
        return FileResponse(path, filename="sample_brats_case.nii.gz")
    except Exception as e:
        raise HTTPException(500, f"Failed to download sample: {str(e)}")


# ─────────────────────────────────────────────────
# ENDPOINT 9: Download segmentation result
# ─────────────────────────────────────────────────
@app.get("/api/results/{job_id}/{filename}")
async def download_result(job_id: str, filename: str):
    path = RESULTS_DIR / job_id / filename
    if not path.exists():
        raise HTTPException(404, "Result not found")
    return FileResponse(str(path), filename=filename)


# ─────────────────────────────────────────────────
# Novel Feature #6: Survival Prediction
# ─────────────────────────────────────────────────
@app.post("/api/survival-prediction")
async def survival_prediction(seg_file: UploadFile = File(...)):
    """Predict overall survival from tumor morphology (Novel Feature #6)."""
    job_id = str(uuid.uuid4())
    seg_path = UPLOAD_DIR / f"{job_id}_seg.nii.gz"
    with open(seg_path, "wb") as f:
        f.write(await seg_file.read())
    try:
        from app.api.survival import run_survival_prediction
        result = run_survival_prediction(str(seg_path), model_manager)
        return result
    except Exception as e:
        raise HTTPException(500, f"Survival prediction failed: {str(e)}")
    finally:
        cleanup_files(str(seg_path))


# ─────────────────────────────────────────────────
# Novel Feature #7: Attention Map Visualization
# ─────────────────────────────────────────────────
@app.post("/api/attention-maps")
async def attention_maps(
    t1ce: UploadFile = File(...),
):
    """Generate Grad-CAM attention maps for model explainability (Novel Feature #7)."""
    job_id = str(uuid.uuid4())
    fpath = UPLOAD_DIR / f"{job_id}_t1ce.nii.gz"
    with open(fpath, "wb") as f:
        f.write(await t1ce.read())
    try:
        from app.api.attention import run_attention_maps
        result = run_attention_maps(str(fpath), model_manager)
        return result
    except Exception as e:
        raise HTTPException(500, f"Attention map generation failed: {str(e)}")
    finally:
        cleanup_files(str(fpath))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
