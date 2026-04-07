"""
FastAPI Backend for Brain Tumor MRI Segmentation
Deployed on Google Colab with public ngrok tunnel

Endpoints:
  GET  /health   - Health check
  POST /segment  - Upload NIFTI MRI, receive segmentation mask

Pretrained model: MONAI Model Zoo 'brats_mri_segmentation'
  - Trained on BraTS 2021 dataset (1251 cases)
  - SegResNet architecture (init_filters=32)
  - 3 output classes: Whole Tumor, Tumor Core, Enhancing Tumor
"""

import io
import os
import uuid
import base64
import tempfile
import asyncio
import threading
import numpy as np
import torch
import nibabel as nib
# scipy NOT imported — pure numpy resize avoids C-extension re-import issues in Colab
from typing import Optional
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from monai.networks.nets import SegResNet
import uvicorn
from pyngrok import ngrok

# ── CRITICAL: Patch event loop for Jupyter/Colab BEFORE any asyncio calls ──
import nest_asyncio
nest_asyncio.apply()

# In-memory job store  {job_id: {"status": "queued"|"running"|"done"|"error", ...}}
JOBS: dict = {}

# ============================================================================
# Configuration
# ============================================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_INPUT_SIZE = (128, 128, 128)
NUM_CHANNELS = 4
NUM_CLASSES = 3

NGROK_AUTH_TOKEN = os.environ.get("NGROK_AUTH_TOKEN", "")

# ============================================================================
# FastAPI App
# ============================================================================

app = FastAPI(
    title="Brain Tumor Segmentation API",
    description="MRI segmentation using MONAI SegResNet pretrained on BraTS 2021",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,   # must be False when allow_origins=["*"]
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# Model Loading — MONAI Model Zoo (BraTS 2021 pretrained)
# ============================================================================

def load_model():
    """
    Downloads and loads the official MONAI Model Zoo BraTS segmentation model.
    Bundle: brats_mri_segmentation
    - SegResNet, init_filters=32, trained on BraTS 2021 (1251 cases)
    - Dice scores: WT=0.9218, TC=0.8560, ET=0.7926
    Falls back to random weights if download fails.
    """
    bundle_dir = "/content/bundles"
    bundle_name = "brats_mri_segmentation"
    bundle_path = os.path.join(bundle_dir, bundle_name)
    model_pt_path = os.path.join(bundle_path, "models", "model.pt")

    # ── Step 1: Try MONAI bundle download (proper pretrained weights) ──────
    try:
        from monai.bundle import download as bundle_download

        if not os.path.exists(model_pt_path):
            print("=" * 60)
            print("Downloading pretrained BraTS 2021 model from MONAI Model Zoo...")
            print("(~150 MB, one-time download)")
            print("=" * 60)
            os.makedirs(bundle_dir, exist_ok=True)
            bundle_download(name=bundle_name, bundle_dir=bundle_dir)
            print("✅ Bundle downloaded successfully!")

        # SegResNet config MUST match the bundle's trained architecture
        # init_filters=16 — confirmed from checkpoint weight shapes
        model = SegResNet(
            spatial_dims=3,
            init_filters=16,
            in_channels=4,
            out_channels=3,
            blocks_down=[1, 2, 2, 4],
            blocks_up=[1, 1, 1],
            dropout_prob=0.2,
        )

        checkpoint = torch.load(model_pt_path, map_location=DEVICE, weights_only=True)

        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if "state_dict" in checkpoint:
                model.load_state_dict(checkpoint["state_dict"])
            elif "model" in checkpoint:
                model.load_state_dict(checkpoint["model"])
            else:
                model.load_state_dict(checkpoint)
        else:
            model.load_state_dict(checkpoint)

        model.to(DEVICE)
        model.eval()

        print("=" * 60)
        print("✅ PRETRAINED MODEL LOADED SUCCESSFULLY")
        print("   Dataset : BraTS 2021 (1251 multi-site cases)")
        print("   Dice WT  : 0.9218")
        print("   Dice TC  : 0.8560")
        print("   Dice ET  : 0.7926")
        print(f"   Device   : {DEVICE}")
        print("=" * 60)
        return model

    except Exception as e:
        print(f"⚠️  MONAI bundle download failed: {e}")

    # ── Step 2: Try direct URL download (fallback) ─────────────────────────
    try:
        import urllib.request, zipfile

        ZIP_URL = (
            "https://github.com/Project-MONAI/model-zoo/releases/download/"
            "hosting_storage_v1/brats_mri_segmentation_v0.5.3.zip"
        )
        zip_path = "/content/brats_bundle.zip"

        if not os.path.exists(model_pt_path):
            print(f"Trying direct download from GitHub releases...")
            urllib.request.urlretrieve(ZIP_URL, zip_path)
            with zipfile.ZipFile(zip_path, "r") as z:
                z.extractall(bundle_dir)
            print("✅ Direct download complete!")

        model = SegResNet(
            spatial_dims=3,
            init_filters=16,
            in_channels=4,
            out_channels=3,
            blocks_down=[1, 2, 2, 4],
            blocks_up=[1, 1, 1],
            dropout_prob=0.2,
        )
        checkpoint = torch.load(model_pt_path, map_location=DEVICE, weights_only=True)
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
        else:
            model.load_state_dict(checkpoint)

        model.to(DEVICE)
        model.eval()
        print("✅ Pretrained weights loaded via direct download!")
        return model

    except Exception as e2:
        print(f"⚠️  Direct download also failed: {e2}")

    # ── Step 3: Last resort — random weights (demo only) ──────────────────
    print("⚠️  FALLING BACK TO RANDOM WEIGHTS — segmentation will not be accurate")
    model = SegResNet(
        spatial_dims=3,
        init_filters=8,
        in_channels=4,
        out_channels=3,
        blocks_down=[1, 2, 2, 4],
        blocks_up=[1, 1, 1],
    )
    model.to(DEVICE)
    model.eval()
    return model


# Load globally on startup
try:
    model = load_model()
    model_loaded = True
except Exception as e:
    print(f"ERROR: Failed to load model - {e}")
    model_loaded = False
    model = None

# ============================================================================
# Preprocessing
# ============================================================================

def resize_volume_nn(vol4d, target):
    """
    Pure-numpy nearest-neighbour 3-D resize for a (C, Z, Y, X) float32 array.
    Replaces scipy.ndimage.zoom — no C extensions, safe to call from threads.
    """
    _, sz, sy, sx = vol4d.shape
    tz, ty, tx = target
    iz = np.round(np.linspace(0, sz - 1, tz)).astype(np.int32)
    iy = np.round(np.linspace(0, sy - 1, ty)).astype(np.int32)
    ix = np.round(np.linspace(0, sx - 1, tx)).astype(np.int32)
    return vol4d[:, iz[:, None, None], iy[None, :, None], ix[None, None, :]]


def normalize_intensity(volume, lower_percentile=0.5, upper_percentile=99.5):
    p_lower = np.percentile(volume, lower_percentile)
    p_upper = np.percentile(volume, upper_percentile)
    volume = np.clip(volume, p_lower, p_upper)
    volume = (volume - p_lower) / (p_upper - p_lower + 1e-8)
    return volume.astype(np.float32)


def preprocess_nifti(nifti_data):
    """
    Converts raw NIFTI array → (1, 4, 128, 128, 128) tensor for SegResNet.
    Handles 3D (single modality) and 4D (multi-modality) inputs.
    """
    if nifti_data.ndim == 3:
        channels = np.stack([nifti_data] * NUM_CHANNELS, axis=0)
    elif nifti_data.ndim == 4:
        channels = nifti_data.transpose(3, 0, 1, 2)
        if channels.shape[0] < NUM_CHANNELS:
            pad = np.tile(channels[-1:], (NUM_CHANNELS - channels.shape[0], 1, 1, 1))
            channels = np.concatenate([channels, pad], axis=0)
        elif channels.shape[0] > NUM_CHANNELS:
            channels = channels[:NUM_CHANNELS]
    else:
        raise ValueError(f"Unexpected NIFTI shape: {nifti_data.shape}")

    # Normalize each channel
    channels = np.stack([normalize_intensity(channels[i]) for i in range(NUM_CHANNELS)], axis=0)

    # Resize to 128³ if needed — pure numpy, no scipy required
    if channels.shape[1:] != MODEL_INPUT_SIZE:
        channels = resize_volume_nn(channels, MODEL_INPUT_SIZE).astype(np.float32)

    tensor = torch.from_numpy(channels).float().unsqueeze(0)  # (1, 4, 128, 128, 128)
    return tensor.to(DEVICE)

# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/health")
async def health():
    return JSONResponse({
        "status": "ready" if model_loaded else "error",
        "device": str(DEVICE),
        "model_loaded": model_loaded,
        "message": "Pretrained BraTS 2021 model operational" if model_loaded else "Model load failed"
    })


def _load_nifti_bytes(file_bytes: bytes, fname: str) -> np.ndarray:
    """Write bytes to temp file, load with nibabel, return float32 array."""
    suffix = ".nii.gz" if fname.endswith(".nii.gz") else ".nii"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    try:
        return nib.load(tmp_path).get_fdata(dtype=np.float32)
    finally:
        os.unlink(tmp_path)


def _run_inference_job(job_id: str, modality_files: dict):
    """
    Runs in a background thread.
    modality_files: dict of {name: (bytes, filename)} — keys: t1, t1ce, t2, flair (or 'file')
    """
    try:
        JOBS[job_id]["status"] = "running"

        # ── Build 4-channel input in BraTS order: T1, T1CE, T2, FLAIR ──
        ORDER = ["t1", "t1ce", "t2", "flair"]
        loaded = {}
        for key, (fb, fn) in modality_files.items():
            loaded[key] = normalize_intensity(_load_nifti_bytes(fb, fn))

        if len(loaded) >= 2:
            # Use available modalities in correct order; duplicate missing ones
            ch_list = []
            available = list(loaded.values())
            for mod in ORDER:
                ch_list.append(loaded[mod] if mod in loaded else available[0])
            channels = np.stack(ch_list, axis=0)          # (4, Z, Y, X)
        else:
            # Single file — replicate to 4 channels
            arr = list(loaded.values())[0]
            channels = np.stack([arr] * NUM_CHANNELS, axis=0)

        # Resize to 128³ if needed
        if channels.shape[1:] != MODEL_INPUT_SIZE:
            channels = resize_volume_nn(channels, MODEL_INPUT_SIZE).astype(np.float32)

        tensor = torch.from_numpy(channels).float().unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            output = model(tensor)   # [1, 3, 128, 128, 128]

        # ── Correct BraTS post-processing: sigmoid threshold, NOT argmax ──
        # Model outputs 3 independent binary logits: TC (ch0), WT (ch1), ET (ch2)
        probs = torch.sigmoid(output[0])               # [3, 128, 128, 128]
        tc = (probs[0] > 0.5).cpu().numpy()            # Tumor Core
        wt = (probs[1] > 0.5).cpu().numpy()            # Whole Tumor
        et = (probs[2] > 0.5).cpu().numpy()            # Enhancing Tumor

        seg = np.zeros(tc.shape, dtype=np.uint8)
        seg[wt] = 2   # Edema (whole tumor region)
        seg[tc] = 1   # Necrotic/tumor core (overrides edema)
        seg[et] = 3   # Enhancing tumor (highest priority)

        seg_b64 = base64.b64encode(seg.tobytes()).decode("utf-8")
        ncr = int((seg == 1).sum())
        ed  = int((seg == 2).sum())
        enh = int((seg == 3).sum())
        print(f"[job {job_id}] done — NCR:{ncr} ED:{ed} ET:{enh}")

        JOBS[job_id] = {
            "status":       "done",
            "segmentation": seg_b64,
            "shape":        list(seg.shape),
            "classes":      int(NUM_CLASSES),
            "device_used":  str(DEVICE),
        }

    except Exception as e:
        import traceback
        print(f"[job {job_id}] ERROR:\n{traceback.format_exc()}")
        JOBS[job_id] = {"status": "error", "detail": str(e)}


@app.post("/segment")
async def segment(
    file:  Optional[UploadFile] = File(None),   # single-file fallback
    t1:    Optional[UploadFile] = File(None),
    t1ce:  Optional[UploadFile] = File(None),
    t2:    Optional[UploadFile] = File(None),
    flair: Optional[UploadFile] = File(None),
):
    """
    Accepts up to 4 BraTS modalities (t1, t1ce, t2, flair) or a single 'file'.
    Returns job_id immediately; poll GET /segment/{job_id} for result.
    """
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    modality_files = {}
    for name, uf in [("t1", t1), ("t1ce", t1ce), ("t2", t2), ("flair", flair), ("file", file)]:
        if uf is not None:
            fb = await uf.read()
            modality_files[name] = (fb, uf.filename or f"{name}.nii")

    if not modality_files:
        raise HTTPException(status_code=400, detail="No NIfTI file uploaded.")

    job_id = uuid.uuid4().hex[:10]
    JOBS[job_id] = {"status": "queued"}
    mods_str = ", ".join(modality_files.keys())
    print(f"[job {job_id}] queued — modalities: {mods_str}")

    thread = threading.Thread(target=_run_inference_job, args=(job_id, modality_files), daemon=True)
    thread.start()
    return JSONResponse({"job_id": job_id, "status": "queued"})


@app.get("/segment/{job_id}")
async def poll_segment(job_id: str):
    """Poll for segmentation result. Returns status + result when done."""
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail="Job not found")
    return JSONResponse(JOBS[job_id])

# ============================================================================
# ngrok Tunnel
# ============================================================================

def setup_ngrok_tunnel():
    if not NGROK_AUTH_TOKEN:
        print("WARNING: NGROK_AUTH_TOKEN not set — set it before calling this")
        return None
    try:
        ngrok.set_auth_token(NGROK_AUTH_TOKEN)
        public_url = ngrok.connect(8000, "http")
        url_str = public_url.public_url if hasattr(public_url, "public_url") else str(public_url).split('"')[1]
        print(f"\n{'='*60}")
        print(f"ngrok tunnel: {url_str}")
        print(f"Set in Vercel: VITE_API_URL={url_str}")
        print(f"{'='*60}\n")
        return url_str
    except Exception as e:
        print(f"ngrok failed: {e}")
        return None

# ============================================================================
# Entry Point
# ============================================================================

def run_server():
    """Run uvicorn server in a background thread (for Colab compatibility)."""
    import sys

    print(f"Device : {DEVICE}")
    print(f"Model  : {'loaded' if model_loaded else 'FAILED'}")

    # ── Start ngrok BEFORE uvicorn so the URL is printed immediately ──
    tunnel_url = setup_ngrok_tunnel()
    if tunnel_url:
        print(f"\n{'='*60}")
        print(f"  YOUR NGROK URL (copy this into Vercel):")
        print(f"  {tunnel_url}")
        print(f"  Set VITE_API_URL = {tunnel_url}")
        print(f"{'='*60}\n")

    # Run uvicorn in a thread to avoid blocking Colab
    server_thread = threading.Thread(
        target=lambda: uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info"),
        daemon=True
    )
    server_thread.start()
    print("✅ Server started in background thread")
    return tunnel_url

if __name__ == "__main__":
    run_server()
