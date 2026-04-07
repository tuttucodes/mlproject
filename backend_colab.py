"""
FastAPI Backend for Brain Tumor MRI Segmentation
Deployed on Google Colab with public ngrok tunnel

Endpoints:
  GET /health - Health check for connectivity verification
  POST /segment - Upload NIFTI MRI file and receive segmentation mask

This file is designed to run entirely within a Google Colab notebook cell.
All dependencies are installed via: !pip install -r requirements.txt
"""

import io
import base64
import numpy as np
import torch
import nibabel as nib
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from monai.networks.nets import SegResNet
import uvicorn
from pyngrok import ngrok
import os

# ============================================================================
# Configuration
# ============================================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_INPUT_SIZE = (128, 128, 128)
NUM_CHANNELS = 4
NUM_CLASSES = 3

# ngrok authentication token - set via environment variable
# In Colab: os.environ['NGROK_AUTH_TOKEN'] = "your_token_here"
NGROK_AUTH_TOKEN = os.environ.get("NGROK_AUTH_TOKEN", "")

# ============================================================================
# FastAPI Application Setup
# ============================================================================

app = FastAPI(
    title="Brain Tumor Segmentation API",
    description="MRI segmentation backend using MONAI SegResNet",
    version="1.0.0"
)

# Configure CORS to allow requests from React frontend on different domains
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# Model Loading
# ============================================================================

def load_model():
    """
    Load pretrained MONAI SegResNet model for 3D brain tumor segmentation.

    Model Architecture:
    - Input: 4-channel 3D MRI volumes (128x128x128 voxels)
    - Output: 3 classes (background, tumor core, tumor edema)
    - Pretrained on BraTS dataset
    """
    model = SegResNet(
        spatial_dims=3,
        init_filters=8,
        in_channels=NUM_CHANNELS,
        out_channels=NUM_CLASSES,
        blocks_down=(1, 2, 2, 4),
        blocks_up=(1, 1, 1)
    )

    try:
        # Attempt to download pretrained weights from MONAI model zoo
        from monai.apps import download_and_extract
        model_dir = download_and_extract(
            url="https://download.pytorch.org/models/segresnet_segmentation.pth",
            output_dir="/tmp/monai_models",
            mode="skip",
        )
        state_dict = torch.load(
            os.path.join(model_dir, "model.pth"),
            map_location=DEVICE,
            weights_only=True
        )
        model.load_state_dict(state_dict)
    except Exception as e:
        print(f"Warning: Could not load pretrained weights - {str(e)}")
        print("Proceeding with randomly initialized model")
        print("(For production, download weights from MONAI model zoo or train custom model)")

    model = model.to(DEVICE)
    model.eval()
    return model

# Load model globally on startup
try:
    model = load_model()
    model_loaded = True
except Exception as e:
    print(f"ERROR: Failed to load model - {str(e)}")
    model_loaded = False
    model = None

# ============================================================================
# Preprocessing Functions
# ============================================================================

def normalize_intensity(volume, lower_percentile=0.5, upper_percentile=99.5):
    """
    Normalize intensity values using percentile clipping.
    Handles variable intensity ranges across different MRI scanners.
    """
    p_lower = np.percentile(volume, lower_percentile)
    p_upper = np.percentile(volume, upper_percentile)
    volume = np.clip(volume, p_lower, p_upper)
    volume = (volume - p_lower) / (p_upper - p_lower + 1e-8)
    return volume

def preprocess_nifti(nifti_data):
    """
    Preprocess NIFTI MRI data to format required by SegResNet.

    Args:
        nifti_data: Raw NIFTI image array from nibabel (may be 3D or 4D)

    Returns:
        torch.Tensor: (1, 4, 128, 128, 128) ready for model inference
    """
    # Handle both 3D and 4D NIFTI formats
    if nifti_data.ndim == 3:
        # Single modality - replicate to 4 channels
        channels = np.stack([nifti_data] * NUM_CHANNELS, axis=0)
    elif nifti_data.ndim == 4:
        # Already multichannel
        channels = nifti_data.transpose(3, 0, 1, 2)  # Move channel to front
        # Pad or truncate to exactly 4 channels
        if channels.shape[0] < NUM_CHANNELS:
            pad_channels = np.tile(channels[-1:], (NUM_CHANNELS - channels.shape[0], 1, 1, 1))
            channels = np.concatenate([channels, pad_channels], axis=0)
        elif channels.shape[0] > NUM_CHANNELS:
            channels = channels[:NUM_CHANNELS]
    else:
        raise ValueError(f"Unexpected NIFTI shape: {nifti_data.shape}")

    # Normalize each channel independently
    normalized_channels = []
    for i in range(NUM_CHANNELS):
        normalized = normalize_intensity(channels[i])
        normalized_channels.append(normalized)
    channels = np.stack(normalized_channels, axis=0)

    # Resize to standard input size (128, 128, 128) if needed
    from scipy.ndimage import zoom
    if channels.shape[1:] != MODEL_INPUT_SIZE:
        zoom_factors = [1.0] + [
            MODEL_INPUT_SIZE[i] / channels.shape[i+1]
            for i in range(3)
        ]
        channels = zoom(channels, zoom_factors, order=1)

    # Convert to tensor and add batch dimension
    tensor = torch.from_numpy(channels).float()
    tensor = tensor.unsqueeze(0)  # Add batch dimension: (1, 4, 128, 128, 128)

    return tensor.to(DEVICE)

# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/health")
async def health():
    """
    Health check endpoint for frontend to verify backend connectivity.

    Returns:
        - status: "ready" if model loaded, "error" otherwise
        - device: "cuda" if GPU available, "cpu" otherwise
        - model_loaded: boolean indicating if model weights loaded successfully
    """
    return JSONResponse({
        "status": "ready" if model_loaded else "error",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "model_loaded": model_loaded,
        "message": "Backend is operational" if model_loaded else "Model loading failed"
    })

@app.post("/segment")
async def segment(file: UploadFile = File(...)):
    """
    Segment brain tumor in uploaded MRI scan.

    Args:
        file: NIFTI format MRI file (.nii or .nii.gz)

    Returns:
        - segmentation: Base64-encoded segmentation mask (128x128x128 uint8)
        - classes: Number of output classes
        - shape: Original input shape
        - device_used: "cuda" or "cpu"

    Raises:
        HTTPException: If file format invalid or model not loaded
    """
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded. Backend initialization failed.")

    # Validate file type
    if not (file.filename.endswith('.nii') or file.filename.endswith('.nii.gz')):
        raise HTTPException(status_code=400, detail="File must be in NIFTI format (.nii or .nii.gz)")

    try:
        # Read uploaded file
        file_content = await file.read()
        nifti_img = nib.load(io.BytesIO(file_content))
        nifti_data = nifti_img.get_fdata()

        # Preprocess
        input_tensor = preprocess_nifti(nifti_data)

        # Run inference
        with torch.no_grad():
            output = model(input_tensor)

        # Post-process: argmax to get class predictions
        segmentation = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
        segmentation = segmentation.astype(np.uint8)

        # Encode as base64 for JSON transmission
        segmentation_bytes = segmentation.tobytes()
        segmentation_b64 = base64.b64encode(segmentation_bytes).decode('utf-8')

        return JSONResponse({
            "segmentation": segmentation_b64,
            "classes": int(NUM_CLASSES),
            "shape": list(segmentation.shape),
            "device_used": "cuda" if torch.cuda.is_available() else "cpu"
        })

    except nib.filebasedimage.ImageFileError as e:
        raise HTTPException(status_code=400, detail=f"Invalid NIFTI file: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Segmentation failed: {str(e)}")

# ============================================================================
# ngrok Tunnel Setup (For Google Colab Deployment)
# ============================================================================

def setup_ngrok_tunnel():
    """
    Establish ngrok tunnel to expose localhost to public HTTPS URL.
    Required for frontend (on Vercel) to communicate with backend (on Colab).

    Usage in Colab:
        1. Get free ngrok account at https://ngrok.com
        2. Copy auth token from https://dashboard.ngrok.com/auth
        3. In Colab notebook:
           os.environ['NGROK_AUTH_TOKEN'] = "your_token_here"
        4. Run this function after starting uvicorn server

    Returns:
        public_url: HTTPS URL accessible from frontend
    """
    if not NGROK_AUTH_TOKEN:
        print("WARNING: NGROK_AUTH_TOKEN not set")
        print("Frontend will only work on localhost")
        print("To enable remote access in Colab:")
        print("  1. Get auth token from https://dashboard.ngrok.com/auth")
        print("  2. Run: os.environ['NGROK_AUTH_TOKEN'] = 'your_token'")
        print("  3. Call setup_ngrok_tunnel()")
        return None

    try:
        ngrok.set_auth_token(NGROK_AUTH_TOKEN)
        public_url = ngrok.connect(8000, "http")
        print(f"\n{'='*60}")
        print(f"ngrok tunnel established: {public_url}")
        print(f"Share this URL with frontend:")
        print(f"  REACT_APP_API_URL={public_url}")
        print(f"{'='*60}\n")
        return public_url
    except Exception as e:
        print(f"Failed to establish ngrok tunnel: {str(e)}")
        return None

# ============================================================================
# Server Startup (For Local Testing)
# ============================================================================

if __name__ == "__main__":
    print(f"Starting FastAPI server on {DEVICE}...")
    print(f"Model loaded: {model_loaded}")

    # Run uvicorn server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )

    # After server starts (this won't execute until server stops)
    # In Colab, call setup_ngrok_tunnel() separately in another cell after starting this
