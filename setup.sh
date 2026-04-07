#!/bin/bash
# setup.sh — One-command setup for Brain Tumor Segmentation Project
# ML Project by Rahul & Krishnaa for Dr. Valarmathi

set -e

echo "╔══════════════════════════════════════════════════════════╗"
echo "║  Brain Tumor Segmentation — Setup Script                ║"
echo "║  ML Project by Rahul & Krishnaa for Dr. Valarmathi      ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

# 1. Check Python
echo -e "${CYAN}[1/5] Checking Python...${NC}"
if command -v python3 &> /dev/null; then
    PYTHON=python3
elif command -v python &> /dev/null; then
    PYTHON=python
else
    echo "❌ Python not found. Install Python 3.10+ first."
    exit 1
fi
echo -e "${GREEN}✅ Python: $($PYTHON --version)${NC}"

# 2. Create virtual environment
echo -e "${CYAN}[2/5] Setting up virtual environment...${NC}"
if [ ! -d "venv" ]; then
    $PYTHON -m venv venv
    echo -e "${GREEN}✅ Virtual environment created${NC}"
else
    echo -e "${YELLOW}⏩ Virtual environment already exists${NC}"
fi

# Activate
source venv/bin/activate

# 3. Install backend dependencies
echo -e "${CYAN}[3/5] Installing backend dependencies...${NC}"
pip install --upgrade pip -q
pip install -r backend/requirements.txt -q

# Install CPU PyTorch if not present
$PYTHON -c "import torch" 2>/dev/null || {
    echo "Installing PyTorch (CPU)..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu -q
}
echo -e "${GREEN}✅ Backend dependencies installed${NC}"

# 4. Create model directory
echo -e "${CYAN}[4/6] Setting up model directory...${NC}"
mkdir -p models
echo -e "${YELLOW}ℹ  Place trained model weights in ./models/${NC}"
echo "   - unet3d_brats.pth (segmentation)"
echo "   - grading_classifier.pkl (grading)"
echo "   Or run the Google Colab notebook to generate them."

# 5. Kaggle credentials
echo -e "${CYAN}[5/6] Checking Kaggle credentials...${NC}"
if [ -f "$HOME/.kaggle/kaggle.json" ]; then
    echo -e "${GREEN}✅ Found existing ~/.kaggle/kaggle.json${NC}"
elif [ -n "$KAGGLE_USERNAME" ] && [ -n "$KAGGLE_KEY" ]; then
    mkdir -p "$HOME/.kaggle"
    echo "{\"username\":\"$KAGGLE_USERNAME\",\"key\":\"$KAGGLE_KEY\"}" > "$HOME/.kaggle/kaggle.json"
    chmod 600 "$HOME/.kaggle/kaggle.json"
    echo -e "${GREEN}✅ Created kaggle.json from env vars${NC}"
else
    echo -e "${YELLOW}⚠  Kaggle credentials not found.${NC}"
    echo "   To enable real BraTS dataset download:"
    echo "   1. Go to https://www.kaggle.com/settings → API → Create New Token"
    echo "   2. Save kaggle.json to ~/.kaggle/kaggle.json"
    echo "   3. chmod 600 ~/.kaggle/kaggle.json"
    echo ""
    echo "   Or set environment variables:"
    echo "   export KAGGLE_USERNAME=your_username"
    echo "   export KAGGLE_KEY=your_api_key"
    echo ""
    echo "   Without Kaggle, the backend still works using synthetic/fallback data."
fi

# 6. Test backend
echo -e "${CYAN}[6/6] Testing backend startup...${NC}"
cd backend
$PYTHON -c "
from app.main import app
print('FastAPI app loaded successfully')
from app.models.model_manager import ModelManager
mm = ModelManager()
mm.load_all()
print('Models initialized (fallback mode if no weights)')
"
cd ..

echo ""
echo -e "${GREEN}╔══════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║  ✅ Setup Complete!                                     ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════════╝${NC}"
echo ""
echo "To start the backend:"
echo -e "  ${CYAN}source venv/bin/activate${NC}"
echo -e "  ${CYAN}cd backend && uvicorn app.main:app --reload --port 8000${NC}"
echo ""
echo "To download BraTS dataset (requires kaggle.json):"
echo -e "  ${CYAN}python scripts/download_brats.py --output ./data --dataset brats2024${NC}"
echo ""
echo "To start the frontend (open in Claude.ai or copy to your project):"
echo -e "  ${CYAN}The frontend/index.jsx is a React component${NC}"
echo -e "  ${CYAN}Deploy as a standalone artifact or integrate into Next.js${NC}"
echo ""
echo "API Documentation:"
echo -e "  ${CYAN}http://localhost:8000/docs${NC}"
echo ""
echo "Google Colab Training:"
echo -e "  ${CYAN}Upload notebooks/training.ipynb to Google Colab${NC}"
echo ""
