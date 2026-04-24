#!/bin/bash
# CHTC Environment Setup Script

set -e  # Exit on error

echo "=========================================="
echo "Setting up ORCA + LoRA environment"
echo "=========================================="

# Extract project archive
echo "Extracting project files..."
tar -xzf orca_lora_project.tar.gz

# Create virtual environment
echo "Creating Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install PyTorch (CPU version for CHTC)
echo "Installing PyTorch..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies
echo "Installing other dependencies..."
pip install -r requirements.txt

# Install OTDD package
echo "Installing OTDD..."
cd ORCA/src/otdd
pip install -e .
cd ../../..

# Download alignment data
echo "Downloading alignment data..."
mkdir -p datasets
cd datasets
if [ ! -f text_xs.npy ]; then
    gdown 1NzLuAEkF6r5v-iCbbTB7bb9-l6liweVo  # text_xs.npy
fi
if [ ! -f text_ys.npy ]; then
    gdown 1skrhO9OKFgtE1cr2BeOcv2AQdGomTcg1  # text_ys.npy
fi
cd ..

# Download real ECG dataset
echo "Downloading ECG dataset..."
cd datasets
if [ ! -f challenge2017.pkl ]; then
    # Try Google Drive download (adjust file ID if needed)
    gdown --fuzzy "https://drive.google.com/file/d/FILEID/view?usp=sharing" -O challenge2017.pkl || \
    echo "Warning: Could not download ECG dataset automatically"
fi
cd ..

echo "=========================================="
echo "Setup complete!"
echo "=========================================="
