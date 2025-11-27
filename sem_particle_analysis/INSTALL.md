# Installation Guide

## Quick Installation

### 1. Install the package

```bash
cd sem_particle_analysis
pip install -e .
```

### 2. Download SAM model checkpoint

Choose one based on your needs:

**ViT-H (Recommended - Best Quality)**
```bash
curl -O https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

**ViT-L (Good Balance)**
```bash
curl -O https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth
```

**ViT-B (Fastest)**
```bash
curl -O https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```

### 3. Test the installation

```python
from sem_particle_analysis import SAMModel, ScaleDetector
print("Installation successful!")
```

## Detailed Installation

### Using conda (Recommended)

```bash
# Create environment
conda create -n sem_analysis python=3.10
conda activate sem_analysis

# Install PyTorch (adjust CUDA version as needed)
# For CUDA 11.8:
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia

# For CPU only:
conda install pytorch torchvision cpuonly -c pytorch

# Install the package
cd sem_particle_analysis
pip install -e .
```

### Using pip with virtualenv

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

### For Development

```bash
# Install with development dependencies
pip install -e ".[dev]"

# Run tests (when available)
pytest

# Format code
black sem_particle_analysis/
```

### For Jupyter Notebooks

```bash
# Install with Jupyter support
pip install -e ".[jupyter]"

# Launch Jupyter
jupyter lab
```

## Troubleshooting

### CUDA/GPU Issues

If you have GPU but it's not being detected:

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"MPS available: {torch.backends.mps.is_available()}")
```

Reinstall PyTorch with the correct CUDA version from [pytorch.org](https://pytorch.org/get-started/locally/)

### EasyOCR Installation Issues

If EasyOCR fails to install:

```bash
pip install --upgrade pip
pip install easyocr --no-cache-dir
```

### OpenCV Issues

If you get OpenCV import errors:

```bash
pip uninstall opencv-python opencv-python-headless
pip install opencv-python
```

## System Requirements

- **Python**: 3.8 or higher
- **RAM**: Minimum 8GB, 16GB+ recommended
- **GPU**: Optional but recommended (NVIDIA with CUDA or Apple Silicon)
- **Storage**: ~3GB for ViT-H model, more for processed images

## Verifying Installation

Run this test script:

```python
from sem_particle_analysis import (
    SAMModel,
    ScaleDetector,
    ParticleSegmenter,
    ParticleAnalyzer,
    ResultsManager
)

print("All modules imported successfully!")
print("Installation verified.")
```
