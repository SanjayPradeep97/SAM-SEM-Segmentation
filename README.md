# CNT Particle Segmentation Tool

Interactive tool for segmenting and analyzing carbon nanotube particles in electron microscopy images using Meta's Segment Anything Model (SAM).

## What It Does

- Loads SEM/TEM images and detects scale bars automatically
- Uses SAM to segment CNT particles with interactive mask selection
- Counts particles and measures sizes (area, diameter)
- Exports results to CSV for batch analysis

## Requirements

- Python 3.10+
- PyTorch
- Segment Anything Model
- OpenCV, scikit-image, ipywidgets

## Quick Start
```bash
# Create environment
conda create -n cnt_analysis python=3.10
conda activate cnt_analysis

# Install dependencies
conda install -c conda-forge numpy pandas matplotlib opencv scikit-image ipywidgets
pip install git+https://github.com/facebookresearch/segment-anything.git
pip install easyocr

# Download SAM model (choose one)
curl -O https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth

# Launch Jupyter
jupyter lab
```

## Usage

1. Run initialization cell to load SAM model
2. Set your image folder path
3. Process images interactively (Proceed/Skip/Jump)
4. Select best segmentation mask
5. Review particle measurements
6. Export results to CSV

## Citation

Built using Meta AI's Segment Anything Model (Kirillov et al., 2023).
