# SAM-SEM Segmentation Toolkit

> **AI-powered segmentation and analysis of particles in electron microscopy images using Meta's Segment Anything Model (SAM)**

This toolkit provides both a **Python package** for programmatic access and a **web-based GUI** for interactive analysis.

---

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Create conda environment
conda create -n SEM_analysis python=3.11
conda activate SEM_analysis

# Install PyTorch (choose based on your system)
# For RTX 5080 (Blackwell architecture):
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128

# For RTX 30/40 series:
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia

# For Apple Silicon (M1/M2/M3/M4):
conda install pytorch torchvision -c pytorch

# For CPU only:
conda install pytorch torchvision cpuonly -c pytorch

# Install dependencies
pip install -r requirements.txt

# Install the package (editable mode)
cd sem_particle_analysis
pip install -e .
cd ..
```

### 2. Download SAM Weights

```bash
# Download ViT-H (best quality, 2.4GB)
python download_sam_weights.py

# Weights will be saved to: sam_weights/
```

### 3. Run the Web Application

**Windows:**
```bash
run_app.bat
```

**macOS/Linux:**
```bash
python sem_analysis_app/sem_analysis_app.py
```

Open your browser to `http://127.0.0.1:7860`

### 4. Use as Python Package

```python
from sem_particle_analysis import SAMModel, ParticleAnalyzer

# Your code here
```

---

## 📦 What's Included

### 1. **Gradio Web Application** (`sem_analysis_app/`)

A beautiful, production-ready web interface with:

- **🤖 AI-Powered Segmentation**: Automatic particle detection using SAM
- **📏 Auto Scale Detection**: OCR-based scale bar recognition and calibration
- **✏️ Interactive Refinement**: Add, delete, merge particles; point-based refinement with live preview
- **📊 Real-time Analysis**: Particle measurements with histograms and statistics
- **💾 Batch Processing**: Process multiple images with session tracking
- **📈 Results Management**: CSV export, duplicate removal, row deletion
- **↩️ Undo/Redo**: Click-level undo for refinement operations
- **🎯 Advanced Features**: Edge particle removal, particle number toggle, size filtering

See [`sem_analysis_app/README.md`](sem_analysis_app/README.md) for detailed usage instructions.

### 2. **Python Package** (`sem_particle_analysis/`)

A clean, modular Python library for programmatic access:

- Scale detection and image preprocessing
- SAM-based particle segmentation
- Particle analysis and measurements
- Results export to CSV
- Interactive Jupyter notebook widgets (legacy)

See [`sem_particle_analysis/README.md`](sem_particle_analysis/README.md) for API documentation.

---

## 🎯 Key Features

### Automatic Scale Detection
- OCR-based scale bar detection using EasyOCR
- Support for both horizontal and vertical scale bars
- Manual override option for non-standard scales

### Advanced Particle Refinement
- **Delete Mode**: Click particles to remove false positives
- **Add Mode**: Click to add missed particles
- **Merge Mode**: Combine touching particles
- **Point Refine Mode**: Iterative refinement with positive/negative points
- Undo individual clicks before applying changes
- Real-time visualization with live previews

### Comprehensive Analysis
- Particle count and size distribution
- Area measurements (pixels and nm²)
- Equivalent diameter calculations
- Summary statistics (mean, median, std, min, max)
- Aggregate statistics across all images in session

### Results Management
- Auto-save to CSV after each image
- Session-wide tracking and export
- Duplicate detection and removal
- Individual row deletion
- State persistence across sessions

---

## 💻 System Requirements

- **Python**: 3.11 or higher
- **GPU**: Optional but recommended
  - NVIDIA GPU with CUDA support (RTX 5080, 4090, 3090, etc.)
  - Apple Silicon (M1/M2/M3/M4) with MPS support
  - CPU fallback available (slower)
- **RAM**: 8GB minimum, 16GB+ recommended
- **Storage**: ~3GB for SAM model + your images

### Performance Expectations
- **RTX 5080 (CUDA)**: 1-3 seconds per image
- **Apple Silicon (MPS)**: 2-5 seconds per image
- **CPU**: 10-30 seconds per image

---

## 📚 Documentation

- **Web App Guide**: [`sem_analysis_app/README.md`](sem_analysis_app/README.md)
- **Python API**: [`sem_particle_analysis/README.md`](sem_particle_analysis/README.md)

---

## 🔧 Supported Image Formats

- `.tif`, `.tiff`
- `.png`
- `.jpg`, `.jpeg`

---

## 📊 Output Data

Results are exported as CSV files containing:
- Filename
- Particle count
- Individual particle areas (pixels and nm²)
- Equivalent diameters (pixels and nm)
- Easy integration with Excel, Python, R, etc.

---

## 🎓 Citation

This tool uses Meta's Segment Anything Model (SAM):

```bibtex
@article{kirillov2023segment,
  title={Segment Anything},
  author={Kirillov, Alexander and Mintun, Eric and Ravi, Nikhila and Mao, Hanzi and Rolland, Chloe and Gustafson, Laura and Xiao, Tete and Whitehead, Spencer and Berg, Alexander C. and Lo, Wan-Yen and Doll{\'a}r, Piotr and Girshick, Ross},
  journal={arXiv:2304.02643},
  year={2023}
}
```

---

## 📝 License

MIT License - See LICENSE file for details

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## 🙏 Acknowledgments

Built with:
- [Segment Anything Model](https://segment-anything.com/) by Meta AI
- [Gradio](https://gradio.app/) for the web interface
- [EasyOCR](https://github.com/JaidedAI/EasyOCR) for scale detection
- [scikit-image](https://scikit-image.org/) for image processing
