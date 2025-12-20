# CNT Particle Analysis Toolkit

> **AI-powered segmentation and analysis of carbon nanotube (CNT) particles in electron microscopy images using Meta's Segment Anything Model (SAM)**

This toolkit provides both a **Python package** for programmatic access and a **web-based GUI** for interactive analysis.

---

## 🚀 Quick Start

### Option 1: Web Application (Recommended for most users)

```bash
# Navigate to the app directory
cd cnt_analysis_app

# Install dependencies
pip install -r requirements.txt

# Download SAM model
curl -O https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

# Launch the web interface
python app_v2.py
```

Open your browser to `http://127.0.0.1:7860`

### Option 2: Python Package

```bash
# Navigate to the package directory
cd sem_particle_analysis

# Install the package
pip install -e .

# Use in your Python code
from sem_particle_analysis import SAMModel, ParticleAnalyzer
```

---

## 📦 What's Included

### 1. **Gradio Web Application** (`cnt_analysis_app/`)

A beautiful, production-ready web interface with:

- **🤖 AI-Powered Segmentation**: Automatic particle detection using SAM
- **📏 Auto Scale Detection**: OCR-based scale bar recognition and calibration
- **✏️ Interactive Refinement**: Add, delete, merge particles; point-based refinement with live preview
- **📊 Real-time Analysis**: Particle measurements with histograms and statistics
- **💾 Batch Processing**: Process multiple images with session tracking
- **📈 Results Management**: CSV export, duplicate removal, row deletion
- **↩️ Undo/Redo**: Click-level undo for refinement operations
- **🎯 Advanced Features**: Edge particle removal, particle number toggle, size filtering

See [`cnt_analysis_app/README.md`](cnt_analysis_app/README.md) for detailed usage instructions.

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

- **Python**: 3.8 or higher
- **GPU**: Optional but recommended (CUDA or Apple Silicon MPS)
- **RAM**: 8GB minimum, 16GB+ recommended
- **Storage**: ~3GB for SAM model + your images

---

## 📚 Documentation

- **Web App Guide**: [`cnt_analysis_app/README.md`](cnt_analysis_app/README.md)
- **Quick Start**: [`cnt_analysis_app/QUICKSTART.md`](cnt_analysis_app/QUICKSTART.md)
- **User Guide**: [`cnt_analysis_app/USER_GUIDE.md`](cnt_analysis_app/USER_GUIDE.md)
- **Python API**: [`sem_particle_analysis/README.md`](sem_particle_analysis/README.md)
- **Installation**: [`sem_particle_analysis/INSTALL.md`](sem_particle_analysis/INSTALL.md)

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
