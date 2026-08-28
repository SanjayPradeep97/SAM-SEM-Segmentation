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

# Install the package and all dependencies
pip install -e .
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
python -m sem_analysis_app
```

Open your browser to `http://127.0.0.1:7860`. Use `--port` / `--host` to change
where it listens.

### 4. Establishing Scale

Scale gets its own tab, because every measurement is a pixel count multiplied by
this one number. Three tiers, tried in order of trustworthiness:

| Tier | Method | When it applies |
| --- | --- | --- |
| 1 | Pixel size read from the file's own metadata | Automatic on load. Exact — nothing to check |
| 2 | Read the printed scale bar inside a box you draw | Runs automatically first; drag the box if it got it wrong |
| 3 | Click both ends of the bar and type its length | When the bar is unreadable. A magnifier follows the cursor so you land on the exact pixel |

Tier 2's box is manipulated directly on the image: drag anywhere to draw one,
drag a corner to resize, drag the middle to move. Tier 3 shows a 7x loupe with a
crosshair marking precisely which pixel a click will land on.

A tier-2 reading is marked unconfirmed until you press **Confirm scale**; tier 1
and tier 3 are trusted outright, since neither involves a machine reading a
glyph. The result is shown as "X nm/px · how it was obtained", and it is what the
Processing tab and the results CSV use. Nothing downstream re-detects scale or
overrides what you set here.

Calibration needs only OCR, so you can sort out scale before loading a SAM
checkpoint.

### 5. Batch Analysis Without the GUI

For a dataset you intend to publish, run the pipeline headless. Every run writes
a `run.json` recording the image hashes, model and checkpoint hash, the scale and
how it was obtained, all parameters, library versions and the git revision — so a
result can be traced back to exactly what produced it.

```bash
sem-analyze data/raw/sample-a -o data/processed/sample-a --clear-edges
```

Outputs: `particles.csv` (one row per particle), `per_image_summary.csv`,
`size_distribution.png/.pdf` at 300 dpi, and `run.json`.

Useful flags:

| Flag | Effect |
| --- | --- |
| `--scale-nm-per-px X` | Fix the scale instead of detecting it per image |
| `--scale-method metadata\|ocr\|auto` | How to establish scale (default `auto`) |
| `--clear-edges` | Drop particles touching the frame; they are only partly imaged |
| `--min-size N` | Ignore particles below N pixels (default 30) |
| `--crop-percent P` | Override databar removal (default: measure it) |
| `--model-type vit_b` | Faster, lower quality than the default `vit_h` |

Note that batch mode uses the automatic pipeline only — no interactive
refinement — and picks its mask candidate by heuristic. The chosen mask and the
rejected candidates are recorded in `run.json`.

### 6. Use as Python Package

```python
from sem_particle_analysis import SAMModel, ParticleAnalyzer

# Your code here
```

### 7. Run the Tests

```bash
pytest                      # everything
pytest -m "not slow"        # skip tests needing model weights or OCR
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

Module layout:

```
sem_analysis_app/
├── __main__.py       entry point (python -m sem_analysis_app)
├── ui.py             tab layout and event wiring
├── state.py          shared application state
├── callbacks/        event handlers, one module per tab
└── visualization.py  overlays and figures
```

The app keeps a single process-wide state object, so it is built for one analyst
at a time; two browser tabs pointed at the same server share one session.

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
