# SEM Particle Analysis - User Guide

## Table of Contents
1. [Getting Started](#getting-started)
2. [Interface Overview](#interface-overview)
3. [Detailed Workflow](#detailed-workflow)
4. [Understanding Results](#understanding-results)
5. [Tips & Best Practices](#tips--best-practices)
6. [FAQ](#faq)

---

## Getting Started

### First Launch

1. **Open Terminal/Command Prompt**
   ```bash
   cd sem_analysis_app
   python sem_analysis_app.py
   ```

2. **Open Browser**
   - Navigate to: `http://127.0.0.1:7860`
   - You should see the SEM Particle Analysis interface

3. **Have Ready**
   - Path to SAM checkpoint file (e.g., `sam_vit_h_4b8939.pth`)
   - Path to folder containing your SEM/TEM images

---

## Interface Overview

The application is divided into 6 main sections:

### 1. Configuration (Top)
**Purpose**: One-time setup for each session

**Fields**:
- **SAM Model Checkpoint Path**: Full path to `.pth` file
- **Model Type**: Choose `vit_h` (best quality) or `vit_b` (faster)
- **Image Folder Path**: Folder containing your images

**Buttons**:
- `⚡ Initialize SAM Model`: Loads the AI model (takes ~10 seconds)
- `📁 Load Images`: Scans folder for images

### 2. Image Processing
**Purpose**: Navigate through your image collection

**Controls**:
- `◄ Previous`: Go to previous image
- `Jump to Image #`: Enter number and click "Jump"
- `Next ►`: Go to next image

**Display**: Shows current image and filename

### 3. Scale Detection
**Purpose**: Calibrate measurements using scale bar

**Auto Mode**:
- Click `🔍 Detect Scale`
- System reads scale bar text automatically
- Shows detected scale (e.g., "500 nm")

**Manual Mode**:
- Select "Manual" radio button
- Enter scale value in nm
- Click "Set Manual Scale"

### 4. Segmentation
**Purpose**: AI-powered particle detection

**Process**:
1. Click `🤖 Segment with SAM`
2. Wait for 3 mask candidates to appear
3. Review all 3 masks visually
4. Select best mask using radio buttons
5. Click `✓ Select Mask`

**What to Look For**:
- Mask that captures all particles
- Clear separation between particles
- Minimal false positives

### 5. Analysis
**Purpose**: Extract particle measurements

**Process**:
1. Click `🔬 Analyze Particles`
2. View visualization with numbered particles
3. Review measurements in tables

**Two Tabs**:
- **Particle Measurements**: Individual particle data
- **Summary Statistics**: Mean, median, std dev, etc.

### 6. Session Summary
**Purpose**: Track progress across all images

**Features**:
- Click `🔄 Refresh Summary` to update
- Shows images processed, total particles
- Export all results with `💾 Export Results CSV`

---

## Detailed Workflow

### Complete Workflow for One Image

```
┌─────────────────────────────────────────┐
│ Step 1: Load Image                      │
│ - Navigate to desired image             │
│ - View original in display              │
└────────────┬────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────┐
│ Step 2: Detect Scale                    │
│ - Click "Detect Scale"                  │
│ - Verify detected value is correct      │
│ - Use manual if auto fails              │
└────────────┬────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────┐
│ Step 3: Segment Particles               │
│ - Click "Segment with SAM"              │
│ - Wait for 3 masks to generate          │
│ - Review mask visualizations            │
└────────────┬────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────┐
│ Step 4: Select Best Mask                │
│ - Compare scores (higher is better)     │
│ - Visually inspect all 3 options        │
│ - Choose mask, click "Select Mask"      │
└────────────┬────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────┐
│ Step 5: Analyze                         │
│ - Click "Analyze Particles"             │
│ - Review visualization                  │
│ - Check measurements table              │
│ - Results auto-saved to CSV             │
└────────────┬────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────┐
│ Step 6: Next Image                      │
│ - Click "Next ►"                        │
│ - Repeat from Step 2                    │
└─────────────────────────────────────────┘
```

### Batch Processing Multiple Images

**Setup Phase** (One Time):
1. Initialize SAM model
2. Load image folder

**For Each Image** (Repeat):
1. Current image loads automatically
2. Detect Scale → Segment → Select Mask → Analyze
3. Click "Next ►"

**Completion Phase**:
1. Click "Refresh Summary"
2. Review statistics
3. Export CSV

**Time Estimate**: ~30-40 seconds per image

---

## Understanding Results

### Particle Measurements Table

**Columns**:
- **Particle ID**: Sequential number (1, 2, 3, ...)
- **Area (nm²)**: Particle area in square nanometers
- **Diameter (nm)**: Equivalent circular diameter

**Example**:
```
Particle ID | Area (nm²) | Diameter (nm)
------------|------------|---------------
1           | 1245.3     | 39.8
2           | 892.1      | 33.7
3           | 2103.5     | 51.7
```

### Summary Statistics

**Metrics Explained**:
- **Mean**: Average value across all particles
- **Median**: Middle value (50th percentile)
- **Std Dev**: Standard deviation (spread)
- **Min**: Smallest particle
- **Max**: Largest particle

**Interpretation**:
- Large std dev → wide size distribution
- Mean ≈ Median → symmetrical distribution
- Mean >> Median → some very large particles

### CSV Output Format

**File Location**: Same folder as images, named `analysis_results.csv`

**Columns**:
1. `file_name`: Image filename
2. `num_particles`: Particle count
3. `particle_areas_px`: Areas in pixels (list)
4. `equiv_diameters_px`: Diameters in pixels (list)
5. `particle_areas_nm2`: Areas in nm² (list)
6. `equiv_diameters_nm`: Diameters in nm (list)

**Usage**:
- Open in Excel for further analysis
- Import into Python/R for statistics
- Create size distribution plots

---

## Tips & Best Practices

### For Best Results

#### Scale Detection
✅ **DO**:
- Verify detected scale matches image
- Use manual mode if auto-detection unreliable
- Check scale bar is clearly visible

❌ **DON'T**:
- Proceed with wrong scale (measurements will be incorrect)
- Ignore detection errors
- Use damaged scale bars

#### Mask Selection
✅ **DO**:
- Always review all 3 masks
- Choose mask that captures all particles
- Prefer masks with high scores (0.9+)

❌ **DON'T**:
- Always pick Mask 1 without looking
- Select masks with obvious errors
- Ignore score information

#### Particle Analysis
✅ **DO**:
- Visually verify particle count seems correct
- Check that all visible particles are detected
- Review a few measurements for sanity

❌ **DON'T**:
- Accept obviously wrong results
- Process too fast without verification
- Skip quality control checks

### Keyboard Shortcuts

Unfortunately, Gradio doesn't support custom keyboard shortcuts, but you can:
- Use **Tab** to move between fields
- Use **Enter** to activate focused button
- Use **Browser's refresh** (F5) to restart app

### Performance Optimization

**For Faster Processing**:
1. Use `vit_b` model instead of `vit_h`
2. Process on GPU if available
3. Close other memory-intensive applications
4. Process smaller images if possible

**For Better Quality**:
1. Use `vit_h` model (slower but more accurate)
2. Ensure images are high resolution
3. Use clear, well-contrasted images
4. Verify scale detection for each image

### Data Organization

**Recommended Folder Structure**:
```
Project/
├── raw_images/
│   ├── sample_001.tif
│   ├── sample_002.tif
│   └── ...
├── analysis_results.csv
└── sam_models/
    └── sam_vit_h_4b8939.pth
```

**Backup Strategy**:
- Export CSV regularly
- Keep original images separate
- Version control your results

---

## FAQ

### General Questions

**Q: How long does it take to process one image?**
A: Typically 30-40 seconds with GPU, 1-2 minutes with CPU.

**Q: Can I process images in different folders?**
A: Yes, just load a new folder. Results will be separate.

**Q: Is my data sent to the cloud?**
A: No, everything runs locally on your computer.

**Q: What image formats are supported?**
A: TIF, TIFF, PNG, JPG, JPEG

### Troubleshooting

**Q: "SAM checkpoint not found" error**
A:
- Verify the file path is correct
- Use absolute path (e.g., `/Users/name/sam_vit_h.pth`)
- Check file exists and is downloaded completely

**Q: Scale detection keeps failing**
A:
- Switch to Manual mode
- Measure scale bar length in pixels
- Enter known scale value in nm

**Q: No particles detected after analysis**
A:
- Try selecting a different mask (Mask 2 or 3)
- Check that scale was detected correctly
- Verify image has visible particles

**Q: App is very slow**
A:
- Use smaller `vit_b` model
- Close other applications
- Check if GPU is being utilized
- Process smaller images

**Q: Results look wrong**
A:
- Verify scale detection is correct
- Check selected mask captures particles well
- Review visualization for obvious errors

**Q: Can't export CSV**
A:
- Check folder permissions
- Ensure disk space is available
- Try exporting to different location

### Advanced Usage

**Q: Can I adjust particle detection parameters?**
A: Currently hardcoded. Edit `app.py` line ~520 to change `min_area`, `min_size`, etc.

**Q: How do I process thousands of images?**
A: Consider writing a batch script or using the underlying Python package directly.

**Q: Can I run this on a server?**
A: Yes, modify `app.launch()` parameters to allow remote access.

**Q: How do I customize the visualization?**
A: Edit `visualization.py` to change colors, styles, etc.

### Data Questions

**Q: What units are measurements in?**
A: All measurements are in nanometers (nm) after scale calibration.

**Q: How is diameter calculated?**
A: Equivalent circular diameter: diameter of circle with same area.

**Q: Can I get other measurements (perimeter, etc.)?**
A: Not in current UI, but available via Python API (see `ParticleAnalyzer`).

**Q: How do I analyze the CSV in Python?**
A:
```python
import pandas as pd
df = pd.read_csv('analysis_results.csv')
print(df['num_particles'].sum())  # Total particles
```

---

## Getting Help

### Support Resources

1. **Documentation**:
   - README.md: Installation and overview
   - QUICKSTART.md: 5-minute guide
   - This file: Detailed usage

2. **Code Examples**:
   - Check `sem_particle_analysis/examples/`
   - Review notebook workflows

3. **Community**:
   - GitHub Issues for bugs
   - Discussions for questions

### Reporting Issues

When reporting problems, include:
- Error message (full text)
- Steps to reproduce
- Image characteristics (size, format)
- System info (OS, Python version, GPU)

### Feature Requests

Suggest new features via GitHub Issues with:
- Use case description
- Expected behavior
- Why it would be useful

---

## Next Steps

### After Processing Your First Batch

1. **Analyze Results**:
   - Open CSV in Excel
   - Calculate statistics
   - Create plots

2. **Optimize Workflow**:
   - Note which mask typically works best
   - Identify common scale bar issues
   - Develop quality control checklist

3. **Share with Team**:
   - Train colleagues on the tool
   - Establish standard operating procedures
   - Document project-specific settings

### Going Further

- Explore the Python API for custom workflows
- Automate repetitive tasks
- Integrate with your data analysis pipeline

---

**Happy Analyzing!** 🔬

For updates and more information, visit the GitHub repository.
