# Quick Start Guide

Get up and running with the CNT Particle Analysis app in 5 minutes!

## Prerequisites Checklist

- [ ] Python 3.8+ installed
- [ ] SAM checkpoint file downloaded
- [ ] Folder of SEM/TEM images ready

## Installation (5 minutes)

### Step 1: Install Dependencies

```bash
# Navigate to the app folder
cd cnt_analysis_app

# Install requirements
pip install -r requirements.txt
```

### Step 2: Download SAM Weights

```bash
# Download ViT-H model (recommended, ~2.5GB)
curl -O https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

# Or download ViT-B model (faster, ~375MB)
curl -O https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```

## Launch the App

```bash
python app.py
```

Open your browser to: **http://127.0.0.1:7860**

## First Time Workflow

### 1. Configuration (one-time setup)

**SAM Model Path**: Enter the full path to your downloaded checkpoint
```
Example: /Users/yourname/Downloads/sam_vit_h_4b8939.pth
```

**Model Type**: Select `vit_h` (or `vit_b` if using the smaller model)

**Click**: ⚡ Initialize SAM Model

Wait for "✅ SAM model initialized successfully"

**Image Folder**: Enter path to your images
```
Example: /Users/yourname/Documents/SEM_Images
```

**Click**: 📁 Load Images

### 2. Process First Image

**Step 1**: Click "🔍 Detect Scale"
- Wait for scale detection to complete
- Verify the detected scale looks correct

**Step 2**: Click "🤖 Segment with SAM"
- Wait ~10-30 seconds for segmentation
- You'll see 3 mask options

**Step 3**: Select the best mask
- Look at all 3 visualizations
- Choose the one that captures particles best
- Click "✓ Select Mask"

**Step 4**: Click "🔬 Analyze Particles"
- View detected particles with numbered labels
- See measurements table
- Results automatically saved!

**Step 5**: Click "Next ►" to process next image

### 3. Batch Processing

Repeat steps from "Process First Image" for each image:
1. Detect Scale
2. Segment with SAM
3. Select Mask
4. Analyze Particles
5. Next Image

### 4. Export Results

**Click**: 🔄 Refresh Summary
- See how many images processed
- View total particle counts

**Click**: 💾 Export Results CSV
- Download complete results file

## Tips

### Faster Processing
- Use ViT-B model instead of ViT-H
- Process on GPU if available
- Skip images with "Next ►" button

### Better Results
- Always verify scale detection is correct
- Compare all 3 mask candidates before selecting
- Use manual scale if auto-detection fails

### Common Shortcuts

**Navigation**:
- Previous image: Click "◄ Previous"
- Next image: Click "Next ►"
- Jump to specific: Enter number, click "Jump"

**Quick Process**:
1. Detect Scale → 2. Segment → 3. Select Mask 1 → 4. Analyze → 5. Next

## Example Session

**Typical workflow for 50 images:**
1. One-time setup (2 minutes)
2. Process images (8-10 minutes @ 10-12 seconds per image)
3. Export results (5 seconds)

**Total time: ~10-15 minutes for 50 images**

## Need Help?

### Scale Detection Issues
- **Problem**: "Scale bar not detected"
- **Solution**: Switch to "Manual" mode and enter the scale value (in nm)

### Segmentation Issues
- **Problem**: SAM takes too long
- **Solution**: Use ViT-B model, or ensure GPU is being used

### No Particles Found
- **Problem**: Analysis shows 0 particles
- **Solution**: Try a different mask (Mask 2 or Mask 3)

## What Next?

1. **Check your CSV file**: Find `analysis_results.csv` in your image folder
2. **Open in Excel/Python**: Analyze particle size distributions
3. **Process more batches**: Point to different image folders
4. **Customize**: Adjust parameters in the code if needed

---

Happy analyzing! 🔬
