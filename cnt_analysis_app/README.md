# CNT Particle Analysis - Gradio Application

A beautiful, user-friendly web interface for segmenting and analyzing carbon nanotube (CNT) particles in electron microscopy images using Meta's Segment Anything Model (SAM).

## Features

- **🤖 AI-Powered Segmentation**: Leverages Meta's SAM for accurate particle detection
- **📏 Automatic Scale Detection**: Uses OCR to detect and calibrate scale bars
- **🔍 Interactive Mask Selection**: Choose from 3 AI-generated mask candidates
- **📊 Comprehensive Analysis**: Particle count, area, diameter measurements in nm
- **📁 Batch Processing**: Process multiple images sequentially with navigation
- **💾 Auto-Save Results**: Automatically exports data to CSV after each image
- **📈 Session Tracking**: View summary statistics across all processed images

## Screenshot

The application provides a complete workflow:
1. Configure SAM model and image folder
2. Navigate through images
3. Detect scale bars automatically
4. Segment particles with SAM
5. Select the best mask from 3 candidates
6. Analyze particles and view measurements
7. Export results to CSV

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (optional, recommended for faster processing)

### Step 1: Clone Repository

```bash
git clone <repository-url>
cd cnt_analysis_app
```

### Step 2: Create Virtual Environment

```bash
# Using conda (recommended)
conda create -n cnt_analysis python=3.10
conda activate cnt_analysis

# Or using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Download SAM Model Checkpoint

Download one of the SAM model checkpoints:

```bash
# ViT-H (best quality, ~2.5GB) - Recommended
curl -O https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

# ViT-L (good balance, ~1.2GB)
curl -O https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth

# ViT-B (fastest, ~375MB)
curl -O https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```

## Usage

### Launch the Application

```bash
python app.py
```

The application will start on `http://127.0.0.1:7860`

### Workflow

#### 1. **Configuration**
   - Enter the path to your SAM checkpoint file (e.g., `sam_vit_h_4b8939.pth`)
   - Select model type (ViT-H or ViT-B)
   - Click "⚡ Initialize SAM Model"
   - Enter path to your image folder
   - Click "📁 Load Images"

#### 2. **Process Images**
   - Navigate using Previous/Next buttons or jump to specific image
   - Click "🔍 Detect Scale" to automatically detect scale bar
   - Or select "Manual" mode to enter scale manually

#### 3. **Segment Particles**
   - Click "🤖 Segment with SAM" to generate 3 mask candidates
   - View all 3 masks with confidence scores
   - Select the best mask using radio buttons
   - Click "✓ Select Mask"

#### 4. **Analyze Results**
   - Click "🔬 Analyze Particles"
   - View detected particles with numbered labels
   - See individual particle measurements (area, diameter)
   - Review summary statistics
   - Results automatically saved to CSV

#### 5. **Batch Processing**
   - Process each image in sequence
   - Click "Next ►" to move to next image
   - Click "🔄 Refresh Summary" to see session statistics
   - Export all results using "💾 Export Results CSV"

## File Structure

```
cnt_analysis_app/
├── app.py                  # Main Gradio application
├── visualization.py        # Visualization helper functions
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Output Files

The application creates:

- **`analysis_results.csv`**: CSV file in the image folder containing:
  - File name
  - Particle count
  - Individual particle areas (pixels and nm²)
  - Equivalent diameters (pixels and nm)

## Supported Image Formats

- `.tif`, `.tiff`
- `.png`
- `.jpg`, `.jpeg`

## Tips for Best Results

1. **Scale Detection**:
   - Ensure scale bars are clearly visible in the bottom region
   - If auto-detection fails, use manual mode
   - Check that detected scale value matches the actual scale bar

2. **Mask Selection**:
   - Higher scores generally indicate better segmentation
   - Visually inspect all 3 masks before selecting
   - Choose the mask that best captures all particles

3. **Performance**:
   - Use GPU for faster processing (CUDA or Apple Silicon MPS)
   - ViT-H provides best quality but is slower
   - ViT-B is faster for large batches

## Troubleshooting

### SAM Model Not Loading
- Verify checkpoint file path is correct
- Ensure sufficient disk space (~2.5GB for ViT-H)
- Check that model type matches the checkpoint

### Scale Detection Fails
- Try adjusting the image crop percent (default 7%)
- Use manual mode if scale bar is non-standard
- Ensure scale bar text is clear and readable

### Out of Memory Errors
- Use smaller model (ViT-B instead of ViT-H)
- Process smaller images
- Close other applications

### No Particles Detected
- Check that scale was detected correctly
- Try selecting a different mask candidate
- Adjust minimum particle size parameters if needed

## Citation

This tool uses Meta's Segment Anything Model (SAM):

```bibtex
@article{kirillov2023segment,
  title={Segment Anything},
  author={Kirillov, Alexander and Mintun, Eric and Ravi, Nikhila and Mao, Hanzi and Rolland, Chloe and Gustafson, Laura and Xiao, Tete and Whitehead, Spencer and Berg, Alexander C. and Lo, Wan-Yen and Doll{\'a}r, Piotr and Girshick, Ross},
  journal={arXiv:2304.02643},
  year={2023}
}
```

## License

MIT License - See LICENSE file for details

## Support

For issues or questions:
- Open an issue on GitHub
- Check existing issues for solutions
- Consult the documentation

## Acknowledgments

Built with:
- [Gradio](https://gradio.app/) for the web interface
- [Segment Anything Model](https://segment-anything.com/) by Meta AI
- [EasyOCR](https://github.com/JaidedAI/EasyOCR) for scale detection
