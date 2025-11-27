# SEM Particle Analysis

A Python package for automated segmentation and analysis of particles in Scanning Electron Microscopy (SEM) and Transmission Electron Microscopy (TEM) images using Meta's Segment Anything Model (SAM).

## Features

- **Automated Scale Bar Detection**: Uses OCR to detect and calibrate scale bars in microscopy images
- **AI-Powered Segmentation**: Leverages SAM for accurate particle segmentation
- **Particle Analysis**: Computes area, diameter, and other morphological measurements
- **Batch Processing**: Process multiple images with consistent parameters
- **Results Management**: Export analysis results to CSV for further processing
- **Flexible API**: Easy-to-use Python API for custom workflows

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (optional, but recommended for faster processing)

### Basic Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/sem-particle-analysis.git
cd sem-particle-analysis

# Install the package
pip install -e .
```

### Download SAM Model Checkpoint

Download one of the SAM model checkpoints:

```bash
# ViT-H (best quality, ~2.5GB)
curl -O https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

# ViT-L (good balance, ~1.2GB)
curl -O https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth

# ViT-B (fastest, ~375MB)
curl -O https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```

## Quick Start

### Basic Usage

```python
from sem_particle_analysis import SAMModel, ScaleDetector, ParticleSegmenter, ParticleAnalyzer
from sem_particle_analysis.utils import load_image, visualize_particles, print_summary

# 1. Initialize SAM model
sam_model = SAMModel(
    checkpoint_path="path/to/sam_vit_h_4b8939.pth",
    model_type="vit_h"
)

# 2. Load and process image
image = load_image("path/to/sem_image.tif")

# 3. Detect scale bar and get conversion factor
scale_detector = ScaleDetector()
scale_info = scale_detector.detect_scale_bar(image)
print(f"Conversion: {scale_info['conversion']:.3f} nm/pixel")

# 4. Crop scale bar from image
cropped_image = scale_detector.crop_scale_bar(image, crop_percent=7.0)

# 5. Segment particles
segmenter = ParticleSegmenter(sam_model)
masks, scores = segmenter.segment_image(cropped_image)
print(f"Generated {len(masks)} mask candidates with scores: {scores}")

# 6. Select best mask
selected_mask = segmenter.select_mask()  # Auto-selects highest scoring mask

# 7. Analyze particles
analyzer = ParticleAnalyzer(conversion_factor=scale_info['conversion'])
binary_mask = segmenter.get_binary_mask(invert=True)
num_particles, regions = analyzer.analyze_mask(binary_mask)

# 8. Get measurements
measurements = analyzer.get_measurements(in_nm=True)
print_summary(measurements)

# 9. Visualize results
fig = visualize_particles(cropped_image, analyzer.labeled_mask, regions)
fig.savefig("results.png")
```

### Batch Processing Multiple Images

```python
from sem_particle_analysis import SAMModel, ScaleDetector, ParticleSegmenter, ParticleAnalyzer, ResultsManager
from sem_particle_analysis.utils import find_images_in_folder, load_image
import os

# Initialize components
sam_model = SAMModel("path/to/sam_checkpoint.pth", model_type="vit_h")
scale_detector = ScaleDetector()
segmenter = ParticleSegmenter(sam_model)
results_manager = ResultsManager(csv_file="batch_results.csv")

# Find all images
image_paths = find_images_in_folder("path/to/image/folder")
print(f"Found {len(image_paths)} images to process")

# Process each image
for image_path in image_paths:
    print(f"\nProcessing: {os.path.basename(image_path)}")

    # Load image
    image = load_image(image_path)

    # Detect scale bar
    try:
        scale_info = scale_detector.detect_scale_bar(image)
        cropped_image = scale_detector.crop_scale_bar(image)
    except Exception as e:
        print(f"Scale detection failed: {e}")
        continue

    # Segment and analyze
    masks, scores = segmenter.segment_image(cropped_image)
    selected_mask = segmenter.select_mask()

    analyzer = ParticleAnalyzer(conversion_factor=scale_info['conversion'])
    binary_mask = segmenter.get_binary_mask(invert=True)
    num_particles, regions = analyzer.analyze_mask(binary_mask)

    # Save results
    measurements = analyzer.get_measurements(in_nm=True)
    results_manager.add_result(os.path.basename(image_path), measurements)

    print(f"Detected {num_particles} particles")

# Print summary
results_manager.print_summary()
print(f"Results saved to: {results_manager.csv_file}")
```

### Advanced: Manual Refinement

```python
# Start with automated segmentation
segmenter = ParticleSegmenter(sam_model)
masks, scores = segmenter.segment_image(image)
selected_mask = segmenter.select_mask()

analyzer = ParticleAnalyzer(conversion_factor=conversion)
binary_mask = segmenter.get_binary_mask(invert=True)
num_particles, regions = analyzer.analyze_mask(binary_mask)

# Remove edge particles
analyzer.clear_edge_particles(buffer_size=10)

# Delete specific particles by label
analyzer.delete_particles(labels_to_delete=[3, 5, 7])

# Merge particles
analyzer.merge_particles(labels_to_merge=[1, 2])

# Add a particle using SAM refinement
point_coords = [[150, 200]]  # [x, y]
point_labels = [1]  # 1 = positive point
refined_mask, score = segmenter.refine_with_sam(
    image,
    point_coords,
    point_labels,
    base_mask=analyzer.mask
)
analyzer.add_particle_from_sam(refined_mask)

# Get final measurements
final_measurements = analyzer.get_measurements(in_nm=True)
```

## API Reference

### SAMModel
Manages the Segment Anything Model initialization and predictions.

- `__init__(checkpoint_path, model_type, device)`: Initialize SAM model
- `set_image(image)`: Set image for encoding (run once per image)
- `predict(point_coords, point_labels, box, multimask_output)`: Generate segmentation masks

### ScaleDetector
Detects and calibrates scale bars in microscopy images.

- `detect_scale_bar(image, region_width, region_height, vertical_offset, threshold)`: Detect scale bar
- `crop_scale_bar(image, crop_percent)`: Remove scale bar from image
- `get_conversion_factor()`: Get nm/pixel conversion
- `set_manual_scale(scale_nm, pixel_length)`: Manually set scale

### ParticleSegmenter
Segments particles using SAM.

- `segment_image(image, multimask_output)`: Generate mask candidates
- `select_mask(mask_index)`: Select a specific mask
- `get_binary_mask(invert)`: Get binary mask
- `refine_with_sam(image, point_coords, point_labels, base_mask)`: Refine with point prompts

### ParticleAnalyzer
Analyzes segmented particles and computes measurements.

- `analyze_mask(mask, min_area, min_size, remove_border, border_buffer)`: Analyze binary mask
- `get_measurements(in_nm)`: Get particle measurements
- `clear_edge_particles(buffer_size)`: Remove edge-touching particles
- `delete_particles(labels_to_delete)`: Delete specific particles
- `merge_particles(labels_to_merge)`: Merge multiple particles
- `get_summary_statistics()`: Get statistical summary

### ResultsManager
Manages saving and loading of analysis results.

- `add_result(file_name, measurements)`: Add analysis result
- `get_results()`: Get all results as DataFrame
- `delete_result(index)`: Delete a result
- `export_results(output_file)`: Export to CSV
- `get_summary()`: Get summary statistics
- `print_summary()`: Print formatted summary

## Project Structure

```
sem_particle_analysis/
├── sem_particle_analysis/
│   ├── __init__.py           # Package initialization
│   ├── model.py              # SAM model management
│   ├── scale_detection.py    # Scale bar detection
│   ├── segmentation.py       # Particle segmentation
│   ├── analysis.py           # Particle analysis
│   ├── data_manager.py       # Results management
│   └── utils.py              # Utility functions
├── examples/
│   └── process_images.py     # Example usage script
├── setup.py                  # Package setup
├── pyproject.toml            # Build configuration
├── requirements.txt          # Dependencies
└── README.md                 # This file
```

## Citation

This package uses Meta's Segment Anything Model (SAM):

```
@article{kirillov2023segment,
  title={Segment Anything},
  author={Kirillov, Alexander and Mintun, Eric and Ravi, Nikhila and Mao, Hanzi and Rolland, Chloe and Gustafson, Laura and Xiao, Tete and Whitehead, Spencer and Berg, Alexander C. and Lo, Wan-Yen and Doll{\'a}r, Piotr and Girshick, Ross},
  journal={arXiv:2304.02643},
  year={2023}
}
```

## License

MIT License - see LICENSE file for details

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Support

For issues, questions, or contributions, please open an issue on GitHub.
