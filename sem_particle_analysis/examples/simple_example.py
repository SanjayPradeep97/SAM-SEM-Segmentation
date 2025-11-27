"""
Simple example of using the sem_particle_analysis package

This demonstrates the basic workflow for analyzing a single SEM image.
"""

from sem_particle_analysis import (
    SAMModel,
    ScaleDetector,
    ParticleSegmenter,
    ParticleAnalyzer,
)
from sem_particle_analysis.utils import (
    load_image,
    visualize_particles,
    print_summary
)


def main():
    # Configuration
    IMAGE_PATH = "path/to/your/sem_image.tif"
    SAM_CHECKPOINT = "path/to/sam_vit_h_4b8939.pth"

    print("="*60)
    print("SEM Particle Analysis - Simple Example")
    print("="*60)

    # Step 1: Initialize SAM model
    print("\n1. Loading SAM model...")
    sam_model = SAMModel(
        checkpoint_path=SAM_CHECKPOINT,
        model_type="vit_h"  # Options: "vit_h", "vit_l", "vit_b"
    )

    # Step 2: Load image
    print("\n2. Loading image...")
    image = load_image(IMAGE_PATH)
    print(f"   Image size: {image.shape[1]} x {image.shape[0]} pixels")

    # Step 3: Detect scale bar
    print("\n3. Detecting scale bar...")
    scale_detector = ScaleDetector()

    try:
        scale_info = scale_detector.detect_scale_bar(image)
        print(f"   Scale: {scale_info['pixel_length']} px = "
              f"{scale_info['scale_nm']:.1f} nm")
        print(f"   Conversion: {scale_info['conversion']:.3f} nm/pixel")

        # Crop the scale bar from the image
        cropped_image = scale_detector.crop_scale_bar(image, crop_percent=7.0)
        conversion_factor = scale_info['conversion']

    except Exception as e:
        print(f"   Scale detection failed: {e}")
        print("   Proceeding without scale calibration...")
        cropped_image = image
        conversion_factor = None

    # Step 4: Segment particles
    print("\n4. Segmenting particles with SAM...")
    segmenter = ParticleSegmenter(sam_model)

    masks, scores = segmenter.segment_image(cropped_image)
    print(f"   Generated {len(masks)} mask candidates")
    print(f"   Confidence scores: {scores}")

    # Automatically select the best mask (highest score)
    selected_mask = segmenter.select_mask()

    # Step 5: Analyze particles
    print("\n5. Analyzing particles...")
    analyzer = ParticleAnalyzer(conversion_factor=conversion_factor)

    # Get binary mask (particles = 1, background = 0)
    binary_mask = segmenter.get_binary_mask(invert=True)

    # Analyze the mask to identify individual particles
    num_particles, regions = analyzer.analyze_mask(
        binary_mask,
        min_area=50,          # Minimum particle area in pixels
        remove_border=True,   # Remove particles touching edges
        border_buffer=4       # Border buffer in pixels
    )

    print(f"   Detected {num_particles} particles")

    # Step 6: Get measurements
    print("\n6. Computing measurements...")
    measurements = analyzer.get_measurements(in_nm=bool(conversion_factor))

    # Print summary statistics
    print_summary(measurements)

    # Step 7: Visualize results
    print("\n7. Creating visualization...")
    fig = visualize_particles(
        cropped_image,
        analyzer.labeled_mask,
        regions,
        show_labels=True
    )

    # Save the figure
    output_path = "particle_analysis_result.png"
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"   Saved visualization to: {output_path}")

    # Step 8: Access individual measurements
    print("\n8. Individual particle data:")
    print(f"   Areas: {measurements['areas'][:5]}...")  # First 5
    print(f"   Diameters: {measurements['diameters'][:5]}...")
    print(f"   Unit: {measurements['unit']}")

    print("\n" + "="*60)
    print("Analysis complete!")
    print("="*60)


if __name__ == "__main__":
    main()
