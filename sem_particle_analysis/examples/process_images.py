"""
Example script for batch processing SEM images

This script demonstrates how to use the sem_particle_analysis package
to process multiple SEM images in a folder.
"""

import os
import argparse
from pathlib import Path

from sem_particle_analysis import (
    SAMModel,
    ScaleDetector,
    ParticleSegmenter,
    ParticleAnalyzer,
    ResultsManager
)
from sem_particle_analysis.utils import (
    find_images_in_folder,
    load_image,
    visualize_particles,
    visualize_comparison,
    plot_size_distribution,
    print_summary
)


def process_single_image(image_path, sam_model, scale_detector, output_dir=None,
                        save_visualizations=True):
    """
    Process a single SEM image.

    Args:
        image_path (str): Path to image file
        sam_model: SAMModel instance
        scale_detector: ScaleDetector instance
        output_dir (str, optional): Directory to save visualizations
        save_visualizations (bool): Whether to save visualization plots

    Returns:
        dict: Measurements dictionary or None if processing failed
    """
    filename = os.path.basename(image_path)
    print(f"\n{'='*60}")
    print(f"Processing: {filename}")
    print(f"{'='*60}")

    try:
        # Load image
        print("Loading image...")
        image = load_image(image_path)
        print(f"Image size: {image.shape[1]} x {image.shape[0]} pixels")

        # Detect scale bar
        print("\nDetecting scale bar...")
        try:
            scale_info = scale_detector.detect_scale_bar(
                image,
                region_width=0.5,
                region_height=0.06,
                threshold=250
            )
            print(f"Scale detected: {scale_info['pixel_length']} px = "
                  f"{scale_info['scale_nm']:.1f} nm")
            print(f"Conversion factor: {scale_info['conversion']:.3f} nm/pixel")

            # Crop scale bar
            cropped_image = scale_detector.crop_scale_bar(image, crop_percent=7.0)

        except Exception as e:
            print(f"Scale detection failed: {e}")
            print("Proceeding without scale calibration...")
            cropped_image = image
            scale_info = None

        # Segment particles
        print("\nSegmenting particles with SAM...")
        segmenter = ParticleSegmenter(sam_model)
        masks, scores = segmenter.segment_image(cropped_image)
        print(f"Generated {len(masks)} mask candidates")
        print(f"Scores: {scores}")

        # Select best mask
        selected_mask = segmenter.select_mask()  # Auto-select best

        # Analyze particles
        print("\nAnalyzing particles...")
        conversion = scale_info['conversion'] if scale_info else None
        analyzer = ParticleAnalyzer(conversion_factor=conversion)

        binary_mask = segmenter.get_binary_mask(invert=True)
        num_particles, regions = analyzer.analyze_mask(
            binary_mask,
            min_area=50,
            remove_border=True,
            border_buffer=4
        )

        print(f"Detected {num_particles} particles")

        # Get measurements
        measurements = analyzer.get_measurements(in_nm=bool(scale_info))
        print_summary(measurements, title=f"Results for {filename}")

        # Save visualizations
        if save_visualizations and output_dir:
            print("\nSaving visualizations...")
            base_name = Path(filename).stem

            # Particle visualization
            fig1 = visualize_particles(cropped_image, analyzer.labeled_mask, regions)
            fig1.savefig(os.path.join(output_dir, f"{base_name}_particles.png"),
                        dpi=150, bbox_inches='tight')
            print(f"Saved: {base_name}_particles.png")

            # Comparison view
            fig2 = visualize_comparison(cropped_image, analyzer.labeled_mask, regions)
            fig2.savefig(os.path.join(output_dir, f"{base_name}_comparison.png"),
                        dpi=150, bbox_inches='tight')
            print(f"Saved: {base_name}_comparison.png")

            # Size distribution
            if num_particles > 0:
                fig3 = plot_size_distribution(measurements)
                fig3.savefig(os.path.join(output_dir, f"{base_name}_distribution.png"),
                           dpi=150, bbox_inches='tight')
                print(f"Saved: {base_name}_distribution.png")

        return measurements

    except Exception as e:
        print(f"ERROR processing {filename}: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Main processing function"""
    parser = argparse.ArgumentParser(
        description="Batch process SEM images for particle analysis"
    )
    parser.add_argument(
        "image_folder",
        type=str,
        help="Path to folder containing SEM images"
    )
    parser.add_argument(
        "--sam-checkpoint",
        type=str,
        required=True,
        help="Path to SAM model checkpoint file"
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="vit_h",
        choices=["vit_h", "vit_l", "vit_b"],
        help="SAM model type (default: vit_h)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./output",
        help="Directory to save visualizations (default: ./output)"
    )
    parser.add_argument(
        "--results-file",
        type=str,
        default="analysis_results.csv",
        help="CSV file to save results (default: analysis_results.csv)"
    )
    parser.add_argument(
        "--no-visualizations",
        action="store_true",
        help="Skip saving visualization images"
    )

    args = parser.parse_args()

    # Create output directory
    if not args.no_visualizations:
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"Output directory: {args.output_dir}")

    # Initialize SAM model
    print("\nInitializing SAM model...")
    sam_model = SAMModel(
        checkpoint_path=args.sam_checkpoint,
        model_type=args.model_type
    )

    # Initialize scale detector
    print("Initializing scale detector...")
    scale_detector = ScaleDetector(use_gpu=False)

    # Initialize results manager
    results_manager = ResultsManager(csv_file=args.results_file)

    # Find images
    print(f"\nSearching for images in: {args.image_folder}")
    image_paths = find_images_in_folder(args.image_folder)
    print(f"Found {len(image_paths)} images to process")

    if len(image_paths) == 0:
        print("No images found. Exiting.")
        return

    # Process each image
    successful = 0
    failed = 0

    for idx, image_path in enumerate(image_paths, 1):
        print(f"\n[{idx}/{len(image_paths)}]")

        measurements = process_single_image(
            image_path,
            sam_model,
            scale_detector,
            output_dir=args.output_dir if not args.no_visualizations else None,
            save_visualizations=not args.no_visualizations
        )

        if measurements is not None:
            # Save to results manager
            results_manager.add_result(
                os.path.basename(image_path),
                measurements
            )
            successful += 1
        else:
            failed += 1

    # Print final summary
    print("\n" + "="*60)
    print("BATCH PROCESSING COMPLETE")
    print("="*60)
    print(f"Successfully processed: {successful}/{len(image_paths)}")
    print(f"Failed: {failed}/{len(image_paths)}")

    results_manager.print_summary()
    print(f"\nResults saved to: {args.results_file}")

    if not args.no_visualizations:
        print(f"Visualizations saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
