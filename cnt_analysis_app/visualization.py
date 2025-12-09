"""
Visualization Functions for Gradio App

Provides image overlay and plotting functions optimized for Gradio display.
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage import measure
import cv2


def create_mask_overlay(image, mask, color=(255, 0, 0), alpha=0.5):
    """
    Create an image with colored mask overlay for visualization.

    Args:
        image (np.ndarray): RGB image
        mask (np.ndarray): Boolean mask
        color (tuple): RGB color (0-255)
        alpha (float): Transparency (0-1)

    Returns:
        np.ndarray: RGB image with overlay
    """
    overlay = image.copy().astype(float)

    if mask is not None and mask.any():
        color_array = np.array(color, dtype=float)
        overlay[mask] = overlay[mask] * (1 - alpha) + color_array * alpha

    return overlay.astype(np.uint8)


def create_particle_visualization(image, labeled_mask, regions, show_labels=True):
    """
    Create visualization with particle contours and numbered labels.

    Args:
        image (np.ndarray): RGB image
        labeled_mask (np.ndarray): Labeled segmentation mask
        regions (list): RegionProperties objects
        show_labels (bool): Whether to show particle numbers

    Returns:
        np.ndarray: Annotated image
    """
    # Create a copy to draw on
    vis_image = image.copy()

    # Draw contours and labels for each particle
    for idx, region in enumerate(regions, start=1):
        # Get binary mask for this particle
        mask_i = (labeled_mask == region.label).astype(np.uint8)

        # Find contours
        contours, _ = cv2.findContours(mask_i, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw red contours
        cv2.drawContours(vis_image, contours, -1, (255, 0, 0), 2)

        # Add numbered label at centroid
        if show_labels:
            y, x = int(region.centroid[0]), int(region.centroid[1])

            # Draw text background
            text = str(idx)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2

            (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)

            # Draw black rectangle background
            cv2.rectangle(vis_image,
                         (x - text_w//2 - 3, y - text_h//2 - 3),
                         (x + text_w//2 + 3, y + text_h//2 + 3),
                         (0, 0, 0), -1)

            # Draw white text
            cv2.putText(vis_image, text,
                       (x - text_w//2, y + text_h//2),
                       font, font_scale, (255, 255, 255), thickness)

    return vis_image


def visualize_three_masks(image, masks, scores):
    """
    Create a 3-panel visualization showing all mask candidates.

    Args:
        image (np.ndarray): RGB image
        masks (np.ndarray): Array of 3 masks (3, H, W)
        scores (np.ndarray): Confidence scores

    Returns:
        np.ndarray: Combined 3-panel image
    """
    num_masks = len(masks)

    # Create subplots
    fig, axes = plt.subplots(1, num_masks, figsize=(15, 5))

    if num_masks == 1:
        axes = [axes]

    for idx, (mask, score) in enumerate(zip(masks, scores)):
        ax = axes[idx]

        # Create overlay
        overlay = create_mask_overlay(image, mask, color=(255, 0, 0), alpha=0.5)

        ax.imshow(overlay)
        ax.set_title(f"Mask {idx + 1}\nScore: {score:.3f}", fontsize=14, fontweight='bold')
        ax.axis('off')

    plt.tight_layout()

    # Convert plot to image
    fig.canvas.draw()
    img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)

    return img_array


def create_results_dataframe(measurements):
    """
    Convert measurements dict to a formatted DataFrame for display.

    Args:
        measurements (dict): Measurements from ParticleAnalyzer

    Returns:
        pd.DataFrame: Formatted results table
    """
    import pandas as pd

    if measurements['num_particles'] == 0:
        return pd.DataFrame(columns=['Particle ID', 'Area (nm²)', 'Diameter (nm)'])

    # Create DataFrame
    data = {
        'Particle ID': list(range(1, measurements['num_particles'] + 1)),
        'Area (nm²)': [f"{a:.1f}" for a in measurements['areas']],
        'Diameter (nm)': [f"{d:.1f}" for d in measurements['diameters']],
    }

    df = pd.DataFrame(data)
    return df


def create_summary_statistics_table(measurements):
    """
    Create a summary statistics table.

    Args:
        measurements (dict): Measurements from ParticleAnalyzer

    Returns:
        pd.DataFrame: Summary statistics
    """
    import pandas as pd

    if measurements['num_particles'] == 0:
        return pd.DataFrame()

    areas = np.array(measurements['areas'])
    diameters = np.array(measurements['diameters'])

    stats_data = {
        'Metric': ['Mean', 'Median', 'Std Dev', 'Min', 'Max'],
        'Area (nm²)': [
            f"{np.mean(areas):.1f}",
            f"{np.median(areas):.1f}",
            f"{np.std(areas):.1f}",
            f"{np.min(areas):.1f}",
            f"{np.max(areas):.1f}"
        ],
        'Diameter (nm)': [
            f"{np.mean(diameters):.1f}",
            f"{np.median(diameters):.1f}",
            f"{np.std(diameters):.1f}",
            f"{np.min(diameters):.1f}",
            f"{np.max(diameters):.1f}"
        ]
    }

    return pd.DataFrame(stats_data)
