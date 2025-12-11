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


def create_particle_visualization(image, labeled_mask, regions, show_labels=True,
                                 pending_deletes=None, pending_add_masks=None, pending_merge=None):
    """
    Create visualization with particle contours and numbered labels.

    Args:
        image (np.ndarray): RGB image
        labeled_mask (np.ndarray): Labeled segmentation mask
        regions (list): RegionProperties objects
        show_labels (bool): Whether to show particle numbers
        pending_deletes (list): List of particle labels queued for deletion (yellow outline)
        pending_add_masks (list): List of masks queued for addition (green outline)
        pending_merge (list): List of particle labels selected for merging (blue outline)

    Returns:
        np.ndarray: Annotated image
    """
    # Create a copy to draw on
    vis_image = image.copy()

    pending_deletes = pending_deletes or []
    pending_add_masks = pending_add_masks or []
    pending_merge = pending_merge or []

    # Draw contours and labels for each particle
    for idx, region in enumerate(regions, start=1):
        # Get binary mask for this particle
        mask_i = (labeled_mask == region.label).astype(np.uint8)

        # Find contours
        contours, _ = cv2.findContours(mask_i, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Choose color based on status
        if region.label in pending_deletes:
            color = (255, 255, 0)  # Yellow for pending deletion
            thickness = 4
        elif region.label in pending_merge:
            color = (0, 0, 255)  # Blue for pending merge
            thickness = 4
        else:
            color = (255, 0, 0)  # Red for normal
            thickness = 2

        # Draw contours
        cv2.drawContours(vis_image, contours, -1, color, thickness)

        # Add numbered label at centroid
        if show_labels:
            y, x = int(region.centroid[0]), int(region.centroid[1])

            # Draw text background
            text = str(idx)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            text_thickness = 2

            (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, text_thickness)

            # Draw black rectangle background
            cv2.rectangle(vis_image,
                         (x - text_w//2 - 3, y - text_h//2 - 3),
                         (x + text_w//2 + 3, y + text_h//2 + 3),
                         (0, 0, 0), -1)

            # Draw white text
            cv2.putText(vis_image, text,
                       (x - text_w//2, y + text_h//2),
                       font, font_scale, (255, 255, 255), text_thickness)

    # Draw pending additions in green
    for add_mask in pending_add_masks:
        if add_mask is not None and add_mask.any():
            mask_uint8 = add_mask.astype(np.uint8)
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis_image, contours, -1, (0, 255, 0), 4)  # Green, thick

    return vis_image


def create_point_refine_visualization(image, refined_mask, point_coords, point_labels):
    """
    Create visualization for point refinement mode showing the refined mask and point markers.

    Args:
        image (np.ndarray): RGB image
        refined_mask (np.ndarray): Boolean mask of the refined particle
        point_coords (list): List of (x, y) click coordinates
        point_labels (list): List of point labels (1=positive, 0=negative)

    Returns:
        np.ndarray: Annotated image with mask and point markers
    """
    vis_image = image.copy()

    # Draw the refined mask contour in white
    if refined_mask is not None and refined_mask.any():
        mask_uint8 = refined_mask.astype(np.uint8)
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis_image, contours, -1, (255, 255, 255), 3)  # White thick outline

    # Draw point markers
    for (x, y), label in zip(point_coords, point_labels):
        x_int, y_int = int(x), int(y)
        if label == 1:
            # Positive point: Green + marker
            cv2.drawMarker(vis_image, (x_int, y_int), (0, 255, 0),
                          markerType=cv2.MARKER_CROSS, markerSize=20, thickness=3)
        else:
            # Negative point: Red × marker
            cv2.drawMarker(vis_image, (x_int, y_int), (255, 0, 0),
                          markerType=cv2.MARKER_TILTED_CROSS, markerSize=20, thickness=3)

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
    # Use buffer_rgba() instead of deprecated tostring_rgb()
    buf = fig.canvas.buffer_rgba()
    img_array = np.asarray(buf)
    # Convert RGBA to RGB
    img_array = img_array[:, :, :3]
    plt.close(fig)

    return img_array


def visualize_scale_detection(image, detection_result, crop_percent=7.0):
    """
    Visualize scale bar detection with detected line and crop box.

    Args:
        image (np.ndarray): Original RGB image
        detection_result (dict): Scale detection result containing 'region' and 'line_coords'
        crop_percent (float): Percentage to crop from bottom

    Returns:
        np.ndarray: Annotated image showing scale bar and crop box
    """
    vis_image = image.copy()
    H, W = image.shape[:2]

    # Get detection info
    x0, y0, box_w, box_h = detection_result['region']
    leftmost, rightmost, top_row = detection_result['line_coords']
    pixel_length = detection_result['pixel_length']
    scale_nm = detection_result['scale_nm']

    # Draw the scale bar region box in blue
    cv2.rectangle(vis_image, (x0, y0), (x0 + box_w, y0 + box_h), (0, 0, 255), 2)

    # Draw the detected scale bar line in red (thick)
    line_y = y0 + top_row
    cv2.line(vis_image, (x0 + leftmost, line_y), (x0 + rightmost, line_y), (255, 0, 0), 3)

    # Add scale info text near the line
    scale_text = f"{scale_nm:.0f} nm = {pixel_length} px"
    cv2.putText(vis_image, scale_text, (x0 + leftmost, line_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(vis_image, scale_text, (x0 + leftmost, line_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1)

    # Draw the crop box in green
    crop_height = int(H * (1 - crop_percent / 100))
    cv2.rectangle(vis_image, (0, 0), (W, crop_height), (0, 255, 0), 3)

    # Add crop info text
    crop_text = f"Analysis Region (excludes bottom {crop_percent}%)"
    cv2.putText(vis_image, crop_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3)
    cv2.putText(vis_image, crop_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    return vis_image


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


def create_histogram_plots(measurements):
    """
    Create histograms for particle size distribution.

    Args:
        measurements (dict): Measurements from ParticleAnalyzer

    Returns:
        np.ndarray: Combined histogram image
    """
    if measurements['num_particles'] == 0:
        # Return empty plot
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.text(0.5, 0.5, 'No particles to plot', ha='center', va='center', fontsize=14)
        ax.axis('off')
        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        img_array = np.asarray(buf)[:, :, :3]
        plt.close(fig)
        return img_array

    areas = np.array(measurements['areas'])
    diameters = np.array(measurements['diameters'])
    unit = measurements['unit']

    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Area histogram
    ax1.hist(areas, bins=min(30, len(areas)), color='steelblue', edgecolor='black', alpha=0.7)
    ax1.set_xlabel(f'Particle Area ({unit}²)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax1.set_title(f'Particle Area Distribution\n(n={len(areas)})', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)

    # Add statistics text
    stats_text = f'Mean: {np.mean(areas):.1f}\nMedian: {np.median(areas):.1f}\nStd: {np.std(areas):.1f}'
    ax1.text(0.98, 0.98, stats_text, transform=ax1.transAxes,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
             fontsize=10)

    # Diameter histogram
    ax2.hist(diameters, bins=min(30, len(diameters)), color='coral', edgecolor='black', alpha=0.7)
    ax2.set_xlabel(f'Equivalent Diameter ({unit})', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax2.set_title(f'Particle Diameter Distribution\n(n={len(diameters)})', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)

    # Add statistics text
    stats_text = f'Mean: {np.mean(diameters):.1f}\nMedian: {np.median(diameters):.1f}\nStd: {np.std(diameters):.1f}'
    ax2.text(0.98, 0.98, stats_text, transform=ax2.transAxes,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
             fontsize=10)

    plt.tight_layout()

    # Convert to image
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    img_array = np.asarray(buf)[:, :, :3]
    plt.close(fig)

    return img_array
