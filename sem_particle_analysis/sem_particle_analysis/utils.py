"""
Utility Functions

Common helper functions for image processing and visualization.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure


def load_image(file_path):
    """
    Load an image from file and convert to RGB.

    Args:
        file_path (str): Path to image file

    Returns:
        np.ndarray: RGB image array

    Raises:
        FileNotFoundError: If image file doesn't exist
        ValueError: If image cannot be loaded
    """
    bgr = cv2.imread(file_path)
    if bgr is None:
        raise ValueError(f"Could not load image: {file_path}")

    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def save_image(image, file_path):
    """
    Save an RGB image to file.

    Args:
        image (np.ndarray): RGB image array
        file_path (str): Output file path

    Returns:
        bool: True if successful
    """
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return cv2.imwrite(file_path, bgr)


def visualize_masks(image, masks, scores, figsize=(15, 5)):
    """
    Visualize multiple mask candidates with overlays.

    Args:
        image (np.ndarray): RGB image
        masks (np.ndarray): Array of masks (num_masks, H, W)
        scores (np.ndarray): Confidence scores
        figsize (tuple): Figure size

    Returns:
        matplotlib.figure.Figure: Figure object
    """
    num_masks = len(masks)
    fig, axes = plt.subplots(1, num_masks, figsize=figsize)

    if num_masks == 1:
        axes = [axes]

    for idx, (mask, score) in enumerate(zip(masks, scores)):
        ax = axes[idx]

        # Show original image
        ax.imshow(image)

        # Create red overlay for detected particles
        overlay = np.zeros((*mask.shape, 4))  # RGBA
        overlay[mask] = [1, 0, 0, 0.6]  # Semi-transparent red
        ax.imshow(overlay)

        # Add title with score
        ax.set_title(f"Mask {idx} (score: {score:.3f})")
        ax.axis('off')

    plt.tight_layout()
    return fig


def visualize_particles(image, labeled_mask, regions=None, show_labels=True,
                       contour_color='red', figsize=(12, 8)):
    """
    Visualize detected particles with contours and labels.

    Args:
        image (np.ndarray): RGB image
        labeled_mask (np.ndarray): Labeled segmentation mask
        regions (list, optional): RegionProperties objects
        show_labels (bool): Whether to show particle numbers
        contour_color (str): Color for contours
        figsize (tuple): Figure size

    Returns:
        matplotlib.figure.Figure: Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(image)

    # Draw contours
    if regions is None:
        regions = measure.regionprops(labeled_mask)

    for idx, region in enumerate(regions, start=1):
        # Get contours for this region
        mask_i = (labeled_mask == region.label)
        contours = measure.find_contours(mask_i.astype(float), 0.5)

        for contour in contours:
            ax.plot(contour[:, 1], contour[:, 0], color=contour_color, linewidth=1.5)

        # Add label at centroid
        if show_labels:
            y, x = region.centroid
            ax.text(x, y, str(idx), color='white', fontsize=11,
                   ha='center', va='center',
                   bbox=dict(facecolor='black', alpha=0.6, edgecolor='none'))

    ax.set_title(f"Detected Particles ({len(regions)})")
    ax.axis('off')
    plt.tight_layout()

    return fig


def visualize_comparison(image, labeled_mask, regions, figsize=(16, 8)):
    """
    Create a side-by-side comparison of original image and binary mask.

    Args:
        image (np.ndarray): RGB image
        labeled_mask (np.ndarray): Labeled segmentation mask
        regions (list): RegionProperties objects
        figsize (tuple): Figure size

    Returns:
        matplotlib.figure.Figure: Figure object
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Left: Original with contours
    axes[0].imshow(image)
    for idx, region in enumerate(regions, start=1):
        mask_i = (labeled_mask == region.label)
        contours = measure.find_contours(mask_i.astype(float), 0.5)
        for contour in contours:
            axes[0].plot(contour[:, 1], contour[:, 0], color='red', linewidth=1.2)
        y, x = region.centroid
        axes[0].text(x, y, str(idx), color='white', fontsize=11,
                    ha='center', va='center',
                    bbox=dict(facecolor='black', alpha=0.6, edgecolor='none'))
    axes[0].set_title(f"Original with Contours ({len(regions)} particles)")
    axes[0].axis('off')

    # Right: Binary mask
    axes[1].imshow(labeled_mask > 0, cmap='gray')
    axes[1].set_title("Binary Mask")
    axes[1].axis('off')

    plt.tight_layout()
    return fig


def plot_size_distribution(measurements, bins=20, figsize=(12, 5)):
    """
    Plot histograms of particle size distributions.

    Args:
        measurements (dict): Measurements dictionary from ParticleAnalyzer
        bins (int): Number of histogram bins
        figsize (tuple): Figure size

    Returns:
        matplotlib.figure.Figure: Figure object
    """
    if measurements['num_particles'] == 0:
        print("No particles to plot")
        return None

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    unit = measurements['unit']

    # Area distribution
    axes[0].hist(measurements['areas'], bins=bins, edgecolor='black', alpha=0.7)
    axes[0].set_xlabel(f"Area ({unit}²)" if unit == 'nm' else f"Area ({unit})")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Particle Area Distribution")
    axes[0].grid(alpha=0.3)

    # Diameter distribution
    axes[1].hist(measurements['diameters'], bins=bins, edgecolor='black', alpha=0.7)
    axes[1].set_xlabel(f"Equivalent Diameter ({unit})")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Particle Diameter Distribution")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    return fig


def create_mask_overlay(image, mask, color=(255, 0, 0), alpha=0.4):
    """
    Create an image with a colored mask overlay.

    Args:
        image (np.ndarray): RGB image
        mask (np.ndarray): Boolean mask
        color (tuple): RGB color for overlay (0-255)
        alpha (float): Transparency (0-1)

    Returns:
        np.ndarray: Image with overlay
    """
    overlay = image.copy()
    overlay[mask] = (
        overlay[mask] * (1 - alpha) +
        np.array(color) * alpha
    ).astype(np.uint8)

    return overlay


def find_images_in_folder(folder_path, extensions=None):
    """
    Find all image files in a folder.

    Args:
        folder_path (str): Path to folder
        extensions (list, optional): List of extensions to search for.
            Default: ['png', 'jpg', 'jpeg', 'tif', 'tiff']

    Returns:
        list: Sorted list of image file paths
    """
    import glob
    import os

    if extensions is None:
        extensions = ['png', 'jpg', 'jpeg', 'tif', 'tiff']

    image_paths = []
    for ext in extensions:
        pattern = os.path.join(folder_path, f'*.{ext}')
        image_paths.extend(glob.glob(pattern))

    return sorted(image_paths)


def compute_image_hash(image):
    """
    Compute a simple hash of an image for duplicate detection.

    Args:
        image (np.ndarray): Image array

    Returns:
        str: MD5 hash string
    """
    import hashlib
    return hashlib.md5(image.tobytes()).hexdigest()


def print_summary(measurements, title="Analysis Summary"):
    """
    Print a formatted summary of particle measurements.

    Args:
        measurements (dict): Measurements dictionary
        title (str): Summary title
    """
    print("\n" + "="*60)
    print(title)
    print("="*60)

    print(f"Number of particles: {measurements['num_particles']}")

    if measurements['num_particles'] == 0:
        print("No particles detected")
        return

    unit = measurements['unit']

    print(f"\nArea Statistics ({unit}²):" if unit == 'nm' else f"\nArea Statistics ({unit}):")
    areas = np.array(measurements['areas'])
    print(f"  Mean:   {np.mean(areas):.2f}")
    print(f"  Median: {np.median(areas):.2f}")
    print(f"  Std:    {np.std(areas):.2f}")
    print(f"  Min:    {np.min(areas):.2f}")
    print(f"  Max:    {np.max(areas):.2f}")

    print(f"\nDiameter Statistics ({unit}):")
    diameters = np.array(measurements['diameters'])
    print(f"  Mean:   {np.mean(diameters):.2f}")
    print(f"  Median: {np.median(diameters):.2f}")
    print(f"  Std:    {np.std(diameters):.2f}")
    print(f"  Min:    {np.min(diameters):.2f}")
    print(f"  Max:    {np.max(diameters):.2f}")

    print("="*60 + "\n")
