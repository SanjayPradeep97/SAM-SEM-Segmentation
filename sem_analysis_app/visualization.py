"""
Visualization Functions for Gradio App

Provides image overlay and plotting functions optimized for Gradio display.
"""

import numpy as np
import matplotlib.pyplot as plt
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


def visualize_three_masks(image, masks, labels):
    """
    Create a panel visualization showing the mask candidates side by side.

    Args:
        image (np.ndarray): RGB image
        masks: Sequence of boolean masks (H, W)
        labels: One caption per mask — either a ready-made string describing the
            candidate, or a bare confidence score.

    Returns:
        np.ndarray: Combined panel image
    """
    num_masks = len(masks)

    # Create subplots
    fig, axes = plt.subplots(1, num_masks, figsize=(15, 5))

    if num_masks == 1:
        axes = [axes]

    for idx, (mask, label) in enumerate(zip(masks, labels)):
        ax = axes[idx]

        # Create overlay
        overlay = create_mask_overlay(image, mask, color=(255, 0, 0), alpha=0.5)

        # Labels may be pre-formatted strings or bare confidence scores.
        if isinstance(label, str):
            title = label if label.lower().startswith("option") else f"Mask {idx + 1}\n{label}"
        else:
            title = f"Mask {idx + 1}\nScore: {label:.3f}"

        ax.imshow(overlay)
        ax.set_title(title, fontsize=13, fontweight='bold')
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


def _round_to_nice_number(value):
    """
    Round a value to the nearest 'nice' number for scale bar labels.

    Args:
        value (float): Value in nanometers

    Returns:
        float: Nearest nice number
    """
    nice_values = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000,
                   10000, 20000, 50000, 100000, 200000, 500000, 1000000,
                   2000000, 5000000, 10000000]
    return min(nice_values, key=lambda x: abs(x - value))


def _format_scale_label(nm_value):
    """
    Format a nanometer value with appropriate unit for display.

    Args:
        nm_value (float): Value in nanometers

    Returns:
        str: Formatted string like "500 nm", "1 um", "2 mm"
    """
    if nm_value >= 1_000_000:
        return f"{nm_value / 1_000_000:.0f} mm"
    elif nm_value >= 1000:
        val = nm_value / 1000
        return f"{val:.0f} um" if val == int(val) else f"{val:.1f} um"
    else:
        return f"{nm_value:.0f} nm"


def visualize_scale_verification(image, scale_info, databar_info=None):
    """
    Create a visual verification overlay showing the detected scale.

    Draws a reference measurement bar and info panel on the original (pre-crop)
    image so the user can visually confirm the scale is correct. Works for
    both metadata-based and OCR-based detection.

    Args:
        image (np.ndarray): Original RGB image (before cropping)
        scale_info (dict): Scale detection result containing 'conversion', 'method', etc.
        databar_info (dict, optional): Databar detection result from detect_databar()

    Returns:
        np.ndarray: Annotated image with reference bar, info panel, and crop line
    """
    vis_image = image.copy()
    H, W = image.shape[:2]
    conversion = scale_info['conversion']  # nm/pixel
    method = scale_info.get('method', 'unknown')

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.5, min(0.9, W / 1500))

    # --- Info panel (top-right corner) ---
    info_lines = []

    # Scale value
    if conversion >= 1:
        info_lines.append(f"Scale: {conversion:.2f} nm/pixel")
    else:
        info_lines.append(f"Scale: {conversion:.4f} nm/pixel")

    # Method and source
    if method == 'metadata':
        source = scale_info.get('metadata_source', 'TIFF metadata')
        manufacturer = scale_info.get('manufacturer', 'unknown')
        confidence = scale_info.get('confidence', 'unknown')
        if manufacturer != 'unknown':
            info_lines.append(f"Source: {manufacturer.upper()} metadata")
        else:
            info_lines.append(f"Source: {source}")
        info_lines.append(f"Confidence: {confidence.upper()}")
    elif method == 'ocr':
        info_lines.append("Source: OCR")
        ocr_text = scale_info.get('ocr_text', '')
        if ocr_text:
            info_lines.append(f"Read: \"{ocr_text}\"")

    # Cross-check info
    cross_check = scale_info.get('cross_check')
    if cross_check:
        meta_conv = cross_check.get('metadata_conversion', 0)
        ocr_conv = cross_check.get('ocr_conversion', 0)
        agrees = cross_check.get('agrees', False)

        if meta_conv > 0 and ocr_conv > 0:
            check_mark = "AGREE" if agrees else "DISAGREE"
            info_lines.append(f"Cross-check: {check_mark}")
            if method == 'metadata':
                info_lines.append(f"  OCR: {ocr_conv:.2f} nm/px")
            else:
                info_lines.append(f"  Meta: {meta_conv:.2f} nm/px")

    # Draw info panel background
    line_height = int(22 * font_scale / 0.5)
    panel_w = 320
    panel_h = len(info_lines) * line_height + 16
    panel_x = W - panel_w - 10
    panel_y = 10

    # Semi-transparent dark background
    overlay_region = vis_image[panel_y:panel_y + panel_h, panel_x:panel_x + panel_w].copy()
    cv2.rectangle(vis_image, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h),
                  (0, 0, 0), -1)
    vis_image[panel_y:panel_y + panel_h, panel_x:panel_x + panel_w] = \
        (overlay_region * 0.3 + vis_image[panel_y:panel_y + panel_h, panel_x:panel_x + panel_w] * 0.7).astype(np.uint8)

    # Draw info text
    info_font_scale = max(0.4, min(0.55, W / 2000))
    for i, line in enumerate(info_lines):
        y = panel_y + 20 + i * line_height

        # Color code cross-check results
        if 'AGREE' in line and 'DIS' not in line:
            text_color = (0, 255, 0)  # Green
        elif 'DISAGREE' in line:
            text_color = (0, 100, 255)  # Orange
        elif 'Confidence: HIGH' in line:
            text_color = (0, 255, 0)  # Green
        elif 'Confidence: LOW' in line:
            text_color = (0, 100, 255)  # Orange
        else:
            text_color = (255, 255, 255)

        cv2.putText(vis_image, line, (panel_x + 8, y),
                    font, info_font_scale, text_color, 1, cv2.LINE_AA)

    # --- Crop line (if databar detected) ---
    crop_y = None
    if databar_info and databar_info.get('has_databar'):
        crop_y = H - databar_info['databar_height']
        cv2.line(vis_image, (0, crop_y), (W, crop_y), (0, 255, 0), 2)
        cv2.putText(vis_image, "Crop line (databar)", (10, crop_y - 8),
                    font, info_font_scale, (0, 0, 0), 3)
        cv2.putText(vis_image, "Crop line (databar)", (10, crop_y - 8),
                    font, info_font_scale, (0, 255, 0), 1)

    # --- Verification scale bar (bottom-right, above crop line) ---
    # Draw a reference bar whose pixel length corresponds to a nice round physical value
    # so the user can visually verify the scale is correct.
    if conversion > 0:
        # Choose a bar that's roughly 15-25% of image width
        target_px = W * 0.20
        target_nm = target_px * conversion
        nice_nm = _round_to_nice_number(target_nm)
        bar_px = max(10, int(nice_nm / conversion))

        # Position: bottom-right, above the crop line (or above bottom edge)
        bar_margin = 15
        bar_y = (crop_y - 30) if crop_y else (H - 30)
        bar_x_right = W - bar_margin
        bar_x_left = bar_x_right - bar_px

        # Clamp to image bounds
        bar_x_left = max(bar_margin, bar_x_left)
        bar_y = max(40, bar_y)

        # Draw bar with outline for contrast
        bar_thickness = 4
        cv2.line(vis_image, (bar_x_left, bar_y), (bar_x_right, bar_y),
                 (0, 0, 0), bar_thickness + 2)  # Black outline
        cv2.line(vis_image, (bar_x_left, bar_y), (bar_x_right, bar_y),
                 (255, 255, 255), bar_thickness)  # White bar

        # End ticks
        tick_h = 8
        cv2.line(vis_image, (bar_x_left, bar_y - tick_h), (bar_x_left, bar_y + tick_h),
                 (255, 255, 255), 2)
        cv2.line(vis_image, (bar_x_right, bar_y - tick_h), (bar_x_right, bar_y + tick_h),
                 (255, 255, 255), 2)

        # Label above the bar
        bar_label = _format_scale_label(nice_nm)
        label_font_scale = max(0.5, min(0.7, W / 1500))
        (tw, th), _ = cv2.getTextSize(bar_label, font, label_font_scale, 2)
        label_x = (bar_x_left + bar_x_right) // 2 - tw // 2
        label_y = bar_y - 12
        cv2.putText(vis_image, bar_label, (label_x, label_y),
                    font, label_font_scale, (0, 0, 0), 3)
        cv2.putText(vis_image, bar_label, (label_x, label_y),
                    font, label_font_scale, (255, 255, 255), 1)

    # --- OCR search region highlight ---
    if method == 'ocr' and 'region' in scale_info:
        x0, y0, box_w, box_h = scale_info['region']
        cv2.rectangle(vis_image, (x0, y0), (x0 + box_w, y0 + box_h), (0, 0, 255), 2)

        # Draw detected scale bar line
        if 'line_coords' in scale_info:
            leftmost, rightmost, top_row = scale_info['line_coords']
            line_y = y0 + top_row
            cv2.line(vis_image, (x0 + leftmost, line_y), (x0 + rightmost, line_y), (255, 0, 0), 3)

    return vis_image


def draw_zoom_inset(image, center_x, center_y, zoom_factor=4, inset_size=150):
    """
    Draw a magnified inset of the area around a point, placed in the top-right corner.

    Used in Manual scale mode to help the user click precisely on scale bar endpoints.

    Args:
        image (np.ndarray): RGB image (modified in-place copy)
        center_x (int): X coordinate of the point to zoom into
        center_y (int): Y coordinate of the point to zoom into
        zoom_factor (int): Magnification factor (default 4x)
        inset_size (int): Size of the inset square in pixels

    Returns:
        np.ndarray: Image with zoom inset overlay
    """
    vis = image.copy()
    H, W = image.shape[:2]

    # Extract a small patch around the point
    half = inset_size // (2 * zoom_factor)
    x0 = max(0, center_x - half)
    y0 = max(0, center_y - half)
    x1 = min(W, center_x + half)
    y1 = min(H, center_y + half)
    patch = image[y0:y1, x0:x1]

    if patch.size == 0:
        return vis

    # Resize to inset_size x inset_size
    zoomed = cv2.resize(patch, (inset_size, inset_size), interpolation=cv2.INTER_NEAREST)

    # Draw crosshair on the zoomed patch at the exact click location
    cx_in_patch = int((center_x - x0) * (inset_size / max(1, x1 - x0)))
    cy_in_patch = int((center_y - y0) * (inset_size / max(1, y1 - y0)))
    cv2.drawMarker(zoomed, (cx_in_patch, cy_in_patch), (0, 255, 0),
                   cv2.MARKER_CROSS, 20, 2)

    # Place in top-right corner with white border
    ix = W - inset_size - 10
    iy = 10
    if ix > 0 and iy + inset_size < H:
        cv2.rectangle(vis, (ix - 2, iy - 2),
                      (ix + inset_size + 2, iy + inset_size + 2), (255, 255, 255), 2)
        vis[iy:iy + inset_size, ix:ix + inset_size] = zoomed

        # Label
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(0.4, min(0.55, W / 2000))
        cv2.putText(vis, f"Zoom {zoom_factor}x", (ix, iy + inset_size + 16),
                    font, font_scale, (0, 0, 0), 3)
        cv2.putText(vis, f"Zoom {zoom_factor}x", (ix, iy + inset_size + 16),
                    font, font_scale, (255, 255, 255), 1)

    return vis


def draw_manual_scale_overlay(image, points, pixel_length=None):
    """
    Draw the manual scale bar measurement overlay.

    Shows clicked endpoints, the horizontal measurement line, pixel length label,
    and zoom insets for precise clicking.

    Args:
        image (np.ndarray): RGB image
        points (list): List of (x, y) tuples — 1 or 2 points
        pixel_length (int, optional): Pre-computed pixel length if 2 points

    Returns:
        np.ndarray: Annotated image
    """
    vis = image.copy()
    H, W = image.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.5, min(0.7, W / 1500))

    if len(points) >= 1:
        x1, y1 = int(points[0][0]), int(points[0][1])
        # Draw first endpoint marker
        cv2.circle(vis, (x1, y1), 6, (0, 0, 0), -1)
        cv2.circle(vis, (x1, y1), 5, (255, 0, 0), -1)
        cv2.circle(vis, (x1, y1), 8, (255, 0, 0), 2)
        # Zoom inset for first point
        vis = draw_zoom_inset(vis, x1, y1)

    if len(points) >= 2:
        x2 = int(points[1][0])
        # y is locked to y1 (horizontal constraint)
        # Draw second endpoint
        cv2.circle(vis, (x2, y1), 6, (0, 0, 0), -1)
        cv2.circle(vis, (x2, y1), 5, (255, 0, 0), -1)
        cv2.circle(vis, (x2, y1), 8, (255, 0, 0), 2)

        # Draw horizontal measurement line
        cv2.line(vis, (x1, y1), (x2, y1), (255, 0, 0), 2)

        # End ticks
        tick_h = 10
        cv2.line(vis, (x1, y1 - tick_h), (x1, y1 + tick_h), (255, 0, 0), 2)
        cv2.line(vis, (x2, y1 - tick_h), (x2, y1 + tick_h), (255, 0, 0), 2)

        # Pixel length label
        px_len = abs(x2 - x1)
        label = f"{px_len} px"
        mid_x = (x1 + x2) // 2
        cv2.putText(vis, label, (mid_x - 30, y1 - 15),
                    font, font_scale, (0, 0, 0), 3)
        cv2.putText(vis, label, (mid_x - 30, y1 - 15),
                    font, font_scale, (255, 0, 0), 1)

    return vis


def draw_ocr_box_overlay(image, points, ocr_result=None):
    """
    Draw the OCR search box and detection results overlay.

    Shows the user-defined rectangle, detected scale bar line, and OCR text.

    Args:
        image (np.ndarray): RGB image
        points (list): List of (x, y) tuples — 1 or 2 corner points
        ocr_result (dict, optional): Detection result with 'line_coords', 'ocr_text', etc.

    Returns:
        np.ndarray: Annotated image
    """
    vis = image.copy()
    H, W = image.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.5, min(0.7, W / 1500))

    if len(points) >= 1:
        x1, y1 = int(points[0][0]), int(points[0][1])
        # Draw first corner marker
        cv2.drawMarker(vis, (x1, y1), (0, 120, 255), cv2.MARKER_CROSS, 15, 2)

    if len(points) >= 2:
        x2, y2 = int(points[1][0]), int(points[1][1])
        # Normalize to top-left / bottom-right
        bx0, by0 = min(x1, x2), min(y1, y2)
        bx1, by1 = max(x1, x2), max(y1, y2)

        # Draw blue rectangle
        cv2.rectangle(vis, (bx0, by0), (bx1, by1), (0, 120, 255), 2)
        cv2.putText(vis, "OCR box", (bx0, by0 - 6),
                    font, font_scale * 0.7, (0, 0, 0), 3)
        cv2.putText(vis, "OCR box", (bx0, by0 - 6),
                    font, font_scale * 0.7, (0, 120, 255), 1)

        # If we have OCR results, draw the detected line and text
        if ocr_result:
            if 'line_coords' in ocr_result:
                leftmost, rightmost, bar_row = ocr_result['line_coords']
                # Line coords are relative to the crop — offset to image coords
                line_y = by0 + bar_row
                line_x0 = bx0 + leftmost
                line_x1 = bx0 + rightmost
                cv2.line(vis, (line_x0, line_y), (line_x1, line_y), (255, 0, 0), 3)

                # Pixel length label
                px_len = rightmost - leftmost
                mid_x = (line_x0 + line_x1) // 2
                cv2.putText(vis, f"{px_len} px", (mid_x - 30, line_y - 12),
                            font, font_scale, (0, 0, 0), 3)
                cv2.putText(vis, f"{px_len} px", (mid_x - 30, line_y - 12),
                            font, font_scale, (255, 0, 0), 1)

            # Show OCR text below the box
            ocr_text = ocr_result.get('ocr_text', '')
            if ocr_text:
                cv2.putText(vis, f'OCR: "{ocr_text}"', (bx0, by1 + 20),
                            font, font_scale * 0.7, (0, 0, 0), 3)
                cv2.putText(vis, f'OCR: "{ocr_text}"', (bx0, by1 + 20),
                            font, font_scale * 0.7, (0, 255, 0), 1)

    return vis


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
