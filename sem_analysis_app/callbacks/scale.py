"""
Processing tab: scale detection (metadata, OCR, manual) and cropping.
"""
import gradio as gr

from ..visualization import (
    visualize_scale_verification,
    draw_manual_scale_overlay,
    draw_ocr_box_overlay,
)
from ..state import state

def detect_scale_metadata(progress=gr.Progress()):
    """Detect scale from TIFF metadata (Metadata mode).

    Returns: (status, image_viz, crop_slider_value)
    """
    try:
        if state.current_image is None:
            return "❌ No image loaded", None, gr.update()

        if state.scale_detector is None:
            return "❌ Scale detector not initialized", None, gr.update()

        file_path = state.image_paths[state.current_index] if state.image_paths else None

        # Check if this is the first detection for this image
        # (before overwriting scale_info with new results)
        first_detection = (state.scale_info is None)

        progress(0.2, desc="Reading TIFF metadata...")

        try:
            state.scale_info = state.scale_detector.detect_scale(
                state.current_image,
                file_path=file_path,
                method='metadata'
            )
        except ValueError as e:
            return f"❌ Metadata scale detection failed: {str(e)}", None, gr.update()

        progress(0.5, desc="Detecting databar...")

        databar_info = state.scale_detector.detect_databar(
            state.current_image,
            state.scale_info.get('raw_metadata')
        )

        if databar_info.get('has_databar'):
            auto_crop_pct = round(databar_info['databar_fraction'] * 100, 1)
        else:
            auto_crop_pct = 0.0

        # On first detection, set the slider to the auto-detected value.
        # On subsequent detections, keep the user's slider value.
        if first_detection:
            state.crop_percent = auto_crop_pct
            slider_update = auto_crop_pct
        else:
            slider_update = gr.update()  # Don't touch the slider

        # Apply crop using current state value
        crop_pct = state.crop_percent
        if crop_pct > 0:
            state.cropped_image = state.scale_detector.crop_scale_bar(
                state.current_image, crop_percent=crop_pct
            )
        else:
            state.cropped_image = state.current_image.copy()

        progress(0.8, desc="Creating verification visualization...")

        scale_viz = visualize_scale_verification(
            state.current_image, state.scale_info, databar_info
        )

        progress(1.0, desc="Complete!")

        # Build status
        conversion = state.scale_info['conversion']
        manufacturer = state.scale_info.get('manufacturer', 'unknown')
        confidence = state.scale_info.get('confidence', 'unknown')
        metadata_source = state.scale_info.get('metadata_source', 'TIFF metadata')

        status_parts = []
        if manufacturer != 'unknown':
            status_parts.append(f"✅ {manufacturer.upper()} metadata: {conversion:.4f} nm/pixel [{confidence} confidence]")
        else:
            status_parts.append(f"✅ {metadata_source}: {conversion:.4f} nm/pixel [{confidence} confidence]")

        if auto_crop_pct > 0:
            status_parts.append(f"Databar detected ({auto_crop_pct:.1f}%)")
        else:
            status_parts.append("No databar detected — adjust crop slider if needed")

        return " | ".join(status_parts), scale_viz, slider_update

    except Exception as e:
        return f"❌ Error: {str(e)}", None, gr.update()


def detect_scale_ocr_in_box():
    """Run OCR scale detection inside the user-defined box (OCR mode)."""
    try:
        if state.current_image is None:
            return "❌ No image loaded", None, ""

        if state.scale_detector is None:
            return "❌ Scale detector not initialized", None, ""

        if len(state.ocr_click_points) < 2:
            return "❌ Click two corners on the image first", None, ""

        (x1, y1), (x2, y2) = state.ocr_click_points
        bx0, by0 = min(int(x1), int(x2)), min(int(y1), int(y2))
        bx1, by1 = max(int(x1), int(x2)), max(int(y1), int(y2))

        H, W = state.current_image.shape[:2]
        bx0, by0 = max(0, bx0), max(0, by0)
        bx1, by1 = min(W, bx1), min(H, by1)

        if bx1 - bx0 < 10 or by1 - by0 < 10:
            return "❌ Box too small — click two wider-spaced corners", None, ""

        # Convert box to fractional region parameters for detect_scale_bar
        cx = (bx0 + bx1) / 2 / W
        cy = (by0 + by1) / 2 / H
        rw = (bx1 - bx0) / W
        rh = (by1 - by0) / H

        try:
            result = state.scale_detector.detect_scale_bar(
                state.current_image,
                region_x=cx, region_y=cy,
                region_width=rw, region_height=rh,
                polarity='auto', threshold=200
            )
        except ValueError as e:
            # Still show the box
            viz = draw_ocr_box_overlay(state.current_image, state.ocr_click_points)
            return f"❌ OCR failed: {str(e)}", viz, f"OCR failed: {e}"

        state.scale_info = result

        # Detect databar for cropping
        databar_info = state.scale_detector.detect_databar(state.current_image)
        if databar_info.get('has_databar'):
            crop_pct = databar_info['databar_fraction'] * 100
            state.cropped_image = state.scale_detector.crop_scale_bar(
                state.current_image, crop_percent=crop_pct
            )
        else:
            state.cropped_image = state.current_image.copy()

        # Draw overlay with OCR results
        ocr_result_for_viz = {
            'line_coords': result.get('line_coords'),
            'ocr_text': result.get('ocr_text', ''),
        }
        viz = draw_ocr_box_overlay(state.current_image, state.ocr_click_points, ocr_result_for_viz)

        conversion = result['conversion']
        ocr_text = result.get('ocr_text', '')
        pixel_length = result.get('pixel_length', 0)
        info_text = f"Bar: {pixel_length} px | OCR: \"{ocr_text}\" | {conversion:.4f} nm/px"

        return f"✅ OCR: {conversion:.4f} nm/pixel", viz, info_text

    except Exception as e:
        return f"❌ Error: {str(e)}", None, ""


def apply_manual_scale(um_value):
    """Apply manual scale from clicked endpoints + user-entered µm value (Manual mode)."""
    try:
        if state.current_image is None:
            return "❌ No image loaded", None, ""

        if state.scale_detector is None:
            return "❌ Scale detector not initialized", None, ""

        if len(state.manual_click_points) < 2:
            return "❌ Click two endpoints on the image first", None, ""

        if um_value is None or um_value <= 0:
            return "❌ Enter a positive scale bar length in µm", None, ""

        x1 = int(state.manual_click_points[0][0])
        x2 = int(state.manual_click_points[1][0])
        pixel_length = abs(x2 - x1)

        if pixel_length < 2:
            return "❌ Endpoints too close together", None, ""

        nm_value = float(um_value) * 1000  # µm → nm
        conversion = nm_value / pixel_length

        # Set scale via the detector
        state.scale_detector.set_manual_scale(conversion, 1)
        state.scale_info = state.scale_detector.last_detection

        # Detect databar for cropping
        databar_info = state.scale_detector.detect_databar(state.current_image)
        if databar_info.get('has_databar'):
            crop_pct = databar_info['databar_fraction'] * 100
            state.cropped_image = state.scale_detector.crop_scale_bar(
                state.current_image, crop_percent=crop_pct
            )
        else:
            state.cropped_image = state.current_image.copy()

        # Show overlay
        viz = draw_manual_scale_overlay(state.current_image, state.manual_click_points, pixel_length)

        info = f"Bar: {pixel_length} px = {um_value:.2f} µm → {conversion:.4f} nm/px"
        return f"✅ Manual: {conversion:.4f} nm/pixel ({pixel_length} px = {um_value} µm)", viz, info

    except Exception as e:
        return f"❌ Error: {str(e)}", None, ""


def handle_scale_click(evt: gr.SelectData):
    """Handle clicks on the image for OCR and Manual scale detection modes.

    Returns: (image, scale_status, ocr_info, manual_info)
    """
    try:
        if state.current_image is None:
            return None, "", "", ""

        x, y = evt.index[0], evt.index[1]

        if state.scale_mode == "Metadata":
            # No click interaction in Metadata mode
            return state.current_image, "", "", ""

        elif state.scale_mode == "OCR":
            if len(state.ocr_click_points) >= 2:
                # Already have 2 points — reset for new box
                state.ocr_click_points = []

            state.ocr_click_points.append((x, y))

            if len(state.ocr_click_points) == 1:
                # First corner — draw marker
                viz = draw_ocr_box_overlay(state.current_image, state.ocr_click_points)
                return viz, "", f"Corner 1 at ({x}, {y}). Click corner 2...", ""

            else:
                # Second corner — draw box and auto-run OCR
                status, viz, info = detect_scale_ocr_in_box()
                return viz, status, info, ""

        elif state.scale_mode == "Manual":
            if len(state.manual_click_points) >= 2:
                # Already have 2 points — reset for new measurement
                state.manual_click_points = []

            state.manual_click_points.append((x, y))

            if len(state.manual_click_points) == 1:
                # First endpoint — draw marker with zoom inset
                viz = draw_manual_scale_overlay(state.current_image, state.manual_click_points)
                return viz, "", "", f"Left endpoint at ({x}, {y}). Click the right end..."

            else:
                # Second endpoint — horizontal lock, draw line
                x1, y1 = state.manual_click_points[0]
                # Lock y to first point's y (horizontal constraint)
                state.manual_click_points[1] = (x, y1)
                viz = draw_manual_scale_overlay(state.current_image, state.manual_click_points)
                px_len = abs(int(x) - int(x1))
                return viz, "", "", f"Bar: {px_len} px. Enter length in µm and click Apply."

        return state.current_image, "", "", ""

    except Exception as e:
        return state.current_image, f"❌ Error: {str(e)}", "", ""


def set_scale_mode(mode):
    """Handle scale detection mode change."""
    state.scale_mode = mode
    state.ocr_click_points = []
    state.manual_click_points = []

    show_ocr = (mode == "OCR")
    show_manual = (mode == "Manual")

    # Restore original image display when switching modes
    img = state.current_image

    info_msg = ""
    if mode == "OCR":
        info_msg = "Click point 1 of 2..."
    elif mode == "Manual":
        info_msg = "Click point 1 of 2..."

    return (
        gr.update(visible=show_ocr),    # ocr_controls group
        gr.update(visible=show_manual),  # manual_controls group
        "Ready",                         # scale_status
        img,                             # current_image
        info_msg,                        # ocr_info
        info_msg,                        # manual_info
    )


def reset_scale_clicks():
    """Reset click points for current scale detection mode."""
    if state.scale_mode == "OCR":
        state.ocr_click_points = []
    elif state.scale_mode == "Manual":
        state.manual_click_points = []

    img = state.current_image
    return img, "Reset — click on the image to start."


def adjust_crop(crop_percent):
    """Adjust the bottom crop percentage and re-crop the image."""
    try:
        state.crop_percent = float(crop_percent)

        if state.current_image is None:
            return f"Crop set to {crop_percent:.1f}%", None

        if state.scale_detector is None:
            return f"Crop set to {crop_percent:.1f}%", None

        # Re-crop the image
        if crop_percent > 0:
            state.cropped_image = state.scale_detector.crop_scale_bar(
                state.current_image, crop_percent=crop_percent
            )
        else:
            state.cropped_image = state.current_image.copy()

        # Show the crop line on the image as a preview
        import cv2
        vis = state.current_image.copy()
        H, W = vis.shape[:2]
        crop_y = int(H * (1 - crop_percent / 100))
        cv2.line(vis, (0, crop_y), (W, crop_y), (0, 255, 0), 2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(0.4, min(0.55, W / 2000))
        cv2.putText(vis, f"Crop at {crop_percent:.1f}%", (10, crop_y - 8),
                    font, font_scale, (0, 0, 0), 3)
        cv2.putText(vis, f"Crop at {crop_percent:.1f}%", (10, crop_y - 8),
                    font, font_scale, (0, 255, 0), 1)

        return f"✓ Crop: {crop_percent:.1f}% from bottom", vis

    except Exception as e:
        return f"❌ Error: {str(e)}", gr.update()


def detect_scale_clicked(progress=gr.Progress()):
    """Handle the Detect Scale button click based on current mode.

    Returns: (status, image_viz, crop_slider_value)
    """
    if state.scale_mode == "Metadata":
        return detect_scale_metadata(progress)
    elif state.scale_mode == "OCR":
        if len(state.ocr_click_points) == 2:
            status, viz, info = detect_scale_ocr_in_box()
            return status, viz, gr.update()
        return "❌ Click two corners on the image first", state.current_image, gr.update()
    elif state.scale_mode == "Manual":
        return "Use the image clicks and µm input below", state.current_image, gr.update()
    return "❌ Unknown mode", None, gr.update()
