"""
Interactive Particle Refinement

Provides an interactive Jupyter interface for refining particle segmentation
with click-to-delete, click-to-add, merge, and SAM refinement capabilities.
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage import measure, morphology
from skimage.segmentation import clear_border
import ipywidgets as widgets
from IPython.display import display


class InteractiveRefiner:
    """
    Interactive particle refinement interface for Jupyter notebooks.

    Features:
    - Click to delete particles
    - Click to add particles with SAM
    - Merge mode for combining particles
    - Live SAM refinement with point prompts
    - Edge clearing with buffer
    - Dual-view visualization (original + mask)

    Attributes:
        analyzer (ParticleAnalyzer): Particle analyzer instance
        segmenter (ParticleSegmenter): Particle segmenter with SAM
        image (np.ndarray): Original image
        results_callback (callable): Optional callback to save results
    """

    def __init__(self, image, analyzer, segmenter, results_callback=None):
        """
        Initialize the interactive refiner.

        Args:
            image (np.ndarray): RGB image to display
            analyzer (ParticleAnalyzer): ParticleAnalyzer with initial segmentation
            segmenter (ParticleSegmenter): ParticleSegmenter with SAM model
            results_callback (callable, optional): Function to call when saving results.
                Should accept (filename, measurements) as arguments.
        """
        self.image = image
        self.analyzer = analyzer
        self.segmenter = segmenter
        self.results_callback = results_callback

        # Working state
        self.current_mask = analyzer.mask.copy() if analyzer.mask is not None else np.zeros(image.shape[:2], dtype=bool)
        self.H, self.W = self.current_mask.shape

        # Queued operations (Select/Delete mode)
        self.pending_delete_labels = set()
        self.pending_add_points = []
        self.pending_merge_labels = set()

        # SAM refinement state
        self.sam_points = []
        self.sam_labels = []
        self.current_sam_mask = None
        self.base_mask = None

        # UI components
        self._create_ui()

        # Figure
        self.fig, self.axes = plt.subplots(1, 2, figsize=(14, 7))
        self.ax_left, self.ax_right = self.axes

        # Connect events
        self.cid = self.fig.canvas.mpl_connect('button_press_event', self._on_click)

        # Initial analysis
        self._analyze_and_refresh()

        # Initial draw
        self._redraw()

    def _create_ui(self):
        """Create UI widgets."""
        # Mode toggle
        self.mode_toggle = widgets.ToggleButtons(
            options=[('Select/Delete Particles', 'select'), ('Refine with SAM', 'sam')],
            value='select',
            description='Mode:',
            button_style='info'
        )
        self.mode_toggle.observe(self._on_mode_change, names='value')

        # Merge mode toggle
        self.merge_mode_toggle = widgets.ToggleButton(
            value=False,
            description='Merge mode',
            button_style='',
            tooltip='Toggle: left-click selects regions to merge instead of delete'
        )

        # Select/Delete controls
        self.update_btn = widgets.Button(description="Update (apply queued ops)", button_style='primary')
        self.update_btn.on_click(self._on_update_clicked)

        self.clear_queue_btn = widgets.Button(description="Clear queued ops", button_style='warning')
        self.clear_queue_btn.on_click(self._on_clear_queue_clicked)

        self.merge_btn = widgets.Button(description="Merge selected", button_style='')
        self.merge_btn.on_click(self._on_merge_clicked)

        self.edge_buffer_sel = widgets.IntSlider(
            value=0, min=0, max=50, step=1,
            description='Buffer(px):',
            continuous_update=False
        )

        self.clear_edges_btn_select = widgets.Button(
            description="Clear Edges",
            button_style='danger',
            tooltip="Remove components touching the border or within buffer"
        )
        self.clear_edges_btn_select.on_click(self._on_clear_edges_clicked_sel)

        self.finish_btn_select = widgets.Button(
            description="Finish",
            button_style='success',
            tooltip="Store current analysis results"
        )
        self.finish_btn_select.on_click(self._on_finish_clicked)

        # SAM controls
        self.clear_sam_btn = widgets.Button(description="Clear SAM points", button_style='warning')
        self.clear_sam_btn.on_click(self._on_clear_sam_clicked)

        self.apply_sam_btn = widgets.Button(description="Apply SAM to mask", button_style='success')
        self.apply_sam_btn.on_click(self._on_apply_sam_clicked)

        self.edge_buffer_sam = widgets.IntSlider(
            value=0, min=0, max=50, step=1,
            description='Buffer(px):',
            continuous_update=False
        )

        self.clear_edges_btn_sam = widgets.Button(
            description="Clear Edges",
            button_style='danger',
            tooltip="Remove components touching the border or within buffer"
        )
        self.clear_edges_btn_sam.on_click(self._on_clear_edges_clicked_sam)

        self.finish_btn_sam = widgets.Button(
            description="Finish",
            button_style='success',
            tooltip="Store current analysis results"
        )
        self.finish_btn_sam.on_click(self._on_finish_clicked)

        # Status labels
        self.status_label = widgets.HTML(
            "Select/Delete mode: toggle 'Merge mode' ON to select regions for merge with left-click; "
            "when OFF, left-click queues delete. Right-click queues add. "
            "Use 'Update' / 'Merge selected'. For SAM, switch modes above."
        )
        self.count_label = widgets.HTML("")

        # Layout
        self.top_row = widgets.HBox([self.mode_toggle, self.merge_mode_toggle])
        self.select_controls = widgets.HBox([
            self.update_btn,
            self.clear_queue_btn,
            self.merge_btn,
            self.edge_buffer_sel,
            self.clear_edges_btn_select,
            self.finish_btn_select
        ])
        self.sam_controls = widgets.HBox([
            self.clear_sam_btn,
            self.apply_sam_btn,
            self.edge_buffer_sam,
            self.clear_edges_btn_sam,
            self.finish_btn_sam
        ])

    def display(self):
        """Display the interactive interface."""
        display(self.top_row)
        display(self.select_controls)
        display(self.sam_controls)
        display(self.status_label)
        display(self.count_label)

    def _analyze_and_refresh(self, min_size=20, min_area=50):
        """Recompute labeled regions from current_mask with light cleanup."""
        # Clean mask
        clean = morphology.remove_small_objects(self.current_mask.astype(bool), min_size=min_size)
        clean = self._remove_border_artifacts(clean, border_width=4)

        # Update analyzer
        self.analyzer.mask = clean
        self.analyzer.labeled_mask = measure.label(clean, connectivity=2)
        regs = measure.regionprops(self.analyzer.labeled_mask)
        self.analyzer.regions = [r for r in regs if r.area >= min_area]

        self.current_mask = clean

    def _remove_border_artifacts(self, mask, border_width=4):
        """Remove pixels directly on the image border."""
        m = mask.copy()
        m[:border_width, :] = False
        m[-border_width:, :] = False
        m[:, :border_width] = False
        m[:, -border_width:] = False
        return m

    def _draw_contours_on_axes(self, ax, mask_bool, color='red', lw=1.5, alpha=1.0):
        """Draw contours of a mask on an axes."""
        for ctr in measure.find_contours(mask_bool.astype(float), 0.5):
            ax.plot(ctr[:, 1], ctr[:, 0], color=color, lw=lw, alpha=alpha, zorder=5)

    def _scatter_points(self, ax):
        """Draw SAM points with high contrast."""
        if not self.sam_points:
            return
        pts = np.array(self.sam_points, dtype=float)
        labs = np.array(self.sam_labels, dtype=int)

        if (labs == 1).any():
            pos = pts[labs == 1]
            ax.scatter(pos[:, 0], pos[:, 1], marker='+', s=160, linewidths=3,
                      c='lime', zorder=50)
        if (labs == 0).any():
            neg = pts[labs == 0]
            ax.scatter(neg[:, 0], neg[:, 1], marker='x', s=160, linewidths=3,
                      c='red', zorder=50)

    def _black_mask_view(self, mask_bool):
        """Convert boolean mask to grayscale image."""
        img = np.zeros((self.H, self.W), dtype=np.float32)
        if mask_bool is not None:
            img[mask_bool] = 1.0
        return img

    def _redraw(self):
        """Redraw both panes depending on mode."""
        self.ax_left.clear()
        self.ax_right.clear()

        # Style merge toggle
        self.merge_mode_toggle.button_style = 'info' if self.merge_mode_toggle.value else ''

        if self.mode_toggle.value == 'select':
            # LEFT: original image + red outlines + labels
            self.ax_left.imshow(self.image)
            for i, r in enumerate(self.analyzer.regions, start=1):
                mask_i = (self.analyzer.labeled_mask == r.label)
                self._draw_contours_on_axes(self.ax_left, mask_i, color='red', lw=1.2)
                y, x = r.centroid
                self.ax_left.text(x, y, str(i), color='white', fontsize=11,
                                 ha='center', va='center',
                                 bbox=dict(fc='black', alpha=0.6, ec='none'), zorder=20)

            # Queued deletions in yellow
            for lbl in self.pending_delete_labels:
                self._draw_contours_on_axes(self.ax_left, (self.analyzer.labeled_mask == lbl),
                                           color='yellow', lw=1.2)

            # Queued merges in cyan
            for lbl in self.pending_merge_labels:
                self._draw_contours_on_axes(self.ax_left, (self.analyzer.labeled_mask == lbl),
                                           color='cyan', lw=1.8)

            self.ax_left.set_title("Original with Red Outlines + Labels\n(Yellow=delete, Cyan=merge)")
            self.ax_left.axis('off')

            # RIGHT: mask view
            self.ax_right.imshow(self._black_mask_view(self.analyzer.labeled_mask > 0),
                                cmap='gray', vmin=0, vmax=1)
            self.ax_right.set_title("Current Binary Mask (white = particle)")
            self.ax_right.axis('off')

            self.count_label.value = (
                f"<b>Particles:</b> {len(self.analyzer.regions)} | "
                f"<b>Queued deletes:</b> {len(self.pending_delete_labels)} | "
                f"<b>Queued adds:</b> {len(self.pending_add_points)} | "
                f"<b>Queued merge:</b> {len(self.pending_merge_labels)} | "
                f"<b>Merge mode:</b> {'ON' if self.merge_mode_toggle.value else 'OFF'}"
            )
            self.select_controls.layout.display = ''
            self.sam_controls.layout.display = 'none'

        else:  # SAM mode
            # LEFT: fresh image with click markers
            self.ax_left.imshow(self.image)
            self._scatter_points(self.ax_left)
            self.ax_left.set_title("Refine with SAM: Fresh Image (green=+, red=-)")
            self.ax_left.axis('off')

            # RIGHT: live refined mask
            if self.current_sam_mask is not None:
                self.ax_right.imshow(self._black_mask_view(self.current_sam_mask),
                                    cmap='gray', vmin=0, vmax=1)
                n_sam_regs = len(measure.regionprops(measure.label(self.current_sam_mask, connectivity=2)))
                self.ax_right.set_title(f"Refined Mask (white) — Regions: {n_sam_regs}")
            else:
                self.ax_right.imshow(np.zeros((self.H, self.W)), cmap='gray', vmin=0, vmax=1)
                self.ax_right.set_title("Refined Mask (awaiting clicks)")
            self.ax_right.axis('off')

            self.count_label.value = f"<b>SAM points:</b> {len(self.sam_points)}"
            self.select_controls.layout.display = 'none'
            self.sam_controls.layout.display = ''

        self.fig.tight_layout()
        self.fig.canvas.draw_idle()

    def _sam_live_refine(self):
        """Compute a refined mask from SAM using the current click points."""
        if not self.sam_points:
            self.current_sam_mask = None
            return

        # Compute ROI box from base mask
        roi_box = self._compute_roi_box_from_basemask(pad=10)

        # Get SAM predictions
        refined_mask, score = self.segmenter.refine_with_sam(
            self.image,
            self.sam_points,
            self.sam_labels,
            base_mask=self.base_mask,
            multimask_output=True
        )

        # Clean up
        refined_mask = morphology.remove_small_objects(refined_mask.astype(bool), min_size=20)
        refined_mask = self._remove_border_artifacts(refined_mask, border_width=4)

        self.current_sam_mask = refined_mask.astype(bool)

    def _compute_roi_box_from_basemask(self, pad=10):
        """Build a padded ROI box from base_mask."""
        if self.base_mask is None or not self.base_mask.any():
            return np.array([[0, 0, self.W, self.H]])

        ys, xs = np.where(self.base_mask)
        y0, y1 = ys.min(), ys.max()
        x0, x1 = xs.min(), xs.max()
        x0 = max(0, x0 - pad)
        y0 = max(0, y0 - pad)
        x1 = min(self.W - 1, x1 + pad)
        y1 = min(self.H - 1, y1 + pad)
        return np.array([[x0, y0, x1, y1]])

    def _nearest_region_label(self, x, y):
        """Return label of nearest region to (x,y)."""
        if len(self.analyzer.regions) == 0:
            return None, None

        cents = np.array([[r.centroid[1], r.centroid[0]] for r in self.analyzer.regions])
        d2 = np.sum((cents - np.array([x, y]))**2, axis=1)
        idx = int(np.argmin(d2))
        lbl = self.analyzer.regions[idx].label
        return lbl, idx

    def _on_click(self, event):
        """Handle mouse click events."""
        if event.inaxes is None or event.xdata is None or event.ydata is None:
            return
        x, y = float(event.xdata), float(event.ydata)

        if self.mode_toggle.value == 'select':
            if event.button == 1:  # Left-click
                lbl, idx = self._nearest_region_label(x, y)
                if lbl is None:
                    self.status_label.value = "No particles found."
                    return

                if self.merge_mode_toggle.value:
                    self.pending_merge_labels.add(lbl)
                    self.status_label.value = (
                        f"Queued MERGE for particle #{idx+1} (label={lbl}) at ({x:.1f}, {y:.1f})."
                    )
                else:
                    self.pending_delete_labels.add(lbl)
                    self.status_label.value = (
                        f"Queued DELETE for particle #{idx+1} (label={lbl}) at ({x:.1f}, {y:.1f})."
                    )
                self._redraw()

            elif event.button == 3:  # Right-click
                self.pending_add_points.append((x, y))
                self.status_label.value = f"Queued ADD seed at ({x:.1f}, {y:.1f}). Press 'Update' to apply."
                self._redraw()

        else:  # SAM mode
            if event.button == 1:  # Positive point
                self.sam_points.append([x, y])
                self.sam_labels.append(1)
                self.status_label.value = f"SAM: + point at ({x:.1f}, {y:.1f})."
            elif event.button == 3:  # Negative point
                self.sam_points.append([x, y])
                self.sam_labels.append(0)
                self.status_label.value = f"SAM: − point at ({x:.1f}, {y:.1f})."

            self._sam_live_refine()
            self._redraw()

    def _on_mode_change(self, change):
        """Handle mode toggle change."""
        if change['new'] == 'sam':
            self.base_mask = (self.analyzer.labeled_mask > 0).astype(bool).copy()
            self.sam_points.clear()
            self.sam_labels.clear()
            self.current_sam_mask = None
            self.status_label.value = "SAM mode: left-click=positive, right-click=negative."
        else:
            self.status_label.value = (
                "Select/Delete mode: left-click=delete or merge (depending on Merge mode toggle), "
                "right-click=queue add. Use 'Update' / 'Merge selected' to apply."
            )
        self._redraw()

    def _on_update_clicked(self, b):
        """Apply queued delete/add operations."""
        # Apply deletions
        if self.pending_delete_labels:
            del_mask = np.isin(self.analyzer.labeled_mask, list(self.pending_delete_labels))
            self.current_mask = self.current_mask & (~del_mask)

        # Apply additions
        for (x, y) in self.pending_add_points:
            # Use SAM with single positive point
            add_mask, _ = self.segmenter.refine_with_sam(
                self.image,
                [[x, y]],
                [1],
                base_mask=None,
                multimask_output=True
            )
            add_mask = morphology.remove_small_objects(add_mask, min_size=10)
            self.current_mask = self.current_mask | add_mask

        self.pending_delete_labels.clear()
        self.pending_add_points.clear()
        self._analyze_and_refresh()
        self.status_label.value = "Applied queued DELETE/ADD operations. Labels and mask updated."
        self._redraw()

    def _on_clear_queue_clicked(self, b):
        """Clear all queued operations."""
        self.pending_delete_labels.clear()
        self.pending_add_points.clear()
        self.pending_merge_labels.clear()
        self.status_label.value = "Cleared queued delete/add/merge operations."
        self._redraw()

    def _on_merge_clicked(self, b):
        """Merge selected particles."""
        if not self.pending_merge_labels:
            self.status_label.value = "No regions queued for merge."
            return

        # Create merge mask
        merge_mask = np.isin(self.analyzer.labeled_mask, list(self.pending_merge_labels))

        # Apply morphological closing to bridge gaps
        merge_mask_closed = morphology.binary_closing(
            merge_mask.astype(bool),
            morphology.disk(1)
        )

        # Update main mask
        self.current_mask = (self.current_mask & (~merge_mask)) | merge_mask_closed

        self._analyze_and_refresh()
        self.pending_merge_labels.clear()
        self.status_label.value = f"Merged selected regions. New particle count: {len(self.analyzer.regions)}."
        self._redraw()

    def _on_clear_sam_clicked(self, b):
        """Clear SAM points."""
        self.sam_points.clear()
        self.sam_labels.clear()
        self.current_sam_mask = None
        self.status_label.value = "Cleared SAM points."
        self._redraw()

    def _on_apply_sam_clicked(self, b):
        """Apply SAM-refined mask to working mask."""
        if self.current_sam_mask is not None:
            self.current_mask = self.current_sam_mask.copy()
            self._analyze_and_refresh()
            self.status_label.value = "Applied refined SAM mask to working mask."
        else:
            self.status_label.value = "No refined SAM mask to apply."

        self.sam_points.clear()
        self.sam_labels.clear()
        self.current_sam_mask = None
        self.mode_toggle.value = 'select'
        self._redraw()

    def _clear_edges(self, buffer_px):
        """Remove components touching the border."""
        n_before = len(self.analyzer.regions)
        cleaned = clear_border(self.current_mask.astype(bool), buffer_size=int(buffer_px))
        self.current_mask = cleaned.astype(bool)
        self._analyze_and_refresh()
        n_after = len(self.analyzer.regions)
        return max(0, n_before - n_after)

    def _on_clear_edges_clicked_sel(self, b):
        """Clear edges in select mode."""
        removed = self._clear_edges(int(self.edge_buffer_sel.value))
        self.status_label.value = f"Cleared {removed} edge-touching components (buffer {int(self.edge_buffer_sel.value)} px)."
        self._redraw()

    def _on_clear_edges_clicked_sam(self, b):
        """Clear edges in SAM mode."""
        removed = self._clear_edges(int(self.edge_buffer_sam.value))
        self.base_mask = (self.analyzer.labeled_mask > 0).astype(bool).copy()
        self.current_sam_mask = None
        self.sam_points.clear()
        self.sam_labels.clear()
        self.status_label.value = (
            f"Cleared {removed} edge-touching components (buffer {int(self.edge_buffer_sam.value)} px). SAM reset."
        )
        self._redraw()

    def _on_finish_clicked(self, b):
        """Save current results."""
        if self.results_callback is None:
            self.status_label.value = "No results callback configured."
            return

        # Get measurements
        measurements = self.analyzer.get_measurements(in_nm=bool(self.analyzer.conversion))

        # Call callback
        try:
            self.results_callback(measurements)
            self.status_label.value = f"Saved results — {len(self.analyzer.regions)} particles."
        except Exception as e:
            self.status_label.value = f"<b>Save failed:</b> {e}"

    def get_final_mask(self):
        """
        Get the final refined mask.

        Returns:
            np.ndarray: Final boolean mask
        """
        return self.current_mask.copy()

    def get_measurements(self, in_nm=True):
        """
        Get current particle measurements.

        Args:
            in_nm (bool): Whether to return measurements in nanometers

        Returns:
            dict: Measurements dictionary
        """
        return self.analyzer.get_measurements(in_nm=in_nm)
