"""
SAM Model Initialization and Management
"""

import os
import torch
from segment_anything import sam_model_registry, SamPredictor

# Fix OpenMP library conflict (common with Intel MKL and PyTorch)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


class SAMModel:
    """
    Manages the Segment Anything Model (SAM) initialization and device setup.

    Attributes:
        model_type (str): SAM model variant ('vit_h', 'vit_l', or 'vit_b')
        checkpoint (str): Path to SAM checkpoint file
        device (torch.device): Computing device (GPU/CPU)
        predictor (SamPredictor): SAM predictor instance
    """

    def __init__(self, checkpoint_path, model_type="vit_h", device=None):
        """
        Initialize the SAM model.

        Args:
            checkpoint_path (str): Path to SAM checkpoint file
            model_type (str): Model variant - 'vit_h' (best), 'vit_l', or 'vit_b' (fastest)
            device (str or torch.device, optional): Computing device. Auto-detected if None.

        Raises:
            FileNotFoundError: If checkpoint file doesn't exist
            ValueError: If model_type is invalid
        """
        self.checkpoint = checkpoint_path
        self.model_type = model_type

        # Verify checkpoint exists
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"SAM checkpoint not found: {checkpoint_path}\n"
                f"Download from: https://github.com/facebookresearch/segment-anything"
            )

        # Verify model type
        if model_type not in sam_model_registry:
            raise ValueError(
                f"Invalid model_type: {model_type}. "
                f"Choose from: {list(sam_model_registry.keys())}"
            )

        # Setup device
        self.device = self._setup_device(device)

        # Load model
        print(f"Loading SAM model ({model_type})...")
        self.sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        self.sam.to(self.device)

        # Create predictor
        self.predictor = SamPredictor(self.sam)
        print(f"SAM model loaded successfully on {self.device}")

    def _setup_device(self, device=None):
        """
        Determine the best available computing device.

        Priority: Apple Silicon GPU (MPS) > CUDA GPU > CPU

        Handles compatibility across platforms:
        - macOS: Uses MPS (Metal Performance Shaders) for Apple Silicon
        - Windows/Linux: Uses CUDA for NVIDIA GPUs
        - Fallback: CPU if no GPU available

        Args:
            device (str or torch.device, optional): Requested device

        Returns:
            torch.device: Selected device
        """
        if device is not None:
            requested_device = torch.device(device)
            print(f"Using requested device: {requested_device}")
            return requested_device

        # Auto-detect best device
        # Priority 1: Apple Silicon GPU (MPS) - for macOS with M1/M2/M3/M4
        try:
            if torch.backends.mps.is_available() and torch.backends.mps.is_built():
                device = torch.device("mps")
                device_name = "Apple Silicon GPU (MPS)"
                print(f"✓ Using device: {device_name}")
                return device
        except Exception as e:
            print(f"⚠️  MPS detection error: {e}")

        # Priority 2: NVIDIA CUDA GPU - for Windows/Linux with NVIDIA GPUs
        try:
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                device = torch.device("cuda")
                device_name = f"NVIDIA GPU (CUDA) - {gpu_name}"
                print(f"✓ Using device: {device_name}")
                return device
        except Exception as e:
            print(f"⚠️  CUDA detection error: {e}")

        # Priority 3: CPU fallback
        device = torch.device("cpu")
        device_name = "CPU (no GPU detected)"
        print(f"✓ Using device: {device_name}")
        print(f"⚠️  No GPU acceleration available. Processing will be slower.")
        return device

    def set_image(self, image):
        """
        Set the image for SAM prediction (runs encoder once).

        Args:
            image (np.ndarray): RGB image array (H, W, 3)
        """
        self.predictor.set_image(image)

    def predict(self, point_coords=None, point_labels=None, box=None,
                multimask_output=True):
        """
        Generate segmentation masks using SAM.

        Args:
            point_coords (np.ndarray, optional): Nx2 array of point prompts
            point_labels (np.ndarray, optional): N array of labels (1=positive, 0=negative)
            box (np.ndarray, optional): Bounding box [x0, y0, x1, y1]
            multimask_output (bool): Whether to return multiple mask candidates

        Returns:
            tuple: (masks, scores, logits)
                - masks: Boolean arrays of shape (num_masks, H, W)
                - scores: Confidence scores for each mask
                - logits: Raw mask logits
        """
        return self.predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=box,
            multimask_output=multimask_output
        )
