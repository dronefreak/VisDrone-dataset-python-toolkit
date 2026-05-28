"""
YOLO v8+ model wrappers for VisDrone detection.

Provides unified interface for YOLO models using Ultralytics:
- YOLOv8  (2023): yolov8n/s/m/l/x + seg variants
- YOLOv9  (2024): yolov9c/m/e
- YOLOv10 (2024): yolov10n/s/m/b/l/x
- YOLO11  (2024): yolo11n/s/m/l/x
- YOLO26  (2025): yolo26n/s/m/l/x

Requires: pip install ultralytics>=8.0.0
"""

from typing import Any, Dict, List, Optional

import torch

from .abstract_models import DetectionModel, ModelRegistry
from .format_converters import YOLOFormatConverter


class YOLOv8Base(DetectionModel):
    """
    Base class for YOLOv8 models.

    Wraps Ultralytics YOLO implementation and adapts it to the DetectionModel interface.
    """

    # Model names for Ultralytics
    ULTRALYTICS_MODEL = "yolov8n.pt"  # Will be overridden in subclasses

    def __init__(
        self,
        num_classes: int = 12,
        _pretrained: bool = True,
        device: str = "cuda",
        imgsz: int = 640,
        **_kwargs: Any,
    ):
        """
        Initialize YOLOv8 model.

        Args:
            num_classes: Number of detection classes (default: 12 for VisDrone)
            _pretrained: Load pretrained COCO weights (default: True, unused)
            device: Device to load model on (default: 'cuda')
            imgsz: Input image size (default: 640)
            **_kwargs: Additional arguments for Ultralytics YOLO (unused)
        """
        super().__init__(num_classes=num_classes)

        try:
            from ultralytics import YOLO
        except ImportError as err:
            raise ImportError(
                "Ultralytics YOLO not installed. Install with: pip install ultralytics>=8.0.0"
            ) from err

        # Load model
        self.model = YOLO(self.ULTRALYTICS_MODEL)
        self.device_name = device
        self.imgsz = imgsz
        self.format_converter = YOLOFormatConverter()

        # Set number of classes
        if hasattr(self.model.model, "nc"):
            self.model.model.nc = num_classes
        if hasattr(self.model, "model") and hasattr(self.model.model, "nc"):
            self.model.model.nc = num_classes

        # Move to device
        if device.startswith("cuda"):
            self.model.to(device)

        # Store original forward for delegation
        self._yolo_model = self.model

    def forward(
        self,
        images: List[torch.Tensor],
        targets: Optional[List[Dict[str, torch.Tensor]]] = None,
    ):
        """
        Forward pass for YOLOv8 model.

        Args:
            images: List of input images as tensors with shape (C, H, W)
            targets: List of target dicts (only used in training context)

        Returns:
            During training: Loss value (delegated to Ultralytics training)
            During inference: List of dicts with 'boxes', 'labels', 'scores'
        """
        if not self.training:
            # Inference mode
            return self._inference(images)
        else:
            # Training mode - requires special handling
            if targets is not None:
                return self._training_forward(images, targets)
            else:
                # If no targets in training mode, fall back to inference
                return self._inference(images)

    def _inference(self, images: List[torch.Tensor]) -> List[Dict[str, torch.Tensor]]:
        """
        Perform inference with YOLO model.

        Args:
            images: List of input images

        Returns:
            List of detection dicts with 'boxes', 'labels', 'scores'
        """
        # Convert list of tensors to batch
        # Ultralytics expects batched input
        batch = torch.stack(images) if isinstance(images, list) and len(images) > 0 else images

        # Run inference
        with torch.no_grad():
            results = self._yolo_model(batch, imgsz=self.imgsz, verbose=False)

        # Convert results to standard format
        predictions = []
        for result in results:
            pred_dict = {
                "boxes": result.boxes.xyxy,  # [x1, y1, x2, y2] format
                "labels": result.boxes.cls.long(),
                "scores": result.boxes.conf,
            }
            predictions.append(pred_dict)

        return predictions

    def _training_forward(
        self,
        images: List[torch.Tensor],
        _targets: List[Dict[str, torch.Tensor]],
    ):
        """Not implemented — YOLO training is handled by YOLOTrainer (Ultralytics engine).

        Calling model.forward() in training mode is not meaningful for YOLO.
        Use YOLOTrainer.train() from visdrone_toolkit.yolo_trainer instead.
        """
        raise NotImplementedError(
            "Direct YOLO training via forward() is not supported. "
            "Use YOLOTrainer from visdrone_toolkit.yolo_trainer, which delegates "
            "to the Ultralytics training engine with correct loss computation."
        )

    def get_input_format(self) -> str:
        """Return YOLO input format (normalized coordinates)."""
        return "yolo"

    def get_output_format(self) -> str:
        """Return YOLO output format."""
        return "coco_dict"  # Converted to standard format

    def freeze_backbone(self, num_layers: Optional[int] = None) -> None:
        """Freeze backbone layers for fine-tuning."""
        if hasattr(self.model, "model"):
            backbone = self.model.model
            if hasattr(backbone, "model"):
                # Freeze backbone
                for param in backbone.model[: num_layers or -2].parameters():
                    param.requires_grad = False

    def train(self, mode: bool = True):
        """Set training mode."""
        self.training = mode
        if hasattr(self._yolo_model, "train"):
            self._yolo_model.train(mode)
        return self

    def eval(self):
        """Set evaluation mode."""
        self.training = False
        if hasattr(self._yolo_model, "eval"):
            self._yolo_model.eval()
        return self


@ModelRegistry.register("yolov8n")
class YOLOv8Nano(YOLOv8Base):
    """
    YOLOv8 Nano - Smallest YOLO model.

    Best for:
    - Edge devices with limited compute
    - Real-time inference with low latency
    - Embedded systems and drones

    Specs:
    - Parameters: ~3.2M
    - Speed: ~80 FPS on RTX 4090
    - mAP (COCO): ~37.3%
    - Model size: ~6.3 MB
    """

    ULTRALYTICS_MODEL = "yolov8n.pt"


@ModelRegistry.register("yolov8s")
class YOLOv8Small(YOLOv8Base):
    """
    YOLOv8 Small - Small YOLO model.

    Best for:
    - Balance between speed and accuracy
    - Real-time applications
    - Resource-constrained systems

    Specs:
    - Parameters: ~11.2M
    - Speed: ~28.5 FPS on RTX 4090
    - mAP (COCO): ~44.9%
    - Model size: ~22.5 MB
    """

    ULTRALYTICS_MODEL = "yolov8s.pt"


@ModelRegistry.register("yolov8m")
class YOLOv8Medium(YOLOv8Base):
    """
    YOLOv8 Medium - Medium YOLO model.

    Best for:
    - Good accuracy with reasonable speed
    - Production systems with moderate compute
    - Balanced performance-accuracy trade-off

    Specs:
    - Parameters: ~25.9M
    - Speed: ~17.3 FPS on RTX 4090
    - mAP (COCO): ~50.2%
    - Model size: ~52.0 MB
    """

    ULTRALYTICS_MODEL = "yolov8m.pt"


@ModelRegistry.register("yolov8l")
class YOLOv8Large(YOLOv8Base):
    """
    YOLOv8 Large - Large YOLO model.

    Best for:
    - High accuracy requirements
    - GPU-equipped systems
    - Maximum performance scenarios

    Specs:
    - Parameters: ~43.7M
    - Speed: ~10.8 FPS on RTX 4090
    - mAP (COCO): ~52.9%
    - Model size: ~87.7 MB
    """

    ULTRALYTICS_MODEL = "yolov8l.pt"


@ModelRegistry.register("yolov8x")
class YOLOv8ExtraLarge(YOLOv8Base):
    """
    YOLOv8 Extra Large - Largest YOLO model.

    Best for:
    - Maximum accuracy priority
    - Multi-GPU systems
    - Research and benchmarking

    Specs:
    - Parameters: ~68.2M
    - Speed: ~7.5 FPS on RTX 4090
    - mAP (COCO): ~53.9%
    - Model size: ~135.4 MB
    """

    ULTRALYTICS_MODEL = "yolov8x.pt"


@ModelRegistry.register("yolov8n-seg")
class YOLOv8NanoSeg(YOLOv8Base):
    """YOLOv8 Nano with instance segmentation."""

    ULTRALYTICS_MODEL = "yolov8n-seg.pt"


@ModelRegistry.register("yolov8s-seg")
class YOLOv8SmallSeg(YOLOv8Base):
    """YOLOv8 Small with instance segmentation."""

    ULTRALYTICS_MODEL = "yolov8s-seg.pt"


@ModelRegistry.register("yolov8m-seg")
class YOLOv8MediumSeg(YOLOv8Base):
    """YOLOv8 Medium with instance segmentation."""

    ULTRALYTICS_MODEL = "yolov8m-seg.pt"


@ModelRegistry.register("yolov8l-seg")
class YOLOv8LargeSeg(YOLOv8Base):
    """YOLOv8 Large with instance segmentation."""

    ULTRALYTICS_MODEL = "yolov8l-seg.pt"


@ModelRegistry.register("yolov8x-seg")
class YOLOv8ExtraLargeSeg(YOLOv8Base):
    """YOLOv8 Extra Large with instance segmentation."""

    ULTRALYTICS_MODEL = "yolov8x-seg.pt"


@ModelRegistry.register("yolov9c")
class YOLOv9Compact(YOLOv8Base):
    """
    YOLOv9 Compact - Latest YOLO version (compact variant).

    v9 improvements:
    - Better accuracy
    - Faster inference
    - Improved training stability
    """

    ULTRALYTICS_MODEL = "yolov9c.pt"


@ModelRegistry.register("yolov9m")
class YOLOv9Medium(YOLOv8Base):
    """YOLOv9 Medium - Latest YOLO version (medium variant)."""

    ULTRALYTICS_MODEL = "yolov9m.pt"


@ModelRegistry.register("yolov9e")
class YOLOv9Extended(YOLOv8Base):
    """YOLOv9 Extended - Latest YOLO version (large variant)."""

    ULTRALYTICS_MODEL = "yolov9e.pt"


@ModelRegistry.register("yolov10n")
class YOLOv10Nano(YOLOv8Base):
    """
    YOLOv10 Nano - Next-gen YOLO (nano variant).

    v10 improvements:
    - No anchor NMS (more efficient)
    - Better overall accuracy
    - Improved speed
    """

    ULTRALYTICS_MODEL = "yolov10n.pt"


@ModelRegistry.register("yolov10s")
class YOLOv10Small(YOLOv8Base):
    """YOLOv10 Small - Next-gen YOLO (small variant)."""

    ULTRALYTICS_MODEL = "yolov10s.pt"


@ModelRegistry.register("yolov10m")
class YOLOv10Medium(YOLOv8Base):
    """YOLOv10 Medium - Next-gen YOLO (medium variant)."""

    ULTRALYTICS_MODEL = "yolov10m.pt"


@ModelRegistry.register("yolov10b")
class YOLOv10Base(YOLOv8Base):
    """YOLOv10 Base - Next-gen YOLO (base variant)."""

    ULTRALYTICS_MODEL = "yolov10b.pt"


@ModelRegistry.register("yolov10l")
class YOLOv10Large(YOLOv8Base):
    """YOLOv10 Large - Next-gen YOLO (large variant)."""

    ULTRALYTICS_MODEL = "yolov10l.pt"


@ModelRegistry.register("yolov10x")
class YOLOv10ExtraLarge(YOLOv8Base):
    """YOLOv10 Extra Large - Next-gen YOLO (xl variant)."""

    ULTRALYTICS_MODEL = "yolov10x.pt"


# ---------------------------------------------------------------------------
# YOLO11 — 2024 architecture (C3k2 + C2PSA blocks)
# ---------------------------------------------------------------------------


@ModelRegistry.register("yolo11n")
class YOLO11Nano(YOLOv8Base):
    """
    YOLO11 Nano — 2024 Ultralytics architecture.

    Improvements over v8:
    - C3k2 blocks replace C2f for improved feature extraction
    - C2PSA attention module in the neck
    - Same params as v8n with better accuracy

    Specs:
    - Parameters: ~2.6M
    - mAP (COCO): ~39.5%
    - Model size: ~5.4 MB
    """

    ULTRALYTICS_MODEL = "yolo11n.pt"


@ModelRegistry.register("yolo11s")
class YOLO11Small(YOLOv8Base):
    """YOLO11 Small — 2024 architecture (small variant).

    Specs:
    - Parameters: ~9.5M
    - mAP (COCO): ~47.0%
    - Model size: ~18.4 MB
    """

    ULTRALYTICS_MODEL = "yolo11s.pt"


@ModelRegistry.register("yolo11m")
class YOLO11Medium(YOLOv8Base):
    """YOLO11 Medium — 2024 architecture (medium variant).

    Specs:
    - Parameters: ~20.1M
    - mAP (COCO): ~51.5%
    - Model size: ~38.8 MB
    """

    ULTRALYTICS_MODEL = "yolo11m.pt"


@ModelRegistry.register("yolo11l")
class YOLO11Large(YOLOv8Base):
    """YOLO11 Large — 2024 architecture (large variant).

    Specs:
    - Parameters: ~25.4M
    - mAP (COCO): ~53.4%
    - Model size: ~49.0 MB
    """

    ULTRALYTICS_MODEL = "yolo11l.pt"


@ModelRegistry.register("yolo11x")
class YOLO11ExtraLarge(YOLOv8Base):
    """YOLO11 Extra Large — 2024 architecture (xl variant).

    Specs:
    - Parameters: ~57.0M
    - mAP (COCO): ~54.7%
    - Model size: ~109 MB
    """

    ULTRALYTICS_MODEL = "yolo11x.pt"


# ---------------------------------------------------------------------------
# YOLO26 — 2025 architecture (improved efficiency over v11)
# ---------------------------------------------------------------------------


@ModelRegistry.register("yolo26n")
class YOLO26Nano(YOLOv8Base):
    """
    YOLO26 Nano — 2025 Ultralytics architecture.

    Improvements over v11/v8:
    - More efficient backbone with fewer parameters at same accuracy
    - Better small-object detection (relevant for VisDrone)
    - Refined neck and detection head

    Specs:
    - Parameters: ~2.6M
    - mAP (COCO): ~39+ (better efficiency than v8n)
    - Model size: ~5.3 MB
    """

    ULTRALYTICS_MODEL = "yolo26n.pt"


@ModelRegistry.register("yolo26s")
class YOLO26Small(YOLOv8Base):
    """YOLO26 Small — 2025 architecture (small variant).

    Specs:
    - Parameters: ~10.0M
    - Model size: ~19.5 MB
    """

    ULTRALYTICS_MODEL = "yolo26s.pt"


@ModelRegistry.register("yolo26m")
class YOLO26Medium(YOLOv8Base):
    """YOLO26 Medium — 2025 architecture (medium variant).

    Specs:
    - Parameters: ~21.9M
    - Model size: ~42.2 MB
    """

    ULTRALYTICS_MODEL = "yolo26m.pt"


@ModelRegistry.register("yolo26l")
class YOLO26Large(YOLOv8Base):
    """YOLO26 Large — 2025 architecture (large variant).

    Specs:
    - Parameters: ~26.3M
    - Model size: ~50.7 MB
    """

    ULTRALYTICS_MODEL = "yolo26l.pt"


@ModelRegistry.register("yolo26x")
class YOLO26ExtraLarge(YOLOv8Base):
    """YOLO26 Extra Large — 2025 architecture (xl variant).

    Specs:
    - Parameters: ~59.0M
    - Model size: ~113 MB
    """

    ULTRALYTICS_MODEL = "yolo26x.pt"
