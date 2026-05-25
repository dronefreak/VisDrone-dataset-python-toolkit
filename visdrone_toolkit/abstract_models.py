"""
Abstract base classes and interfaces for detection models.

This module defines the interfaces that all detection models must implement,
enabling seamless integration of different architectures (torchvision, YOLO, DETR, etc.)
into a unified training and inference pipeline.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn


class DetectionModel(nn.Module, ABC):
    """
    Abstract base class for all detection models.

    All detection models must inherit from this class and implement the required methods.
    This ensures a consistent interface across different frameworks (torchvision, YOLO, DETR).
    """

    def __init__(self, num_classes: int = 12, **_kwargs):
        """
        Initialize detection model.

        Args:
            num_classes: Number of detection classes (default: 12 for VisDrone)
            **_kwargs: Model-specific arguments (unused in base class)
        """
        super().__init__()
        self.num_classes = num_classes

    @abstractmethod
    def forward(self, images: List[torch.Tensor], targets: Optional[List[Dict]] = None) -> Any:
        """
        Forward pass for detection model.

        Args:
            images: List of input images as tensors with shape (C, H, W)
            targets: List of target dicts with keys:
                     - 'boxes': Tensor of shape (N, 4) - bounding boxes
                     - 'labels': Tensor of shape (N,) - class labels
                     Only required during training.

        Returns:
            During training: Dict with loss values (model-specific)
            During inference: List of dicts with keys:
                              - 'boxes': Tensor of shape (N, 4)
                              - 'labels': Tensor of shape (N,)
                              - 'scores': Tensor of shape (N,) - confidence scores
        """
        raise NotImplementedError

    @abstractmethod
    def get_input_format(self) -> str:
        """
        Get the box format expected by this model.

        Returns:
            'coco': [x1, y1, x2, y2] format (absolute coordinates)
            'yolo': [x_center, y_center, w, h] format (normalized 0-1)
            Other model-specific formats
        """
        raise NotImplementedError

    @abstractmethod
    def get_output_format(self) -> str:
        """
        Get the output format produced by this model.

        Returns:
            'coco_dict': Standard dict with boxes, labels, scores
            'yolo_results': Ultralytics Results object
            Other model-specific formats
        """
        raise NotImplementedError

    def get_trainable_parameters(self) -> int:
        """Get number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def freeze_backbone(self, num_layers: Optional[int] = None) -> None:
        """
        Freeze backbone layers for fine-tuning.

        Args:
            num_layers: Number of layers from end to freeze.
                       If None, freeze entire backbone.
        """
        # Default implementation - subclasses can override
        pass

    def unfreeze_backbone(self) -> None:
        """Unfreeze all backbone layers."""
        if hasattr(self, "model"):
            for param in self.model.parameters():
                param.requires_grad = True


class FormatConverter(ABC):
    """
    Abstract base class for converting between different box formats.

    Different models expect different box representations:
    - COCO format: [x1, y1, x2, y2] (absolute coordinates)
    - YOLO format: [x_center, y_center, w, h] (normalized 0-1)
    - DETR format: [x1, y1, x2, y2] with additional metadata
    """

    @abstractmethod
    def to_internal_format(
        self, targets: List[Dict[str, torch.Tensor]]
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Convert from model-specific format to internal COCO format.

        Args:
            targets: List of target dicts in model-specific format

        Returns:
            List of target dicts in internal format with keys:
            - 'boxes': Tensor of shape (N, 4) in [x1, y1, x2, y2] format
            - 'labels': Tensor of shape (N,) with class labels
        """
        raise NotImplementedError

    @abstractmethod
    def from_internal_format(
        self, targets: List[Dict[str, torch.Tensor]]
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Convert from internal COCO format to model-specific format.

        Args:
            targets: List of target dicts in internal format

        Returns:
            List of target dicts in model-specific format
        """
        raise NotImplementedError

    @staticmethod
    def coco_to_yolo(boxes: torch.Tensor, image_size: Tuple[int, int]) -> torch.Tensor:
        """
        Convert COCO format to YOLO format.

        Args:
            boxes: Tensor of shape (N, 4) in [x1, y1, x2, y2] format
            image_size: (height, width) of image for normalization

        Returns:
            Tensor of shape (N, 4) in [x_center, y_center, w, h] normalized format
        """
        if len(boxes) == 0:
            return boxes

        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        h, w = image_size

        # Convert to center format
        x_center = (x1 + x2) / 2.0
        y_center = (y1 + y2) / 2.0
        width = x2 - x1
        height = y2 - y1

        # Normalize
        x_center = x_center / w
        y_center = y_center / h
        width = width / w
        height = height / h

        return torch.stack([x_center, y_center, width, height], dim=1)

    @staticmethod
    def yolo_to_coco(boxes: torch.Tensor, image_size: Tuple[int, int]) -> torch.Tensor:
        """
        Convert YOLO format to COCO format.

        Args:
            boxes: Tensor of shape (N, 4) in [x_center, y_center, w, h] normalized format
            image_size: (height, width) of image for denormalization

        Returns:
            Tensor of shape (N, 4) in [x1, y1, x2, y2] absolute format
        """
        if len(boxes) == 0:
            return boxes

        x_center, y_center, width, height = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        h, w = image_size

        # Denormalize
        x_center = x_center * w
        y_center = y_center * h
        width = width * w
        height = height * h

        # Convert to corner format
        x1 = x_center - width / 2.0
        y1 = y_center - height / 2.0
        x2 = x_center + width / 2.0
        y2 = y_center + height / 2.0

        return torch.stack([x1, y1, x2, y2], dim=1)


class TrainingAdapter(ABC):
    """
    Abstract base class for model-specific training logic.

    Different models have different training requirements:
    - torchvision models: Standard PyTorch training with loss_dict
    - YOLO: Custom training loop via Ultralytics
    - DETR: Special loss computation with Hungarian matcher
    """

    @abstractmethod
    def training_step(
        self,
        model: DetectionModel,
        images: List[torch.Tensor],
        targets: List[Dict[str, torch.Tensor]],
        device: torch.device,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scaler: Optional[torch.amp.GradScaler] = None,
        use_amp: bool = False,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Perform one training step.

        Args:
            model: Detection model
            images: List of input images
            targets: List of target dicts
            device: Device to train on (cuda/cpu)
            optimizer: Optimizer for backward pass
            scaler: Gradient scaler for AMP
            use_amp: Whether to use automatic mixed precision

        Returns:
            Tuple of (total_loss, loss_dict) where loss_dict contains individual loss terms
        """
        raise NotImplementedError

    @abstractmethod
    def validation_step(
        self,
        model: DetectionModel,
        images: List[torch.Tensor],
        targets: List[Dict[str, torch.Tensor]],
        device: torch.device,
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Perform validation step (inference with targets available).

        Args:
            model: Detection model
            images: List of input images
            targets: List of target dicts (for metrics computation)
            device: Device to validate on

        Returns:
            List of prediction dicts with keys:
            - 'boxes': Tensor of shape (N, 4)
            - 'labels': Tensor of shape (N,)
            - 'scores': Tensor of shape (N,)
        """
        raise NotImplementedError


class ModelRegistry:
    """
    Registry for detection models with automatic registration.

    Usage:
        @ModelRegistry.register('yolov8n')
        class YOLOv8Nano(DetectionModel):
            ...

        model = ModelRegistry.get('yolov8n', num_classes=12)
    """

    _registry: Dict[str, type] = {}

    @classmethod
    def register(cls, name: str):
        """
        Decorator for registering a model class.

        Args:
            name: Unique model name

        Returns:
            Decorator function
        """

        def decorator(model_class: type) -> type:
            cls._registry[name.lower()] = model_class
            return model_class

        return decorator

    @classmethod
    def get(cls, name: str, **kwargs: Any) -> DetectionModel:
        """
        Get model by name and instantiate with kwargs.

        Args:
            name: Model name (case-insensitive)
            **kwargs: Arguments to pass to model constructor

        Returns:
            Instantiated model

        Raises:
            ValueError: If model name not found
        """
        name_lower = name.lower()
        if name_lower not in cls._registry:
            available = ", ".join(cls._registry.keys())
            raise ValueError(f"Unknown model: {name}. " f"Available models: {available}") from None
        model_class = cls._registry[name_lower]
        return model_class(**kwargs)  # type: ignore[no-any-return]

    @classmethod
    def list_models(cls) -> List[str]:
        """Get list of all registered models."""
        return sorted(cls._registry.keys())

    @classmethod
    def get_model_info(cls, name: str) -> str:
        """Get docstring/info about a model."""
        name_lower = name.lower()
        if name_lower not in cls._registry:
            return f"Model {name} not found"
        model_class = cls._registry[name_lower]
        return model_class.__doc__ or "No documentation available"


# Identity converters for default case
class IdentityFormatConverter(FormatConverter):
    """Converter that assumes already in correct format (no-op)."""

    def to_internal_format(
        self, targets: List[Dict[str, torch.Tensor]]
    ) -> List[Dict[str, torch.Tensor]]:
        """Return targets unchanged."""
        return targets

    def from_internal_format(
        self, targets: List[Dict[str, torch.Tensor]]
    ) -> List[Dict[str, torch.Tensor]]:
        """Return targets unchanged."""
        return targets
