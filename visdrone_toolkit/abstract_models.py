"""
Abstract base classes and interfaces for detection models.

This module defines the interfaces that all detection models must implement,
enabling seamless integration of different architectures (torchvision, YOLO, DETR, RT-DETR, etc.)
into a unified training and inference pipeline.

Key Components:
    - DetectionModel: Base class for all detection models
    - FormatConverter: Convert between different box formats (COCO, YOLO, RT-DETR)
    - TrainingAdapter: Model-specific training logic
    - ModelRegistry: Auto-registration system for models
    - get_model(): Factory function to instantiate models by name

Example:
    >>> from visdrone_toolkit.abstract_models import get_model
    >>> model = get_model("yolov8n", num_classes=12)
    >>> predictions = model(images)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable

import torch
import torch.nn as nn


class DetectionModel(nn.Module, ABC):
    """
    Abstract base class for all detection models.

    All detection models must inherit from this class and implement the required methods.
    This ensures a consistent interface across different frameworks (torchvision, YOLO, DETR, RT-DETR).

    Attributes:
        num_classes (int): Number of detection classes
        model: Underlying framework-specific model (set by subclasses)

    Example:
        >>> class MyModel(DetectionModel):
        ...     def forward(self, images, targets=None):
        ...         return self.model(images, targets)
        ...     def get_input_format(self):
        ...         return "coco"
        ...     def get_output_format(self):
        ...         return "coco_dict"
    """

    def __init__(self, num_classes: int = 12, **_kwargs: Any) -> None:
        """
        Initialize detection model.

        Args:
            num_classes: Number of detection classes (default: 12 for VisDrone)
            **_kwargs: Model-specific arguments (unused in base class)
        """
        super().__init__()
        self.num_classes = num_classes
        self.model: nn.Module | None = None  # To be set by subclasses

    @abstractmethod
    def forward(
        self,
        images: list[torch.Tensor],
        targets: list[dict[str, torch.Tensor]] | None = None,
    ) -> Any:
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
            'rtdetr': [x1, y1, x2, y2] format (normalized 0-1)
            'detr': [x1, y1, x2, y2] format (normalized 0-1)
        """
        raise NotImplementedError

    @abstractmethod
    def get_output_format(self) -> str:
        """
        Get the output format produced by this model.

        Returns:
            'coco_dict': Standard dict with boxes, labels, scores
            'yolo_results': Ultralytics Results object
            'rtdetr_results': HuggingFace object with boxes, labels, scores
            'detr_results': HuggingFace object with boxes, labels, scores
        """
        raise NotImplementedError

    def get_trainable_parameters(self) -> int:
        """Get number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def freeze_backbone(self, _num_layers: int | None = None) -> None:
        """
        Freeze backbone layers for fine-tuning.

        Args:
            _num_layers: Number of layers from end to freeze.
                       If None, freeze entire backbone.
        """
        if self.model is not None:
            for param in self.model.parameters():
                param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        """Unfreeze all backbone layers."""
        if self.model is not None:
            for param in self.model.parameters():
                param.requires_grad = True


class FormatConverter(ABC):
    """
    Abstract base class for converting between different box formats.

    Different models expect different box representations:
    - COCO format: [x1, y1, x2, y2] (absolute coordinates)
    - YOLO format: [x_center, y_center, w, h] (normalized 0-1)
    - DETR format: [x1, y1, x2, y2] with additional metadata
    - RT-DETR format: [x1, y1, x2, y2] (normalized 0-1)

    Example:
        >>> converter = YOLOFormatConverter()
        >>> yolo_boxes = converter.coco_to_yolo(coco_boxes, image_size=(640, 640))
    """

    @abstractmethod
    def to_internal_format(
        self, targets: list[dict[str, torch.Tensor]]
    ) -> list[dict[str, torch.Tensor]]:
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
        self, targets: list[dict[str, torch.Tensor]]
    ) -> list[dict[str, torch.Tensor]]:
        """
        Convert from internal COCO format to model-specific format.

        Args:
            targets: List of target dicts in internal format

        Returns:
            List of target dicts in model-specific format
        """
        raise NotImplementedError

    @staticmethod
    def coco_to_yolo(boxes: torch.Tensor, image_size: tuple[int, int]) -> torch.Tensor:
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

        x_center = (x1 + x2) / 2.0
        y_center = (y1 + y2) / 2.0
        width = x2 - x1
        height = y2 - y1

        x_center = x_center / w
        y_center = y_center / h
        width = width / w
        height = height / h

        return torch.stack([x_center, y_center, width, height], dim=1)

    @staticmethod
    def yolo_to_coco(boxes: torch.Tensor, image_size: tuple[int, int]) -> torch.Tensor:
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

        x_center = x_center * w
        y_center = y_center * h
        width = width * w
        height = height * h

        x1 = x_center - width / 2.0
        y1 = y_center - height / 2.0
        x2 = x_center + width / 2.0
        y2 = y_center + height / 2.0

        return torch.stack([x1, y1, x2, y2], dim=1)

    @staticmethod
    def coco_to_rtdetr(boxes: torch.Tensor, image_size: tuple[int, int]) -> torch.Tensor:
        """
        Convert COCO format to RT-DETR format (normalized [x1, y1, x2, y2]).

        Args:
            boxes: Tensor of shape (N, 4) in [x1, y1, x2, y2] absolute format
            image_size: (height, width) of image for normalization

        Returns:
            Tensor of shape (N, 4) in [x1, y1, x2, y2] normalized format
        """
        if len(boxes) == 0:
            return boxes

        h, w = image_size
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]

        return torch.stack([x1 / w, y1 / h, x2 / w, y2 / h], dim=1)

    @staticmethod
    def rtdetr_to_coco(boxes: torch.Tensor, image_size: tuple[int, int]) -> torch.Tensor:
        """
        Convert RT-DETR format to COCO format.

        Args:
            boxes: Tensor of shape (N, 4) in [x1, y1, x2, y2] normalized format
            image_size: (height, width) of image for denormalization

        Returns:
            Tensor of shape (N, 4) in [x1, y1, x2, y2] absolute format
        """
        if len(boxes) == 0:
            return boxes

        h, w = image_size
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]

        return torch.stack([x1 * w, y1 * h, x2 * w, y2 * h], dim=1)


class TrainingAdapter(ABC):
    """
    Abstract base class for model-specific training logic.

    Different models have different training requirements:
    - torchvision models: Standard PyTorch training with loss_dict
    - YOLO: Custom training loop via Ultralytics
    - DETR: Special loss computation with Hungarian matcher
    - RT-DETR: HuggingFace training with standard loss computation

    Example:
        >>> adapter = YOLOTrainingAdapter()
        >>> loss, loss_dict = adapter.training_step(model, images, targets, device)
    """

    @abstractmethod
    def training_step(
        self,
        model: DetectionModel,
        images: list[torch.Tensor],
        targets: list[dict[str, torch.Tensor]],
        device: torch.device,
        optimizer: torch.optim.Optimizer | None = None,
        scaler: torch.amp.GradScaler | None = None,
        use_amp: bool = False,
    ) -> tuple[float, dict[str, float]]:
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
        images: list[torch.Tensor],
        targets: list[dict[str, torch.Tensor]],
        device: torch.device,
    ) -> list[dict[str, torch.Tensor]]:
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

    This allows models to be registered and instantiated by name,
    making it easy to add new models without modifying existing code.

    Usage:
        @ModelRegistry.register('yolov8n')
        class YOLOv8Nano(DetectionModel):
            ...

        model = ModelRegistry.get('yolov8n', num_classes=12)
        models = ModelRegistry.list_models()

    Attributes:
        _registry: Dictionary mapping model names to model classes
    """

    _registry: dict[str, Callable[..., DetectionModel]] = {}

    @classmethod
    def register(cls, name: str):
        """
        Decorator for registering a model class.

        Args:
            name: Unique model name (case-insensitive)

        Returns:
            Decorator function

        Example:
            >>> @ModelRegistry.register('my_model')
            ... class MyModel(DetectionModel):
            ...     pass
        """

        def decorator(
            model_factory: Callable[..., DetectionModel],
        ) -> Callable[..., DetectionModel]:
            cls._registry[name.lower()] = model_factory
            return model_factory

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

        Example:
            >>> model = ModelRegistry.get('yolov8n', num_classes=12)
        """
        name_lower = name.lower()
        if name_lower not in cls._registry:
            available = ", ".join(cls._registry.keys())
            raise ValueError(f"Unknown model: {name}. Available models: {available}")
        model_factory = cls._registry[name_lower]
        return model_factory(**kwargs)

    @classmethod
    def list_models(cls) -> list[str]:
        """Get list of all registered model names."""
        return sorted(cls._registry.keys())

    @classmethod
    def get_model_info(cls, name: str) -> str:
        """
        Get docstring/info about a model.

        Args:
            name: Model name

        Returns:
            Model docstring or "No documentation available"
        """
        name_lower = name.lower()
        if name_lower not in cls._registry:
            return f"Model {name} not found"
        model_factory = cls._registry[name_lower]
        return model_factory.__doc__ or "No documentation available"

    @classmethod
    def clear(cls) -> None:
        """Clear all registered models (useful for testing)."""
        cls._registry.clear()


class IdentityFormatConverter(FormatConverter):
    """Converter that assumes already in correct format (no-op)."""

    def to_internal_format(
        self, targets: list[dict[str, torch.Tensor]]
    ) -> list[dict[str, torch.Tensor]]:
        """Return targets unchanged."""
        return targets

    def from_internal_format(
        self, targets: list[dict[str, torch.Tensor]]
    ) -> list[dict[str, torch.Tensor]]:
        """Return targets unchanged."""
        return targets


# ============================================================================
# Factory Function
# ============================================================================


def get_model(
    model_name: str,
    num_classes: int = 12,
    pretrained: bool = True,
    **kwargs: Any,
) -> DetectionModel:
    """
    Get a model by name with automatic framework detection.

    This function first tries to use ModelRegistry, then falls back to
    the legacy get_model function for backward compatibility.

    Args:
        model_name: Name of the model to load (case-insensitive)
        num_classes: Number of output classes
        pretrained: Whether to load pretrained weights
        **kwargs: Additional model-specific arguments

    Returns:
        Instantiated model (DetectionModel-compatible)

    Raises:
        ValueError: If model_name is unknown
        ImportError: If a required dependency is missing

    Example:
        >>> model = get_model("yolov8n", num_classes=12)
        >>> model = get_model("fasterrcnn_resnet50", num_classes=12, pretrained=True)
    """
    # Try ModelRegistry first
    if model_name.lower() in ModelRegistry._registry:
        return ModelRegistry.get(
            model_name,
            num_classes=num_classes,
            pretrained=pretrained,
            **kwargs,
        )

    # Fall back to legacy get_model for backward compatibility
    try:
        from visdrone_toolkit.utils import get_model as legacy_get_model

        result: DetectionModel | None = legacy_get_model(
            model_name=model_name,
            num_classes=num_classes,
            pretrained=pretrained,
            **kwargs,
        )
        if result is None:
            raise ValueError(f"Model {model_name} returned None from legacy get_model")
        return result
    except (ImportError, ValueError) as e:
        available_models = ModelRegistry.list_models()
        if available_models:
            raise ValueError(
                f"Unknown model: {model_name}. Available models: {', '.join(available_models)}"
            ) from e
        raise


def list_available_models() -> list[str]:
    """List all available detection models."""
    return ModelRegistry.list_models()
