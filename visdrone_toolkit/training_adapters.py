"""
Training adapters for different detection model types.

Adapters handle model-specific training logic, allowing the main training loop
to remain agnostic to the underlying model implementation.
"""

from typing import Dict, List, Optional, Tuple

import torch
from torch.amp import GradScaler, autocast

from .abstract_models import DetectionModel, TrainingAdapter


class TorchvisionTrainingAdapter(TrainingAdapter):
    """
    Training adapter for torchvision detection models.

    Works with models that follow the torchvision API:
    - Faster R-CNN
    - FCOS
    - RetinaNet
    """

    def training_step(
        self,
        model: DetectionModel,
        images: List[torch.Tensor],
        targets: List[Dict[str, torch.Tensor]],
        device: torch.device,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scaler: Optional[GradScaler] = None,
        use_amp: bool = False,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Perform one training step for torchvision models.

        Args:
            model: Detection model
            images: List of input images
            targets: List of target dicts
            device: Device to train on
            optimizer: Optimizer for backward pass
            scaler: Gradient scaler for AMP
            use_amp: Whether to use automatic mixed precision

        Returns:
            Tuple of (total_loss, loss_dict)
        """
        # Move to device
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        model.train()

        # Forward pass
        if use_amp and scaler is not None:
            with autocast(device_type=device.type):
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            losses.backward()
            if optimizer is not None:
                optimizer.step()

        return losses.item(), loss_dict

    def validation_step(
        self,
        model: DetectionModel,
        images: List[torch.Tensor],
        _targets: List[Dict[str, torch.Tensor]],
        device: torch.device,
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Perform validation step (inference with targets available).

        Args:
            model: Detection model
            images: List of input images
            _targets: List of target dicts (unused, for API compatibility)
            device: Device to validate on

        Returns:
            List of prediction dicts with keys:
            - 'boxes': Tensor of shape (N, 4)
            - 'labels': Tensor of shape (N,)
            - 'scores': Tensor of shape (N,)
        """
        # Move to device
        images = [img.to(device) for img in images]

        model.eval()
        with torch.no_grad():
            predictions = model(images)  # type: ignore[misc]

        return predictions  # type: ignore[no-any-return]


class YOLOTrainingAdapter(TrainingAdapter):
    """Stub adapter for YOLO models — training is NOT handled here.

    YOLO training requires Ultralytics' own engine (TaskAlignedAssigner,
    DFL/box/cls losses, Mosaic augmentation, etc.) and cannot be unified
    with the torchvision training loop at the backward pass level.

    Real YOLO training is delegated to ``YOLOTrainer`` in
    ``visdrone_toolkit.yolo_trainer``, which calls ``ultralytics.YOLO.train()``.

    This adapter only implements ``validation_step`` for inference-based
    evaluation after training.
    """

    def training_step(
        self,
        model: DetectionModel,
        images: List[torch.Tensor],
        targets: List[Dict[str, torch.Tensor]],
        device: torch.device,
        optimizer: Optional[torch.optim.Optimizer] = None,
        _scaler: Optional[GradScaler] = None,
        _use_amp: bool = False,
    ) -> Tuple[float, Dict[str, float]]:
        """Not a real training step — raises to prevent silent no-ops.

        YOLO training must be done via YOLOTrainer, not UnifiedTrainer.
        """
        raise NotImplementedError(
            "YOLO training is not supported through UnifiedTrainer._train_epoch(). "
            "Use YOLOTrainer from visdrone_toolkit.yolo_trainer instead, "
            "or call scripts/train.py which routes YOLO models automatically."
        )

    def validation_step(
        self,
        model: DetectionModel,
        images: List[torch.Tensor],
        _targets: List[Dict[str, torch.Tensor]],
        device: torch.device,
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Perform validation step for YOLO models.

        Args:
            model: YOLO detection model
            images: List of input images
            _targets: List of target dicts (unused)
            device: Device to validate on

        Returns:
            List of prediction dicts in standardized format
        """
        # Move to device
        images = [img.to(device) for img in images]

        model.eval()
        with torch.no_grad():
            predictions = model(images)  # type: ignore[misc]

        # Convert YOLO output to standard format if needed
        if hasattr(model, "_convert_outputs_to_standard"):
            predictions = model._convert_outputs_to_standard(predictions)  # type: ignore[misc]

        return predictions  # type: ignore[no-any-return]


class DETRTrainingAdapter(TrainingAdapter):
    """
    Training adapter for DETR (Detection Transformer) models.

    DETR requires special handling for loss computation with Hungarian matching.
    """

    def __init__(self, criterion=None, matcher=None):
        """
        Initialize DETR adapter.

        Args:
            criterion: DETR criterion for loss computation
            matcher: Hungarian matcher for bipartite matching
        """
        self.criterion = criterion
        self.matcher = matcher

    def training_step(
        self,
        model: DetectionModel,
        images: List[torch.Tensor],
        targets: List[Dict[str, torch.Tensor]],
        device: torch.device,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scaler: Optional[GradScaler] = None,
        use_amp: bool = False,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Perform one training step for DETR models.

        Args:
            model: DETR detection model
            images: List of input images
            targets: List of target dicts with additional DETR-specific fields
            device: Device to train on
            optimizer: Optimizer for backward pass
            scaler: Gradient scaler for AMP
            use_amp: Whether to use automatic mixed precision

        Returns:
            Tuple of (total_loss, loss_dict)
        """
        # Move to device
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        model.train()

        # DETR forward pass with criterion
        if use_amp and scaler is not None:
            with autocast(device_type=device.type):
                outputs = model(images)
                loss_dict = self.criterion(outputs, targets)
                losses = sum(v for v in loss_dict.values() if isinstance(v, torch.Tensor))
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss_dict = self.criterion(outputs, targets)
            losses = sum(v for v in loss_dict.values() if isinstance(v, torch.Tensor))
            losses.backward()
            if optimizer is not None:
                optimizer.step()

        return losses.item(), loss_dict

    def validation_step(
        self,
        model: DetectionModel,
        images: List[torch.Tensor],
        _targets: List[Dict[str, torch.Tensor]],
        device: torch.device,
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Perform validation step for DETR models.

        Args:
            model: DETR detection model
            images: List of input images
            _targets: List of target dicts (unused, for compatibility)
            device: Device to validate on

        Returns:
            List of prediction dicts in standardized format
        """
        # Move to device
        images = [img.to(device) for img in images]

        model.eval()
        with torch.no_grad():
            outputs = model(images)
            # Convert DETR outputs to standard format
            predictions = self._convert_detr_outputs(outputs)

        return predictions

    @staticmethod
    def _convert_detr_outputs(outputs: Dict[str, torch.Tensor]) -> List[Dict[str, torch.Tensor]]:
        """
        Convert DETR model outputs to standard detection format.

        Args:
            outputs: DETR model outputs with 'pred_logits' and 'pred_boxes'

        Returns:
            List of dicts with 'boxes', 'labels', 'scores'
        """
        # This is a placeholder - actual implementation depends on DETR variant
        # For now, convert basic DETR output to standard format
        predictions = []

        pred_logits = outputs.get("pred_logits", None)
        pred_boxes = outputs.get("pred_boxes", None)

        if pred_logits is None or pred_boxes is None:
            return []

        # Apply softmax to logits to get class probabilities
        probabilities = pred_logits.softmax(dim=-1)

        # Get max probability and corresponding class for each query
        scores, labels = probabilities.max(dim=-1)

        # Filter out background predictions (usually last class)
        # Only keep boxes with reasonable confidence scores
        threshold = 0.5
        keep_mask = scores > threshold

        predictions.append(
            {
                "boxes": pred_boxes[keep_mask],
                "labels": labels[keep_mask],
                "scores": scores[keep_mask],
            }
        )

        return predictions
