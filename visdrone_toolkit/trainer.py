"""Unified training interface for all detection models.

Provides a single training loop that works with torchvision, YOLO, DETR, and other
detection models through the TrainingAdapter interface. Handles checkpointing,
metrics computation, device management, and format conversion automatically.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from visdrone_toolkit.abstract_models import DetectionModel, TrainingAdapter
from visdrone_toolkit.training_adapters import (
    DETRTrainingAdapter,
    TorchvisionTrainingAdapter,
    YOLOTrainingAdapter,
)


class UnifiedTrainer:
    """Unified trainer for all detection models.

    Handles training, validation, checkpointing, and metrics computation
    for any model that implements the DetectionModel interface.

    Attributes:
        model: The detection model to train
        device: Device to train on (cuda/cpu)
        adapter: TrainingAdapter for the model's framework
    """

    def __init__(
        self,
        model: DetectionModel,
        device: str | torch.device = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """Initialize trainer.

        Args:
            model: DetectionModel instance to train
            device: Device to train on
        """
        self.model = model
        self.device = torch.device(device) if isinstance(device, str) else device
        self.model = self.model.to(self.device)

        # Auto-select adapter based on model type
        self.adapter = self._select_adapter()

        # Training state
        self.start_epoch: int = 0
        self.best_metric: float = -1.0
        self.training_history: dict[str, list[Any]] = {
            "loss": [],
            "lr": [],
            "val_metrics": [],
        }

    def _select_adapter(self) -> TrainingAdapter:
        """Select appropriate training adapter for the model.

        Returns:
            TrainingAdapter instance for the model's framework
        """
        model_class_name = self.model.__class__.__name__

        if "YOLO" in model_class_name or "yolo" in model_class_name.lower():
            return YOLOTrainingAdapter()
        elif "DETR" in model_class_name or "detr" in model_class_name.lower():
            return DETRTrainingAdapter()
        else:
            return TorchvisionTrainingAdapter()

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        epochs: int = 50,
        optimizer: torch.optim.Optimizer | None = None,
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
        use_amp: bool = False,
        accumulation_steps: int = 1,
        output_dir: str | Path = "outputs",
        save_every: int = 10,
        val_every: int = 5,
    ) -> dict[str, Any]:
        """Train the model.

        Args:
            train_loader: Training DataLoader
            val_loader: Validation DataLoader (optional)
            epochs: Number of epochs to train
            optimizer: Optimizer (default: SGD with lr=0.005, momentum=0.9)
            lr_scheduler: Learning rate scheduler (optional)
            use_amp: Use automatic mixed precision
            accumulation_steps: Gradient accumulation steps
            output_dir: Directory to save checkpoints
            save_every: Save checkpoint every N epochs
            val_every: Validate every N epochs

        Returns:
            Dictionary with training history and final metrics
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create optimizer if not provided
        if optimizer is None:
            optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=0.005,
                momentum=0.9,
                weight_decay=0.0005,
            )

        scaler = GradScaler(enabled=use_amp)

        # Training loop
        for epoch in range(self.start_epoch, epochs):
            # Train step
            epoch_loss = self._train_epoch(
                train_loader,
                optimizer,
                scaler,
                use_amp,
                accumulation_steps,
            )
            self.training_history["loss"].append(epoch_loss)

            # Learning rate
            if lr_scheduler is not None:
                current_lr = optimizer.param_groups[0]["lr"]
                self.training_history["lr"].append(current_lr)
                lr_scheduler.step()

            # Validation step
            if val_loader is not None and (epoch + 1) % val_every == 0:
                val_metrics = self._validate(val_loader)
                self.training_history["val_metrics"].append(val_metrics)

                # Save best model
                if "f1" in val_metrics and val_metrics["f1"] > self.best_metric:
                    self.best_metric = val_metrics["f1"]
                    self._save_checkpoint(output_dir / "best_model.pt", optimizer)

            # Save periodic checkpoint
            if (epoch + 1) % save_every == 0:
                self._save_checkpoint(output_dir / f"checkpoint_epoch_{epoch + 1}.pt", optimizer)

            # Log progress
            log_msg = f"Epoch [{epoch + 1}/{epochs}] Loss: {epoch_loss:.4f}"
            if self.training_history["lr"]:
                log_msg += f" LR: {self.training_history['lr'][-1]:.6f}"
            if self.training_history["val_metrics"]:
                val_m = self.training_history["val_metrics"][-1]
                if isinstance(val_m, dict):
                    log_msg += f" F1: {val_m.get('f1', 0):.4f}"
            print(log_msg)

        # Save final checkpoint
        self._save_checkpoint(output_dir / "final_model.pt", optimizer)

        return {
            "history": self.training_history,
            "best_metric": self.best_metric,
            "final_epoch": epochs,
        }

    def _train_epoch(
        self,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scaler: GradScaler,
        use_amp: bool,
        accumulation_steps: int,
    ) -> float:
        """Train for one epoch.

        Args:
            train_loader: Training DataLoader
            optimizer: Optimizer
            scaler: GradScaler for AMP
            use_amp: Use automatic mixed precision
            accumulation_steps: Gradient accumulation steps

        Returns:
            Average loss for the epoch
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch_idx, (images, targets) in enumerate(train_loader):
            images = [img.to(self.device) for img in images]
            targets = [
                {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in t.items()}
                for t in targets
            ]

            # Forward pass with optional AMP
            with autocast(enabled=use_amp, device_type=self.device.type):
                loss_output = self.adapter.training_step(
                    self.model, images, targets, self.device, optimizer, scaler, use_amp
                )

            # Unpack loss output (could be float or tuple)
            if isinstance(loss_output, tuple):
                loss_value, _ = loss_output  # tuple[float, dict[str, float]]
            else:
                loss_value = loss_output if isinstance(loss_output, float) else loss_output.item()

            # Convert to tensor if needed
            loss_tensor = (
                torch.tensor(loss_value, device=self.device)
                if not isinstance(loss_output, torch.Tensor)
                else loss_output
                if isinstance(loss_output, torch.Tensor)
                else torch.tensor(loss_value, device=self.device)
            )

            # Backward pass with accumulation
            loss_tensor = loss_tensor / accumulation_steps
            scaler.scale(loss_tensor).backward()

            # Update weights
            if (batch_idx + 1) % accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            total_loss += loss_value * accumulation_steps
            num_batches += 1

        return total_loss / num_batches if num_batches > 0 else 0.0

    def _validate(self, val_loader: DataLoader) -> dict[str, Any]:
        """Validate the model.

        Args:
            val_loader: Validation DataLoader

        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()
        predictions = []
        targets = []

        with torch.no_grad():
            for images, target_list in val_loader:
                images = [img.to(self.device) for img in images]

                # Get predictions
                preds = self.adapter.validation_step(self.model, images, target_list, self.device)
                if isinstance(preds, list):
                    predictions.extend(preds)
                else:
                    predictions.append(preds)

                targets.extend(target_list)

        # Compute metrics
        metrics = self._compute_metrics(predictions, targets)
        return metrics

    def _compute_metrics(
        self, predictions: list[dict[str, Any]], targets: list[dict[str, Any]]
    ) -> dict[str, float]:
        """Compute validation metrics.

        Args:
            predictions: List of prediction dicts with 'boxes', 'labels', 'scores'
            targets: List of target dicts with 'boxes', 'labels'

        Returns:
            Dictionary with computed metrics
        """
        total_tp = 0
        total_fp = 0
        total_gt = 0
        iou_threshold = 0.5

        for pred, target in zip(predictions, targets):
            if isinstance(pred, dict):
                pred_boxes = pred.get("boxes", torch.tensor([]))
                pred_labels = pred.get("labels", torch.tensor([]))
                _ = pred.get("scores", torch.ones(len(pred_boxes)))
            else:
                continue

            if isinstance(target, dict):
                gt_boxes = target.get("boxes", torch.tensor([]))
                gt_labels = target.get("labels", torch.tensor([]))
            else:
                continue

            total_gt += len(gt_boxes)

            if len(pred_boxes) == 0:
                continue

            if len(gt_boxes) == 0:
                total_fp += len(pred_boxes)
                continue

            # Compute IoU matrix
            ious = self._box_iou(pred_boxes, gt_boxes)

            # Match predictions to ground truth
            matched_gt = set()
            for i in range(len(pred_boxes)):
                best_iou = 0
                best_gt_idx = -1

                for j in range(len(gt_boxes)):
                    if j in matched_gt:
                        continue
                    if pred_labels[i] != gt_labels[j]:
                        continue
                    if ious[i, j] > best_iou:
                        best_iou = ious[i, j]
                        best_gt_idx = j

                if best_iou >= iou_threshold and best_gt_idx != -1:
                    total_tp += 1
                    matched_gt.add(best_gt_idx)
                else:
                    total_fp += 1

        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        recall = total_tp / total_gt if total_gt > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    @staticmethod
    def _box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
        """Compute IoU between two sets of boxes.

        Args:
            boxes1: Tensor of shape [N, 4] in format [x1, y1, x2, y2]
            boxes2: Tensor of shape [M, 4] in format [x1, y1, x2, y2]

        Returns:
            IoU matrix of shape [N, M]
        """
        if boxes1.dtype == torch.float64:
            boxes1 = boxes1.float()
        if boxes2.dtype == torch.float64:
            boxes2 = boxes2.float()

        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

        lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
        rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
        wh = (rb - lt).clamp(min=0)
        inter = wh[:, :, 0] * wh[:, :, 1]

        union = area1[:, None] + area2 - inter
        iou = inter / union
        return iou

    def _save_checkpoint(self, path: Path | str, optimizer: torch.optim.Optimizer) -> None:
        """Save model checkpoint.

        Args:
            path: Path to save checkpoint
            optimizer: Optimizer to save state
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "model_state": self.model.to("cpu").state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": self.start_epoch,
            "history": self.training_history,
            "best_metric": self.best_metric,
        }

        torch.save(checkpoint, path)
        self.model = self.model.to(self.device)

    def load_checkpoint(
        self, path: Path | str, optimizer: torch.optim.Optimizer | None = None
    ) -> None:
        """Load model checkpoint.

        Args:
            path: Path to checkpoint
            optimizer: Optimizer to load state into (optional)
        """
        path = Path(path)
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state"])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint["optimizer_state"])

        self.start_epoch = checkpoint.get("epoch", 0)
        self.training_history = checkpoint.get("history", {"loss": [], "lr": [], "val_metrics": []})
        self.best_metric = checkpoint.get("best_metric", -1.0)
