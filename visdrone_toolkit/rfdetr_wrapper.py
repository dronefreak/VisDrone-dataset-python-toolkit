"""
RF-DETR model wrapper for VisDrone toolkit.

RF-DETR (Rotary-position-enhanced Feature-augmented Detection TRansformer) is
a real-time object detection transformer from Roboflow that achieves state-of-the-art
accuracy while maintaining high inference speed.

This module provides:
- RFDETRWrapper: torchvision-compatible inference wrapper for RF-DETR
- get_rfdetr_model: factory function used by get_model() in utils.py

For training RF-DETR, use scripts/train_rfdetr.py which leverages the native
rfdetr training loop (DETR-style set-based loss with Hungarian matching).

Install:
    pip install rfdetr

References:
    - RF-DETR paper: https://arxiv.org/abs/2502.11149
    - RF-DETR GitHub: https://github.com/roboflow/rf-detr
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


class RFDETRWrapper(nn.Module):
    """
    Wraps an RF-DETR model to provide a torchvision-compatible inference API.

    Converts between the torchvision detection API format:
        eval: model(images) -> [{'boxes': ..., 'labels': ..., 'scores': ...}, ...]

    and RF-DETR's supervision-based prediction output.

    Note:
        This wrapper is for inference only. For training, use
        scripts/train_rfdetr.py which uses the native rfdetr training loop
        with DETR-style Hungarian matching losses.

    Args:
        rfdetr_obj: An RFDETRBase or RFDETRLarge instance from the rfdetr package.
        score_threshold: Default confidence threshold for filtering predictions.
    """

    def __init__(self, rfdetr_obj: Any, score_threshold: float = 0.05) -> None:
        super().__init__()
        # Store the rfdetr high-level object (not registered as submodule)
        self._rfdetr_obj = rfdetr_obj
        self.score_threshold = score_threshold

        # Try to register the underlying PyTorch model so that parameters()
        # and state_dict() work correctly with the existing checkpoint utilities.
        if hasattr(rfdetr_obj, "model") and isinstance(rfdetr_obj.model, nn.Module):
            self.inner_model = rfdetr_obj.model
        else:
            self.inner_model = None

    # ------------------------------------------------------------------
    # torchvision-compatible forward
    # ------------------------------------------------------------------

    def forward(
        self,
        images: list[torch.Tensor],
        targets: list[dict] | None = None,
    ) -> list[dict[str, torch.Tensor]]:
        """
        Run inference on a batch of images.

        Args:
            images: List of float tensors [C, H, W] in [0, 1] range.
            targets: Must be None. RF-DETR training is handled by train_rfdetr.py.

        Returns:
            List of dicts, one per image, each containing:
                - 'boxes': FloatTensor[N, 4] in xyxy format
                - 'labels': Int64Tensor[N] — 1-indexed VisDrone class IDs
                - 'scores': FloatTensor[N]
        """
        if targets is not None:
            raise RuntimeError(
                "RF-DETR training is not supported through this wrapper. "
                "Please use scripts/train_rfdetr.py for native RF-DETR training, "
                "which uses DETR-style set-based losses with Hungarian matching."
            )

        from torchvision.transforms.functional import to_pil_image

        results = []
        for img_tensor in images:
            img_pil = to_pil_image(img_tensor.cpu())
            detections = self._rfdetr_obj.predict(img_pil, threshold=self.score_threshold)

            n = len(detections)
            if n > 0:
                boxes = torch.from_numpy(detections.xyxy).float()
                scores = torch.from_numpy(detections.confidence).float()
                # rfdetr class_id is 0-indexed; add 1 for VisDrone 1-indexed labels
                labels = torch.from_numpy(detections.class_id).long() + 1
            else:
                boxes = torch.zeros((0, 4), dtype=torch.float32)
                scores = torch.zeros(0, dtype=torch.float32)
                labels = torch.zeros(0, dtype=torch.long)

            results.append({"boxes": boxes, "labels": labels, "scores": scores})

        return results

    # ------------------------------------------------------------------
    # Pass-through helpers so existing training utilities work
    # ------------------------------------------------------------------

    def train(self, mode: bool = True) -> "RFDETRWrapper":  # type: ignore[override]
        """Keep wrapper in eval mode — RF-DETR manages its own training state."""
        # Do NOT propagate to inner_model; rfdetr controls that internally.
        return self

    def eval(self) -> "RFDETRWrapper":  # type: ignore[override]
        if self.inner_model is not None:
            self.inner_model.eval()
        return self

    def to(self, *args: Any, **kwargs: Any) -> "RFDETRWrapper":
        if self.inner_model is not None:
            self.inner_model.to(*args, **kwargs)
        return self

    def cuda(self, device: int | None = None) -> "RFDETRWrapper":
        if self.inner_model is not None:
            self.inner_model.cuda(device)
        return self

    def cpu(self) -> "RFDETRWrapper":
        if self.inner_model is not None:
            self.inner_model.cpu()
        return self


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def get_rfdetr_model(
    model_name: str,
    num_classes: int,
    pretrained: bool = True,
    **kwargs: Any,
) -> RFDETRWrapper:
    """
    Create an RF-DETR model wrapped for torchvision-compatible inference.

    Args:
        model_name: 'rfdetr_base' or 'rfdetr_large'.
        num_classes: Number of detection classes (12 for VisDrone).
        pretrained: Load RF-DETR pretrained COCO weights.
        **kwargs: Extra keyword arguments forwarded to the rfdetr constructor.

    Returns:
        RFDETRWrapper instance.

    Raises:
        ImportError: If the rfdetr package is not installed.
        ValueError: If model_name is not recognised.
    """
    try:
        from rfdetr import RFDETRBase, RFDETRLarge
    except ImportError as exc:
        raise ImportError(
            "RF-DETR requires the 'rfdetr' package. "
            "Install it with:\n"
            "    pip install rfdetr\n"
            "or:\n"
            "    pip install -r requirements.txt"
        ) from exc

    model_name = model_name.lower()
    if model_name == "rfdetr_base":
        rfdetr_cls = RFDETRBase
    elif model_name == "rfdetr_large":
        rfdetr_cls = RFDETRLarge
    else:
        raise ValueError(
            f"Unknown RF-DETR variant: '{model_name}'. "
            "Choose from: 'rfdetr_base', 'rfdetr_large'."
        )

    # RF-DETR uses 0-indexed classes internally; num_classes here matches VisDrone
    rfdetr_obj = rfdetr_cls(
        num_classes=num_classes,
        pretrain_weights="rf-detr-base.pth" if (pretrained and model_name == "rfdetr_base") else (
            "rf-detr-large.pth" if (pretrained and model_name == "rfdetr_large") else None
        ),
        **kwargs,
    )

    return RFDETRWrapper(rfdetr_obj, score_threshold=0.05)
