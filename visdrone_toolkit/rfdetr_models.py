"""RF-DETR model wrappers for VisDrone detection.

RF-DETR (Roboflow Detection Transformer) is a DINOv2-based detector from
Roboflow (2025) that achieves strong accuracy with efficient inference.
Training uses the rfdetr package's PyTorch Lightning engine.

Available detection variants:
- rfdetr-nano:   ~7M params  — fastest, mobile-class
- rfdetr-small:  ~14M params — good speed/accuracy tradeoff
- rfdetr-medium: ~32M params — balanced
- rfdetr-large:  ~42M params — highest accuracy

VisDrone-specific notes:
- Trained on 10 classes (``others`` category filtered for consistency with YOLO pipeline)
- Uses pre-prepared Roboflow COCO data at ``data/VisDrone2019-DET-RF-DETR/``
- Inference returns ``supervision.Detections`` (auto-converted for compatibility)

Requires: pip install rfdetr "rfdetr[train]" supervision
"""

from typing import Any, List, Optional

import torch

from .abstract_models import DetectionModel, ModelRegistry


class RFDETRBase(DetectionModel):
    """Base class for RF-DETR models.

    Wraps the rfdetr package and adapts it to the DetectionModel interface.
    Training is fully delegated to ``RFDETRTrainer`` (rfdetr PyTorch Lightning stack).
    """

    # rfdetr class name — overridden in subclasses
    RFDETR_CLASS = "RFDETRLarge"

    def __init__(
        self,
        num_classes: int = 10,
        _pretrained: bool = True,
        device: str = "cuda",
        **_kwargs: Any,
    ) -> None:
        super().__init__(num_classes=num_classes)

        try:
            import rfdetr as _rfdetr_pkg
        except ImportError as err:
            raise ImportError("rfdetr is required. Install with: pip install rfdetr") from err

        cls = getattr(_rfdetr_pkg, self.RFDETR_CLASS)
        self.model = cls(num_classes=num_classes)
        self.device_name = device
        self._rfdetr_pkg = _rfdetr_pkg

    def forward(
        self,
        images: List[torch.Tensor],
        targets: Optional[List[dict]] = None,
    ) -> Any:
        """RF-DETR forward pass.

        Training must go through ``RFDETRTrainer`` (PyTorch Lightning engine).
        This method exists only for inference and DetectionModel interface compliance.

        Returns:
            List of dicts with keys ``boxes``, ``scores``, ``labels`` (xyxy, float, int).
        """
        if targets is not None:
            raise NotImplementedError(
                "RF-DETR training must go through RFDETRTrainer. "
                "Use scripts/train.py which routes rfdetr-* models to the rfdetr engine."
            )

        import numpy as np
        from PIL import Image as PILImage

        results = []
        for img_tensor in images:
            # Convert CHW float tensor [0,1] → HWC uint8 PIL image
            img_np = (img_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            pil_img = PILImage.fromarray(img_np)

            detections = self.model.predict(pil_img, threshold=0.5)

            results.append(
                {
                    "boxes": torch.from_numpy(detections.xyxy).float()
                    if len(detections) > 0
                    else torch.zeros((0, 4)),
                    "scores": torch.from_numpy(detections.confidence).float()
                    if len(detections) > 0
                    else torch.zeros(0),
                    "labels": torch.from_numpy(detections.class_id).long()
                    if len(detections) > 0
                    else torch.zeros(0, dtype=torch.long),
                }
            )

        return results

    def _training_forward(self, images, targets):
        raise NotImplementedError("Use RFDETRTrainer for RF-DETR training.")

    def get_input_format(self) -> str:
        return "pil"

    def get_output_format(self) -> str:
        return "coco_dict"


# ---------------------------------------------------------------------------
# RF-DETR variants
# ---------------------------------------------------------------------------


@ModelRegistry.register("rfdetr-nano")
class RFDETRNano(RFDETRBase):
    """RF-DETR Nano — smallest RF-DETR variant.

    Architecture:
    - Backbone: DINOv2 ViT-S (small)
    - Decoder: lightweight transformer decoder

    Specs:
    - Parameters: ~7M
    - Speed: fastest inference
    - Best for: edge deployment, real-time constraints

    VisDrone notes:
    - Trained on 10 classes (pedestrian, people, bicycle, car, van, truck,
      tricycle, awning-tricycle, bus, motor)
    """

    RFDETR_CLASS = "RFDETRNano"


@ModelRegistry.register("rfdetr-small")
class RFDETRSmall(RFDETRBase):
    """RF-DETR Small — lightweight RF-DETR variant.

    Specs:
    - Parameters: ~14M
    - Good speed/accuracy tradeoff
    """

    RFDETR_CLASS = "RFDETRSmall"


@ModelRegistry.register("rfdetr-medium")
class RFDETRMedium(RFDETRBase):
    """RF-DETR Medium — balanced RF-DETR variant.

    Architecture:
    - Backbone: DINOv2 ViT-B (base)

    Specs:
    - Parameters: ~32M
    - Balanced speed and accuracy
    """

    RFDETR_CLASS = "RFDETRMedium"


@ModelRegistry.register("rfdetr-large")
class RFDETRLarge(RFDETRBase):
    """RF-DETR Large — highest accuracy RF-DETR variant.

    Architecture:
    - Backbone: DINOv2 ViT-L (large)

    Specs:
    - Parameters: ~42M
    - mAP@0.5:0.95 (COCO): ~55.3%
    - Best accuracy, higher VRAM requirement

    VisDrone notes:
    - Default choice for production-quality aerial detection
    """

    RFDETR_CLASS = "RFDETRLarge"
