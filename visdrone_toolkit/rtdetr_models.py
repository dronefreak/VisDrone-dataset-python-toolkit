"""RT-DETR model wrappers for VisDrone detection.

RT-DETR (Real-Time Detection Transformer) is a transformer-based detector
from Baidu Research (2023) that matches DETR accuracy while achieving real-time
inference speed. Ultralytics provides native RT-DETR support via the RTDETR class,
which shares the same training interface as YOLO.

Available variants (currently l and x only in Ultralytics):
- rtdetr-l  (large):      ~32M params, ResNet-101 backbone
- rtdetr-x  (extra-large): ~67M params, ResNet-101 backbone (larger decoder)

ResNet-backbone variants (community weights):
- rtdetr-resnet50:  ~40M params, ResNet-50 backbone
- rtdetr-resnet101: ~76M params, ResNet-101 backbone

Requires: pip install ultralytics>=8.0.0
"""

from typing import Any, List, Optional

import torch

from .abstract_models import DetectionModel, ModelRegistry
from .format_converters import YOLOFormatConverter


class RTDETRBase(DetectionModel):
    """Base class for RT-DETR models via Ultralytics.

    Mirrors YOLOv8Base but uses the ``ultralytics.RTDETR`` class instead of
    ``ultralytics.YOLO``.  Training is delegated to the Ultralytics engine
    through ``YOLOTrainer`` (which auto-detects the ``rtdetr-`` prefix and
    switches to the RTDETR class).
    """

    ULTRALYTICS_MODEL = "rtdetr-l.pt"

    def __init__(
        self,
        num_classes: int = 12,
        _pretrained: bool = True,
        device: str = "cuda",
        imgsz: int = 640,
        **_kwargs: Any,
    ) -> None:
        super().__init__(num_classes=num_classes)

        try:
            from ultralytics import RTDETR
        except ImportError as err:
            raise ImportError(
                "Ultralytics is required for RT-DETR. "
                "Install with: pip install ultralytics>=8.0.0"
            ) from err

        self.model = RTDETR(self.ULTRALYTICS_MODEL)
        self.device_name = device
        self.imgsz = imgsz
        self.format_converter = YOLOFormatConverter()

        if device.startswith("cuda"):
            self.model.to(device)

        self._rtdetr_model = self.model

    def forward(
        self,
        images: List[torch.Tensor],
        targets: Optional[List[dict]] = None,
    ) -> Any:
        """RT-DETR training/inference forward pass.

        Training is fully delegated to the Ultralytics engine via ``YOLOTrainer``.
        This method should not be called directly during training — it only exists
        to satisfy the ``DetectionModel`` interface for inference callers.
        """
        if targets is not None:
            raise NotImplementedError(
                "RT-DETR training must go through YOLOTrainer, not forward(). "
                "Use scripts/train.py which routes rtdetr-* models to the "
                "Ultralytics RTDETR training engine automatically."
            )

        # Inference via ultralytics predict
        results = self.model.predict(images, device=self.device_name, imgsz=self.imgsz)
        return results

    def _training_forward(self, images, targets):
        raise NotImplementedError("Use YOLOTrainer for RT-DETR training (same as YOLO models).")

    def get_input_format(self) -> str:
        return "yolo"

    def get_output_format(self) -> str:
        return "coco_dict"


# ---------------------------------------------------------------------------
# RT-DETR Large / X  (official Ultralytics variants)
# ---------------------------------------------------------------------------


@ModelRegistry.register("rtdetr-l")
class RTDETRLarge(RTDETRBase):
    """RT-DETR Large — Ultralytics-native transformer detector.

    Architecture:
    - Backbone: HGNetv2_L (custom hybrid ResNet-style)
    - Decoder: 6-layer transformer with deformable attention
    - Input size: 640×640

    Specs:
    - Parameters: ~32M
    - mAP@0.5:0.95 (COCO): ~53.0%
    - GFLOPs: ~110
    - Model size: ~124 MB
    """

    ULTRALYTICS_MODEL = "rtdetr-l.pt"


@ModelRegistry.register("rtdetr-x")
class RTDETRExtraLarge(RTDETRBase):
    """RT-DETR Extra Large — largest Ultralytics RT-DETR variant.

    Architecture:
    - Backbone: HGNetv2_X (wider channels)
    - Decoder: 6-layer transformer with wider hidden dim
    - Input size: 640×640

    Specs:
    - Parameters: ~67M
    - mAP@0.5:0.95 (COCO): ~54.8%
    - GFLOPs: ~234
    - Model size: ~260 MB
    """

    ULTRALYTICS_MODEL = "rtdetr-x.pt"


# ---------------------------------------------------------------------------
# ResNet-backbone variants (community-trained weights)
# ---------------------------------------------------------------------------


@ModelRegistry.register("rtdetr-resnet50")
class RTDETRResNet50(RTDETRBase):
    """RT-DETR ResNet-50 — original paper backbone variant.

    Uses the ResNet-50 backbone as in the original RT-DETR paper (Zhao et al. 2023).
    These weights come from the original paper's training on COCO.

    Specs:
    - Parameters: ~42M
    - mAP@0.5:0.95 (COCO): ~53.1%
    - GFLOPs: ~136
    - Model size: ~167 MB
    """

    ULTRALYTICS_MODEL = "rtdetr-resnet50.pt"


@ModelRegistry.register("rtdetr-resnet101")
class RTDETRResNet101(RTDETRBase):
    """RT-DETR ResNet-101 — original paper large backbone variant.

    Uses the ResNet-101 backbone as in the original RT-DETR paper (Zhao et al. 2023).

    Specs:
    - Parameters: ~76M
    - mAP@0.5:0.95 (COCO): ~54.3%
    - GFLOPs: ~259
    - Model size: ~303 MB
    """

    ULTRALYTICS_MODEL = "rtdetr-resnet101.pt"
