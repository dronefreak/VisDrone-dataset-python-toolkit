"""Torchvision model wrappers for unified interface."""

from __future__ import annotations

from typing import Any

import torch
from torchvision.models.detection import (
    FasterRCNN_MobileNet_V3_Large_FPN_Weights,
    FasterRCNN_ResNet50_FPN_Weights,
    FCOS_ResNet50_FPN_Weights,
    RetinaNet_ResNet50_FPN_V2_Weights,
    fasterrcnn_mobilenet_v3_large_fpn,
    fasterrcnn_resnet50_fpn,
    fcos_resnet50_fpn,
    retinanet_resnet50_fpn_v2,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.fcos import FCOSClassificationHead
from torchvision.models.detection.retinanet import RetinaNetClassificationHead

from visdrone_toolkit.abstract_models import DetectionModel, ModelRegistry


class FasterRCNNWrapper(DetectionModel):
    """FasterRCNN wrapper for unified interface."""

    def __init__(self, backbone: str = "resnet50", num_classes: int = 12, pretrained: bool = True):
        """Initialize FasterRCNN wrapper."""
        super().__init__(num_classes=num_classes)

        if backbone == "mobilenet":
            weights = FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT if pretrained else None
            model = fasterrcnn_mobilenet_v3_large_fpn(weights=weights)
        else:
            weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None
            model = fasterrcnn_resnet50_fpn(weights=weights)

        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        object.__setattr__(self, "_model", model)
        self.num_classes = num_classes

    def forward(self, images: list[torch.Tensor], targets: list[dict[str, Any]] | None = None):
        return self._model(images, targets)

    def get_input_format(self) -> str:
        return "coco"

    def get_output_format(self) -> str:
        return "coco_dict"

    def to(self, device):
        self._model.to(device)
        return self

    def train(self, mode: bool = True):
        self._model.train(mode)
        return self

    def eval(self):
        self._model.eval()
        return self

    def parameters(self):
        return self._model.parameters()

    def state_dict(self):
        return self._model.state_dict()

    def load_state_dict(self, state_dict, strict: bool = True):
        return self._model.load_state_dict(state_dict, strict=strict)

    @property
    def device(self):
        return next(self._model.parameters()).device

    def __getattr__(self, name: str):
        if name == "training":
            try:
                model = object.__getattribute__(self, "_model")
                return model.training
            except AttributeError:
                return False
        try:
            model = object.__getattribute__(self, "_model")
            return getattr(model, name)
        except AttributeError:
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{name}'"
            ) from None


class FCOSWrapper(DetectionModel):
    """FCOS wrapper for unified interface."""

    def __init__(self, num_classes: int = 12, pretrained: bool = True):
        super().__init__(num_classes=num_classes)

        weights = FCOS_ResNet50_FPN_Weights.DEFAULT if pretrained else None
        model = fcos_resnet50_fpn(weights=weights)

        num_anchors = model.head.classification_head.num_anchors
        model.head.classification_head = FCOSClassificationHead(
            in_channels=model.backbone.out_channels,
            num_anchors=num_anchors,
            num_classes=num_classes,
        )

        object.__setattr__(self, "_model", model)
        self.num_classes = num_classes

    def forward(self, images: list[torch.Tensor], targets: list[dict[str, Any]] | None = None):
        return self._model(images, targets)

    def get_input_format(self) -> str:
        return "coco"

    def get_output_format(self) -> str:
        return "coco_dict"

    def to(self, device):
        self._model.to(device)
        return self

    def train(self, mode: bool = True):
        self._model.train(mode)
        return self

    def eval(self):
        self._model.eval()
        return self

    def parameters(self):
        return self._model.parameters()

    def state_dict(self):
        return self._model.state_dict()

    def load_state_dict(self, state_dict, strict: bool = True):
        return self._model.load_state_dict(state_dict, strict=strict)

    @property
    def device(self):
        return next(self._model.parameters()).device

    def __getattr__(self, name: str):
        if name == "training":
            try:
                model = object.__getattribute__(self, "_model")
                return model.training
            except AttributeError:
                return False
        try:
            model = object.__getattribute__(self, "_model")
            return getattr(model, name)
        except AttributeError:
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{name}'"
            ) from None


class RetinaNetWrapper(DetectionModel):
    """RetinaNet wrapper for unified interface."""

    def __init__(self, num_classes: int = 12, pretrained: bool = True):
        super().__init__(num_classes=num_classes)

        weights = RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT if pretrained else None
        model = retinanet_resnet50_fpn_v2(weights=weights)

        num_anchors = model.head.classification_head.num_anchors
        model.head.classification_head = RetinaNetClassificationHead(
            in_channels=model.backbone.out_channels,
            num_anchors=num_anchors,
            num_classes=num_classes,
        )

        object.__setattr__(self, "_model", model)
        self.num_classes = num_classes

    def forward(self, images: list[torch.Tensor], targets: list[dict[str, Any]] | None = None):
        return self._model(images, targets)

    def get_input_format(self) -> str:
        return "coco"

    def get_output_format(self) -> str:
        return "coco_dict"

    def to(self, device):
        self._model.to(device)
        return self

    def train(self, mode: bool = True):
        self._model.train(mode)
        return self

    def eval(self):
        self._model.eval()
        return self

    def parameters(self):
        return self._model.parameters()

    def state_dict(self):
        return self._model.state_dict()

    def load_state_dict(self, state_dict, strict: bool = True):
        return self._model.load_state_dict(state_dict, strict=strict)

    @property
    def device(self):
        return next(self._model.parameters()).device

    def __getattr__(self, name: str):
        if name == "training":
            try:
                model = object.__getattribute__(self, "_model")
                return model.training
            except AttributeError:
                return False
        try:
            model = object.__getattribute__(self, "_model")
            return getattr(model, name)
        except AttributeError:
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{name}'"
            ) from None


# Register models
@ModelRegistry.register("fasterrcnn_resnet50")
def _create_fasterrcnn_resnet50(**kwargs):
    return FasterRCNNWrapper(
        backbone="resnet50",
        num_classes=kwargs.get("num_classes", 12),
        pretrained=kwargs.get("pretrained", True),
    )


@ModelRegistry.register("fasterrcnn_mobilenet")
def _create_fasterrcnn_mobilenet(**kwargs):
    return FasterRCNNWrapper(
        backbone="mobilenet",
        num_classes=kwargs.get("num_classes", 12),
        pretrained=kwargs.get("pretrained", True),
    )


@ModelRegistry.register("fcos_resnet50")
def _create_fcos_resnet50(**kwargs):
    return FCOSWrapper(
        num_classes=kwargs.get("num_classes", 12),
        pretrained=kwargs.get("pretrained", True),
    )


@ModelRegistry.register("retinanet_resnet50")
def _create_retinanet_resnet50(**kwargs):
    return RetinaNetWrapper(
        num_classes=kwargs.get("num_classes", 12),
        pretrained=kwargs.get("pretrained", True),
    )
