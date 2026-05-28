"""VisDrone Toolkit - Modern PyTorch-based toolkit for VisDrone dataset.

A comprehensive toolkit for working with the VisDrone dataset, featuring:
- Native PyTorch Dataset class
- Multiple annotation format converters (COCO, YOLO)
- Visualization utilities
- Training scripts for modern object detection models
- Support for YOLO v8+, torchvision, and DETR models

"""

__version__ = "2.0.0"
__author__ = "Saumya Kumaar Saksena"
__license__ = "Apache-2.0"

from visdrone_toolkit.dataset import VisDroneDataset

# Register all models
from visdrone_toolkit.torchvision_models import (  # noqa: F401
    FasterRCNNWrapper,
    FCOSWrapper,
    RetinaNetWrapper,
)
from visdrone_toolkit.trainer import UnifiedTrainer  # noqa: F401
from visdrone_toolkit.utils import VISDRONE_CLASSES, collate_fn, get_model
from visdrone_toolkit.visualization import visualize_annotations, visualize_predictions
from visdrone_toolkit.yolo_models import YOLOv8Base  # noqa: F401

__all__ = [
    "VisDroneDataset",
    "VISDRONE_CLASSES",
    "get_model",
    "collate_fn",
    "visualize_annotations",
    "visualize_predictions",
    "UnifiedTrainer",
    "FasterRCNNWrapper",
    "FCOSWrapper",
    "RetinaNetWrapper",
    "YOLOv8Base",
]
