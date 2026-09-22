"""Format converters for VisDrone annotations.

Supports conversion to:
- COCO format (JSON)
- YOLO format (TXT)
- PASCAL VOC format (XML) - legacy support

Supports conversion from:
- COCO format (JSON)
"""

from visdrone_toolkit.converters.coco_to_visdrone import coco_to_visdrone, validate_visdrone_format
from visdrone_toolkit.converters.visdrone_to_coco import convert_to_coco
from visdrone_toolkit.converters.visdrone_to_yolo import convert_to_yolo

__all__ = [
    "convert_to_coco",
    "convert_to_yolo",
    "coco_to_visdrone",
    "validate_visdrone_format",
]
