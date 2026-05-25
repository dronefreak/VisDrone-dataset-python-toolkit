"""
Format converters for different object detection formats.

Converts between different bounding box representations used by different frameworks:
- COCO: [x1, y1, x2, y2] in absolute pixel coordinates
- YOLO: [x_center, y_center, w, h] in normalized (0-1) coordinates
- DETR: [x_center, y_center, w, h] in normalized coordinates with metadata
"""

from typing import Dict, List

import torch

from .abstract_models import FormatConverter


class YOLOFormatConverter(FormatConverter):
    """
    Converter between COCO and YOLO bounding box formats.

    COCO format: [x1, y1, x2, y2] (absolute coordinates)
    YOLO format: [x_center, y_center, w, h] (normalized 0-1)
    """

    def to_internal_format(
        self, targets: List[Dict[str, torch.Tensor]]
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Convert from YOLO format to internal COCO format.

        Args:
            targets: List of target dicts with YOLO format boxes

        Returns:
            List of target dicts with COCO format boxes
        """
        converted = []

        for target in targets:
            boxes = target.get("boxes", torch.empty((0, 4)))

            if len(boxes) > 0:
                # Get image dimensions
                # For YOLO, we need to know the image size
                # This should be provided in the target dict
                image_height = target.get("image_height", 640)
                image_width = target.get("image_width", 640)

                boxes_coco = self.yolo_to_coco(boxes, (image_height, image_width))
            else:
                boxes_coco = boxes

            new_target = dict(target)
            new_target["boxes"] = boxes_coco

            # Remove YOLO-specific fields
            new_target.pop("image_height", None)
            new_target.pop("image_width", None)

            converted.append(new_target)

        return converted

    def from_internal_format(
        self, targets: List[Dict[str, torch.Tensor]]
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Convert from internal COCO format to YOLO format.

        Args:
            targets: List of target dicts with COCO format boxes

        Returns:
            List of target dicts with YOLO format boxes
        """
        converted = []

        for target in targets:
            boxes = target.get("boxes", torch.empty((0, 4)))

            if len(boxes) > 0:
                # Get image dimensions
                # These should be provided separately or stored in the batch
                image_height = target.get("image_height", 640)
                image_width = target.get("image_width", 640)

                boxes_yolo = self.coco_to_yolo(boxes, (image_height, image_width))
            else:
                boxes_yolo = boxes

            new_target = dict(target)
            new_target["boxes"] = boxes_yolo
            new_target["image_height"] = target.get("image_height", 640)
            new_target["image_width"] = target.get("image_width", 640)

            converted.append(new_target)

        return converted


class DETRFormatConverter(FormatConverter):
    """
    Converter for DETR (Detection Transformer) format.

    DETR uses COCO format with additional metadata:
    - boxes: [x_center, y_center, w, h] in normalized coordinates
    - labels: class indices
    - image_id: image identifier
    - area: bounding box area
    - iscrowd: crowd annotation flag
    """

    def to_internal_format(
        self, targets: List[Dict[str, torch.Tensor]]
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Convert from DETR format to internal COCO format.

        DETR uses normalized coordinates, so convert to absolute.

        Args:
            targets: List of target dicts with DETR format

        Returns:
            List of target dicts with COCO format (absolute coordinates)
        """
        converted = []

        for target in targets:
            boxes = target.get("boxes", torch.empty((0, 4)))

            if len(boxes) > 0:
                # DETR boxes are normalized [x_center, y_center, w, h]
                # Convert to absolute [x1, y1, x2, y2]
                image_height = target.get("image_height", 640)
                image_width = target.get("image_width", 640)

                boxes_coco = self.yolo_to_coco(boxes, (image_height, image_width))
            else:
                boxes_coco = boxes

            new_target = dict(target)
            new_target["boxes"] = boxes_coco

            # Keep only essential fields for internal use
            # Remove DETR-specific metadata
            for key in ["image_id", "area", "iscrowd", "image_height", "image_width"]:
                new_target.pop(key, None)

            converted.append(new_target)

        return converted

    def from_internal_format(
        self, targets: List[Dict[str, torch.Tensor]]
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Convert from internal COCO format to DETR format.

        Adds required DETR metadata and converts to normalized coordinates.

        Args:
            targets: List of target dicts with COCO format (absolute coordinates)

        Returns:
            List of target dicts with DETR format (normalized coordinates)
        """
        converted = []

        for target in targets:
            boxes = target.get("boxes", torch.empty((0, 4)))

            if len(boxes) > 0:
                # COCO boxes are absolute [x1, y1, x2, y2]
                # Convert to normalized [x_center, y_center, w, h]
                image_height = target.get("image_height", 640)
                image_width = target.get("image_width", 640)

                boxes_detr = self.coco_to_yolo(boxes, (image_height, image_width))

                # Compute area for DETR
                x1, y1, x2, y2 = (boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3])
                areas = (x2 - x1) * (y2 - y1)
            else:
                boxes_detr = boxes
                areas = torch.empty((0,), dtype=torch.float32)

            new_target = dict(target)
            new_target["boxes"] = boxes_detr
            new_target["area"] = areas
            new_target["iscrowd"] = target.get(
                "iscrowd", torch.zeros(len(boxes), dtype=torch.int64)
            )
            new_target["image_id"] = target.get("image_id", torch.tensor(0))
            new_target["image_height"] = target.get("image_height", 640)
            new_target["image_width"] = target.get("image_width", 640)

            converted.append(new_target)

        return converted


class COCOFormatConverter(FormatConverter):
    """Identity converter for COCO format (no conversion needed)."""

    def to_internal_format(
        self, targets: List[Dict[str, torch.Tensor]]
    ) -> List[Dict[str, torch.Tensor]]:
        """Return targets unchanged (already in internal format)."""
        return targets

    def from_internal_format(
        self, targets: List[Dict[str, torch.Tensor]]
    ) -> List[Dict[str, torch.Tensor]]:
        """Return targets unchanged (already in internal format)."""
        return targets
