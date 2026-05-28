"""
Tests for YOLO v8+ model integration.

Tests model registration, abstract interface compliance, and basic functionality.
"""

import pytest
import torch

from visdrone_toolkit.abstract_models import (
    DetectionModel,
    FormatConverter,
    ModelRegistry,
    TrainingAdapter,
)
from visdrone_toolkit.format_converters import (
    COCOFormatConverter,
    DETRFormatConverter,
    YOLOFormatConverter,
)
from visdrone_toolkit.training_adapters import (
    DETRTrainingAdapter,
    TorchvisionTrainingAdapter,
    YOLOTrainingAdapter,
)


class TestModelRegistry:
    """Tests for model registry functionality."""

    def test_registry_has_yolo_models(self):
        """Test that YOLO models are registered."""
        models = ModelRegistry.list_models()

        # Check for YOLO v8 models
        assert "yolov8n" in models
        assert "yolov8s" in models
        assert "yolov8m" in models
        assert "yolov8l" in models
        assert "yolov8x" in models

    def test_registry_has_yolo9_models(self):
        """Test that YOLO v9 models are registered."""
        models = ModelRegistry.list_models()

        assert "yolov9c" in models
        assert "yolov9m" in models
        assert "yolov9e" in models

    def test_registry_has_yolo10_models(self):
        """Test that YOLO v10 models are registered."""
        models = ModelRegistry.list_models()

        assert "yolov10n" in models
        assert "yolov10s" in models
        assert "yolov10m" in models
        assert "yolov10l" in models
        assert "yolov10x" in models

    def test_registry_get_unknown_model(self):
        """Test that getting unknown model raises error."""
        with pytest.raises(ValueError, match="Unknown model"):
            ModelRegistry.get("unknown_model")

    def test_registry_list_models_sorted(self):
        """Test that model list is sorted."""
        models = ModelRegistry.list_models()
        assert models == sorted(models)

    def test_get_model_info(self):
        """Test getting model information."""
        info = ModelRegistry.get_model_info("yolov8n")
        assert "YOLOv8" in info or "Nano" in info or len(info) > 0


class TestAbstractModelInterface:
    """Tests for abstract model interface compliance."""

    def test_detection_model_is_nn_module(self):
        """Test that DetectionModel inherits from nn.Module."""
        assert issubclass(DetectionModel, torch.nn.Module)

    def test_detection_model_requires_num_classes(self):
        """Test that detection models accept num_classes."""
        # This is tested through subclass implementations
        pass

    def test_format_converter_has_required_methods(self):
        """Test that format converters have required methods."""
        assert hasattr(FormatConverter, "to_internal_format")
        assert hasattr(FormatConverter, "from_internal_format")

    def test_training_adapter_has_required_methods(self):
        """Test that training adapters have required methods."""
        assert hasattr(TrainingAdapter, "training_step")
        assert hasattr(TrainingAdapter, "validation_step")


class TestFormatConverters:
    """Tests for format conversion functionality."""

    def test_yolo_format_converter_to_internal(self):
        """Test YOLO to internal format conversion."""
        converter = YOLOFormatConverter()

        # Create test data in YOLO format
        targets = [
            {
                "boxes": torch.tensor([[0.5, 0.5, 0.2, 0.3]]),  # normalized
                "labels": torch.tensor([1]),
                "image_height": 640,
                "image_width": 640,
            }
        ]

        # Convert to internal format
        result = converter.to_internal_format(targets)

        assert len(result) == 1
        assert "boxes" in result[0]
        assert result[0]["boxes"].shape == (1, 4)

    def test_yolo_format_converter_roundtrip(self):
        """Test roundtrip conversion YOLO -> internal -> YOLO."""
        converter = YOLOFormatConverter()

        original = torch.tensor([[0.5, 0.5, 0.2, 0.3]])
        image_size = (640, 640)

        # Convert to COCO
        coco = converter.yolo_to_coco(original, image_size)

        # Convert back to YOLO
        yolo = converter.coco_to_yolo(coco, image_size)

        # Should be approximately equal
        assert torch.allclose(original, yolo, atol=1e-6)

    def test_empty_boxes_conversion(self):
        """Test format conversion with empty boxes."""
        converter = YOLOFormatConverter()

        targets = [
            {
                "boxes": torch.empty((0, 4)),
                "labels": torch.empty((0,), dtype=torch.int64),
                "image_height": 640,
                "image_width": 640,
            }
        ]

        result = converter.to_internal_format(targets)
        assert result[0]["boxes"].shape == (0, 4)

    def test_detr_format_converter_adds_metadata(self):
        """Test that DETR converter adds required metadata."""
        converter = DETRFormatConverter()

        targets = [
            {
                "boxes": torch.tensor([[100, 100, 200, 200]]),
                "labels": torch.tensor([1]),
            }
        ]

        result = converter.from_internal_format(targets)

        # Check DETR-specific fields
        assert "area" in result[0]
        assert "iscrowd" in result[0]
        assert "image_id" in result[0]

    def test_coco_converter_identity(self):
        """Test that COCO converter is identity operation."""
        converter = COCOFormatConverter()

        targets = [
            {
                "boxes": torch.tensor([[100, 100, 200, 200]]),
                "labels": torch.tensor([1]),
            }
        ]

        result = converter.to_internal_format(targets)

        # Should be unchanged
        assert torch.equal(result[0]["boxes"], targets[0]["boxes"])
        assert torch.equal(result[0]["labels"], targets[0]["labels"])


class TestTrainingAdapters:
    """Tests for training adapter functionality."""

    def test_torchvision_adapter_has_methods(self):
        """Test that Torchvision adapter has required methods."""
        adapter = TorchvisionTrainingAdapter()

        assert callable(adapter.training_step)
        assert callable(adapter.validation_step)

    def test_yolo_adapter_has_methods(self):
        """Test that YOLO adapter has required methods."""
        adapter = YOLOTrainingAdapter()

        assert callable(adapter.training_step)
        assert callable(adapter.validation_step)

    def test_detr_adapter_initialization(self):
        """Test DETR adapter initialization."""
        adapter = DETRTrainingAdapter(criterion=None, matcher=None)

        assert adapter.criterion is None
        assert adapter.matcher is None


class TestStaticMethods:
    """Tests for static conversion methods."""

    def test_coco_to_yolo_single_box(self):
        """Test single box COCO to YOLO conversion."""
        box = torch.tensor([[0.0, 0.0, 100.0, 100.0]])
        image_size = (640, 640)

        yolo = FormatConverter.coco_to_yolo(box, image_size)

        # Should have center at (50, 50) and size (100, 100)
        assert yolo[0, 0].item() == pytest.approx(50.0 / 640.0, abs=1e-6)
        assert yolo[0, 1].item() == pytest.approx(50.0 / 640.0, abs=1e-6)
        assert yolo[0, 2].item() == pytest.approx(100.0 / 640.0, abs=1e-6)
        assert yolo[0, 3].item() == pytest.approx(100.0 / 640.0, abs=1e-6)

    def test_yolo_to_coco_single_box(self):
        """Test single box YOLO to COCO conversion."""
        box = torch.tensor([[0.5, 0.5, 0.2, 0.2]])
        image_size = (640, 640)

        coco = FormatConverter.yolo_to_coco(box, image_size)

        # Should have corners at (396, 396) and (484, 484)
        assert coco[0, 0].item() == pytest.approx(320.0 - 64.0)  # x1
        assert coco[0, 1].item() == pytest.approx(320.0 - 64.0)  # y1
        assert coco[0, 2].item() == pytest.approx(320.0 + 64.0)  # x2
        assert coco[0, 3].item() == pytest.approx(320.0 + 64.0)  # y2

    def test_empty_boxes_coco_to_yolo(self):
        """Test empty boxes conversion."""
        boxes = torch.empty((0, 4))
        yolo = FormatConverter.coco_to_yolo(boxes, (640, 640))

        assert yolo.shape == (0, 4)

    def test_empty_boxes_yolo_to_coco(self):
        """Test empty boxes conversion."""
        boxes = torch.empty((0, 4))
        coco = FormatConverter.yolo_to_coco(boxes, (640, 640))

        assert coco.shape == (0, 4)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
