"""
Tests for utility functions.
"""

import pytest
import torch

from visdrone_toolkit.utils import (
    NUM_CLASSES,
    VISDRONE_CLASSES,
    box_iou,
    collate_fn,
    compute_metrics,
    get_model,
)


class TestModelFactory:
    """Tests for get_model function."""

    def test_get_fasterrcnn_resnet50(self, num_classes):
        """Test creating Faster R-CNN ResNet50 model."""
        model = get_model("fasterrcnn_resnet50", num_classes=num_classes, pretrained=False)
        assert model is not None
        assert hasattr(model, "roi_heads")

    def test_get_fasterrcnn_mobilenet(self, num_classes):
        """Test creating Faster R-CNN MobileNet model."""
        model = get_model("fasterrcnn_mobilenet", num_classes=num_classes, pretrained=False)
        assert model is not None
        assert hasattr(model, "roi_heads")

    def test_get_fcos_resnet50(self, num_classes):
        """Test creating FCOS model."""
        model = get_model("fcos_resnet50", num_classes=num_classes, pretrained=False)
        assert model is not None
        assert hasattr(model, "head")

    def test_get_retinanet_resnet50(self, num_classes):
        """Test creating RetinaNet model."""
        model = get_model("retinanet_resnet50", num_classes=num_classes, pretrained=False)
        assert model is not None
        assert hasattr(model, "head")

    def test_invalid_model_name(self, num_classes):
        """Test error handling for invalid model name."""
        with pytest.raises(ValueError):
            get_model("invalid_model", num_classes=num_classes)

    def test_model_eval_mode(self, num_classes):
        """Test model can be set to eval mode."""
        model = get_model("fasterrcnn_resnet50", num_classes=num_classes, pretrained=False)
        model.eval()
        assert not model.training

    def test_model_parameters(self, num_classes):
        """Test model has trainable parameters."""
        model = get_model("fasterrcnn_resnet50", num_classes=num_classes, pretrained=False)
        params = list(model.parameters())
        assert len(params) > 0
        assert any(p.requires_grad for p in params)


class TestCollateFn:
    """Tests for collate_fn."""

    def test_collate_single_batch(self, sample_image_array, sample_target):
        """Test collating a single batch."""
        batch = [
            (torch.randn(3, 480, 640), sample_target),
        ]

        images, targets = collate_fn(batch)

        assert len(images) == 1
        assert len(targets) == 1
        assert isinstance(images[0], torch.Tensor)
        assert isinstance(targets[0], dict)

    def test_collate_multiple_batches(self, sample_target):
        """Test collating multiple items."""
        batch = [
            (torch.randn(3, 480, 640), sample_target),
            (torch.randn(3, 480, 640), sample_target),
            (torch.randn(3, 480, 640), sample_target),
        ]

        images, targets = collate_fn(batch)

        assert len(images) == 3
        assert len(targets) == 3


class TestBoxIoU:
    """Tests for box_iou function."""

    def test_iou_identical_boxes(self):
        """Test IoU of identical boxes should be 1.0."""
        boxes1 = torch.tensor([[100, 100, 200, 200]], dtype=torch.float32)
        boxes2 = torch.tensor([[100, 100, 200, 200]], dtype=torch.float32)

        iou = box_iou(boxes1, boxes2)

        assert torch.isclose(iou[0, 0], torch.tensor(1.0))

    def test_iou_non_overlapping_boxes(self):
        """Test IoU of non-overlapping boxes should be 0.0."""
        boxes1 = torch.tensor([[0, 0, 100, 100]], dtype=torch.float32)
        boxes2 = torch.tensor([[200, 200, 300, 300]], dtype=torch.float32)

        iou = box_iou(boxes1, boxes2)

        assert torch.isclose(iou[0, 0], torch.tensor(0.0))

    def test_iou_partial_overlap(self):
        """Test IoU of partially overlapping boxes."""
        boxes1 = torch.tensor([[0, 0, 100, 100]], dtype=torch.float32)
        boxes2 = torch.tensor([[50, 50, 150, 150]], dtype=torch.float32)

        iou = box_iou(boxes1, boxes2)

        # IoU should be 0.25 (intersection=2500, union=10000+10000-2500=17500)
        expected_iou = 2500 / 17500
        assert torch.isclose(iou[0, 0], torch.tensor(expected_iou), atol=1e-4)

    def test_iou_multiple_boxes(self):
        """Test IoU computation for multiple boxes."""
        boxes1 = torch.tensor(
            [
                [0, 0, 100, 100],
                [50, 50, 150, 150],
            ],
            dtype=torch.float32,
        )
        boxes2 = torch.tensor(
            [
                [0, 0, 100, 100],
                [200, 200, 300, 300],
            ],
            dtype=torch.float32,
        )

        iou = box_iou(boxes1, boxes2)

        assert iou.shape == (2, 2)
        assert torch.isclose(iou[0, 0], torch.tensor(1.0))
        assert torch.isclose(iou[0, 1], torch.tensor(0.0))


class TestComputeMetrics:
    """Tests for compute_metrics function."""

    def test_perfect_predictions(self):
        """Test metrics with perfect predictions."""
        predictions = [
            {
                "boxes": torch.tensor([[100, 100, 200, 200]], dtype=torch.float32),
                "labels": torch.tensor([1], dtype=torch.int64),
                "scores": torch.tensor([0.95], dtype=torch.float32),
            }
        ]
        targets = [
            {
                "boxes": torch.tensor([[100, 100, 200, 200]], dtype=torch.float32),
                "labels": torch.tensor([1], dtype=torch.int64),
            }
        ]

        metrics = compute_metrics(predictions, targets, iou_threshold=0.5)

        assert metrics["tp"] == 1
        assert metrics["fp"] == 0
        assert metrics["fn"] == 0
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0
        assert metrics["f1"] == 1.0

    def test_no_predictions(self):
        """Test metrics with no predictions."""
        predictions = [
            {
                "boxes": torch.tensor([], dtype=torch.float32).reshape(0, 4),
                "labels": torch.tensor([], dtype=torch.int64),
                "scores": torch.tensor([], dtype=torch.float32),
            }
        ]
        targets = [
            {
                "boxes": torch.tensor([[100, 100, 200, 200]], dtype=torch.float32),
                "labels": torch.tensor([1], dtype=torch.int64),
            }
        ]

        metrics = compute_metrics(predictions, targets, iou_threshold=0.5)

        assert metrics["tp"] == 0
        assert metrics["fp"] == 0
        assert metrics["fn"] == 1
        assert metrics["precision"] == 0.0
        assert metrics["recall"] == 0.0

    def test_false_positives(self):
        """Test metrics with false positive predictions."""
        predictions = [
            {
                "boxes": torch.tensor(
                    [
                        [100, 100, 200, 200],
                        [300, 300, 400, 400],  # False positive
                    ],
                    dtype=torch.float32,
                ),
                "labels": torch.tensor([1, 1], dtype=torch.int64),
                "scores": torch.tensor([0.95, 0.90], dtype=torch.float32),
            }
        ]
        targets = [
            {
                "boxes": torch.tensor([[100, 100, 200, 200]], dtype=torch.float32),
                "labels": torch.tensor([1], dtype=torch.int64),
            }
        ]

        metrics = compute_metrics(predictions, targets, iou_threshold=0.5)

        assert metrics["tp"] == 1
        assert metrics["fp"] == 1
        assert metrics["fn"] == 0

    def test_wrong_class_prediction(self):
        """Test metrics when predicted class is wrong."""
        predictions = [
            {
                "boxes": torch.tensor([[100, 100, 200, 200]], dtype=torch.float32),
                "labels": torch.tensor([2], dtype=torch.int64),  # Wrong class
                "scores": torch.tensor([0.95], dtype=torch.float32),
            }
        ]
        targets = [
            {
                "boxes": torch.tensor([[100, 100, 200, 200]], dtype=torch.float32),
                "labels": torch.tensor([1], dtype=torch.int64),  # Correct class
            }
        ]

        metrics = compute_metrics(predictions, targets, iou_threshold=0.5)

        # Should be false positive + false negative
        assert metrics["tp"] == 0
        assert metrics["fp"] == 1
        assert metrics["fn"] == 1


class TestConstants:
    """Tests for constants."""

    def test_visdrone_classes_count(self):
        """Test number of VisDrone classes."""
        assert len(VISDRONE_CLASSES) == 12

    def test_num_classes_constant(self):
        """Test NUM_CLASSES constant."""
        assert NUM_CLASSES == 12
        assert len(VISDRONE_CLASSES) == NUM_CLASSES

    def test_visdrone_classes_names(self):
        """Test VisDrone class names."""
        assert VISDRONE_CLASSES[0] == "ignored-regions"
        assert VISDRONE_CLASSES[1] == "pedestrian"
        assert VISDRONE_CLASSES[4] == "car"
        assert VISDRONE_CLASSES[9] == "bus"
        assert VISDRONE_CLASSES[11] == "others"
