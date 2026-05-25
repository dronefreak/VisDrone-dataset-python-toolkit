"""
Integration tests for VisDrone toolkit end-to-end workflows.

Tests full training pipeline, empty annotation handling, metrics computation,
and soft-NMS functionality to ensure no regressions in core workflows.
"""


import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader

from visdrone_toolkit.augmentations import get_training_augmentation
from visdrone_toolkit.dataset import VisDroneDataset
from visdrone_toolkit.soft_nms_utils import soft_nms
from visdrone_toolkit.utils import collate_fn, compute_metrics, get_model


class TestEmptyAnnotationHandling:
    """Test handling of images with no annotations (critical bug fix)."""

    def test_empty_annotation_returns_empty_tensors(self, temp_dir):
        """Test that empty annotations return empty tensors, not dummy boxes."""
        # Create dataset with one image and empty annotation
        img_dir = temp_dir / "images"
        ann_dir = temp_dir / "annotations"
        img_dir.mkdir(parents=True)
        ann_dir.mkdir(parents=True)

        # Create image
        image = Image.new("RGB", (640, 480), color="red")
        img_path = img_dir / "empty_000000.jpg"
        image.save(img_path)

        # Create empty annotation (no objects)
        ann_path = ann_dir / "empty_000000.txt"
        ann_path.write_text("")  # Empty file

        dataset = VisDroneDataset(image_dir=str(img_dir), annotation_dir=str(ann_dir))
        img_tensor, target = dataset[0]

        # Check that boxes are empty, not dummy [0,0,1,1]
        assert target["boxes"].shape == (0, 4), "Expected empty boxes tensor (0, 4)"
        assert target["labels"].shape == (0,), "Expected empty labels tensor (0,)"
        assert target["image_id"].shape == (1,), "Expected image_id tensor"

        # Verify no fake pedestrian boxes
        assert len(target["boxes"]) == 0, "Should have no boxes for empty annotation"
        assert torch.all(target["boxes"] != torch.tensor([0, 0, 1, 1])), "Should not have dummy box"

    def test_augmentation_with_empty_annotation(self, temp_dir):
        """Test augmentation on image with empty annotations."""
        img_dir = temp_dir / "images"
        ann_dir = temp_dir / "annotations"
        img_dir.mkdir(parents=True)
        ann_dir.mkdir(parents=True)

        # Create image
        image = Image.new("RGB", (640, 480), color="blue")
        img_path = img_dir / "aug_empty_000000.jpg"
        image.save(img_path)

        # Empty annotation
        ann_path = ann_dir / "aug_empty_000000.txt"
        ann_path.write_text("")

        # Create dataset with augmentation
        augmentation = get_training_augmentation()
        dataset = VisDroneDataset(
            image_dir=str(img_dir),
            annotation_dir=str(ann_dir),
            transforms=augmentation,
        )

        img_tensor, target = dataset[0]

        # Verify empty tensors are preserved through augmentation
        assert target["boxes"].shape == (0, 4), "Augmentation should preserve empty boxes"
        assert target["labels"].shape == (0,), "Augmentation should preserve empty labels"

    def test_dataloader_with_empty_annotations(self, temp_dir):
        """Test DataLoader works correctly with empty annotations."""
        img_dir = temp_dir / "images"
        ann_dir = temp_dir / "annotations"
        img_dir.mkdir(parents=True)
        ann_dir.mkdir(parents=True)

        # Create mix of empty and non-empty annotations
        for i in range(3):
            image = Image.new("RGB", (640, 480), color="green")
            img_path = img_dir / f"mixed_{i:06d}.jpg"
            image.save(img_path)

            ann_path = ann_dir / f"mixed_{i:06d}.txt"
            if i == 1:
                # One empty annotation
                ann_path.write_text("")
            else:
                # Two with objects
                ann_path.write_text("100,100,50,50,1,1,0,0\n250,150,50,50,1,4,0,0\n")

        dataset = VisDroneDataset(image_dir=str(img_dir), annotation_dir=str(ann_dir))
        dataloader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)

        # Iterate through batches
        batch_count = 0
        for images, targets in dataloader:
            batch_count += 1
            assert isinstance(images, tuple), "Images should be tuple from collate_fn"
            assert isinstance(targets, tuple), "Targets should be tuple from collate_fn"
            assert len(images) == len(targets), "Batch size mismatch"

        assert batch_count > 0, "DataLoader should produce batches"


class TestSoftNMSDeviceHandling:
    """Test soft-NMS device compatibility (critical bug fix)."""

    def test_soft_nms_with_cpu_tensors(self):
        """Test soft-NMS works with CPU tensors (no .cpu() crash)."""
        boxes = torch.tensor(
            [[10.0, 10.0, 100.0, 100.0], [20.0, 20.0, 150.0, 150.0], [11.0, 11.0, 101.0, 101.0]],
            dtype=torch.float32,
        )
        scores = torch.tensor([0.9, 0.8, 0.85], dtype=torch.float32)

        # Should not crash with device-agnostic conversion
        keep, updated_scores = soft_nms(boxes, scores, iou_threshold=0.5)

        assert isinstance(keep, torch.Tensor), "keep should be tensor"
        assert isinstance(updated_scores, torch.Tensor), "scores should be tensor"
        assert len(keep) > 0, "Should keep at least one box"
        assert len(updated_scores) == len(scores), "Scores length should match input"

    def test_soft_nms_with_detached_tensors(self):
        """Test soft-NMS with .detach() tensors (model outputs)."""
        # Simulate model output (detached from computation graph)
        boxes = torch.tensor(
            [[10.0, 10.0, 100.0, 100.0], [20.0, 20.0, 150.0, 150.0]],
            dtype=torch.float32,
        ).detach()
        scores = torch.tensor([0.9, 0.8], dtype=torch.float32).detach()

        # Should handle detached tensors
        keep, updated_scores = soft_nms(boxes, scores)

        assert keep is not None
        assert updated_scores is not None

    def test_soft_nms_empty_tensors(self):
        """Test soft-NMS handles empty tensors gracefully."""
        empty_boxes = torch.tensor([], dtype=torch.float32).reshape(0, 4)
        empty_scores = torch.tensor([], dtype=torch.float32)

        keep, updated_scores = soft_nms(empty_boxes, empty_scores)

        assert keep.shape == (0,), "Keep should be empty tensor"
        assert updated_scores.shape == (0,), "Scores should be empty tensor"

    def test_soft_nms_single_box(self):
        """Test soft-NMS with single box (edge case)."""
        boxes = torch.tensor([[10.0, 10.0, 100.0, 100.0]], dtype=torch.float32)
        scores = torch.tensor([0.95], dtype=torch.float32)

        keep, updated_scores = soft_nms(boxes, scores)

        assert len(keep) == 1, "Should keep single box"
        assert updated_scores[keep[0]] > 0.9, "Score should be preserved"


class TestMetricsComputation:
    """Test metrics computation and documentation."""

    def test_compute_metrics_with_perfect_predictions(self):
        """Test metrics with perfect predictions (100% precision/recall)."""
        predictions = [
            {
                "boxes": torch.tensor([[100.0, 100.0, 200.0, 200.0]], dtype=torch.float32),
                "labels": torch.tensor([1], dtype=torch.int64),
                "scores": torch.tensor([0.99], dtype=torch.float32),
            }
        ]

        targets = [
            {
                "boxes": torch.tensor([[100.0, 100.0, 200.0, 200.0]], dtype=torch.float32),
                "labels": torch.tensor([1], dtype=torch.int64),
            }
        ]

        metrics = compute_metrics(predictions, targets)

        assert metrics["precision"] == 1.0, "Perfect predictions should have 100% precision"
        assert metrics["recall"] == 1.0, "Perfect predictions should have 100% recall"
        assert metrics["f1"] == 1.0, "Perfect predictions should have 100% F1"
        assert metrics["tp"] == 1, "Should have 1 TP"
        assert metrics["fp"] == 0, "Should have 0 FP"
        assert metrics["fn"] == 0, "Should have 0 FN"

    def test_compute_metrics_with_no_predictions(self):
        """Test metrics with missing predictions (FN case)."""
        predictions = [{"boxes": torch.tensor([], dtype=torch.float32).reshape(0, 4)}]

        targets = [
            {
                "boxes": torch.tensor([[100.0, 100.0, 200.0, 200.0]], dtype=torch.float32),
                "labels": torch.tensor([1], dtype=torch.int64),
            }
        ]

        metrics = compute_metrics(predictions, targets)

        assert metrics["tp"] == 0
        assert metrics["fn"] == 1, "Missing prediction should count as FN"
        assert metrics["precision"] == 0, "No predictions = 0 precision"
        assert metrics["recall"] == 0, "Missed all objects = 0 recall"

    def test_compute_metrics_empty_image(self):
        """Test metrics with empty image (no objects)."""
        predictions = [{"boxes": torch.tensor([], dtype=torch.float32).reshape(0, 4)}]
        targets = [{"boxes": torch.tensor([], dtype=torch.float32).reshape(0, 4)}]

        metrics = compute_metrics(predictions, targets)

        # Empty vs empty should not affect metrics
        assert metrics["tp"] == 0
        assert metrics["fp"] == 0
        assert metrics["fn"] == 0

    def test_compute_metrics_docstring_clarity(self):
        """Test that compute_metrics has clear documentation about limitations."""
        docstring = compute_metrics.__doc__

        assert docstring is not None, "Should have docstring"
        assert "training monitoring only" in docstring.lower(), "Should warn about limitations"
        assert "official" in docstring.lower(), "Should mention official evaluation"
        assert "pycocotools" in docstring.lower(), "Should reference alternative methods"


class TestMinimalTrainingPipeline:
    """Test minimal end-to-end training workflow."""

    def test_model_creation_and_forward_pass(self, device, num_classes):
        """Test model creation and forward pass with sample data."""
        # Create model
        model = get_model("fasterrcnn_mobilenet", num_classes=num_classes, pretrained=False)
        model = model.to(device)
        model.train()

        # Create dummy batch
        images = [torch.randn(3, 480, 640).to(device) for _ in range(2)]
        targets = [
            {
                "boxes": torch.tensor([[100.0, 100.0, 200.0, 200.0]], dtype=torch.float32).to(
                    device
                ),
                "labels": torch.tensor([1], dtype=torch.int64).to(device),
            },
            {
                "boxes": torch.tensor(
                    [[50.0, 50.0, 150.0, 150.0], [200.0, 200.0, 300.0, 300.0]],
                    dtype=torch.float32,
                ).to(device),
                "labels": torch.tensor([2, 4], dtype=torch.int64).to(device),
            },
        ]

        # Forward pass should return losses
        loss_dict = model(images, targets)

        assert isinstance(loss_dict, dict), "Should return loss dict in training mode"
        assert "loss_classifier" in loss_dict or "loss_objectness" in loss_dict
        assert all(v.requires_grad for v in loss_dict.values()), "Losses should be differentiable"

    def test_model_inference_mode(self, device, num_classes):
        """Test model in inference mode."""
        model = get_model("fasterrcnn_mobilenet", num_classes=num_classes, pretrained=False)
        model = model.to(device)
        model.eval()

        # Inference should not require targets
        images = [torch.randn(3, 480, 640).to(device) for _ in range(2)]

        with torch.no_grad():
            predictions = model(images)

        assert isinstance(predictions, list), "Should return list of predictions"
        assert len(predictions) == 2, "Should have same number of predictions as images"

        for pred in predictions:
            assert "boxes" in pred
            assert "labels" in pred
            assert "scores" in pred

    def test_gradient_flow_single_step(self, device, num_classes):
        """Test that gradients flow through training step."""
        model = get_model("fasterrcnn_mobilenet", num_classes=num_classes, pretrained=False)
        model = model.to(device)
        model.train()

        optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

        images = [torch.randn(3, 480, 640).to(device)]
        targets = [
            {
                "boxes": torch.tensor([[100.0, 100.0, 200.0, 200.0]], dtype=torch.float32).to(
                    device
                ),
                "labels": torch.tensor([1], dtype=torch.int64).to(device),
            }
        ]

        # Training step
        optimizer.zero_grad()
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        losses.backward()
        optimizer.step()

        # Check that gradients were computed
        has_gradients = any(
            p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters()
        )
        assert has_gradients, "Should have non-zero gradients after backward pass"

    def test_multi_class_batch_processing(self, device, num_classes):
        """Test batch with multiple object classes."""
        model = get_model("fasterrcnn_mobilenet", num_classes=num_classes, pretrained=False)
        model = model.to(device)
        model.eval()

        batch_size = 3
        images = [torch.randn(3, 480, 640).to(device) for _ in range(batch_size)]

        with torch.no_grad():
            predictions = model(images)

        assert len(predictions) == batch_size, "Should handle multi-class batch"

        for pred in predictions:
            if len(pred["boxes"]) > 0:
                # If any predictions, check they're in valid range
                assert torch.all(pred["labels"] < num_classes), "Labels should be < num_classes"
                assert torch.all(pred["scores"] >= 0.0) and torch.all(pred["scores"] <= 1.0)


class TestDatasetIntegration:
    """Test dataset integration with models."""

    def test_dataset_dataloader_model_pipeline(self, temp_dir):
        """Test full pipeline: dataset -> dataloader -> model."""
        # Create small dataset
        img_dir = temp_dir / "images"
        ann_dir = temp_dir / "annotations"
        img_dir.mkdir(parents=True)
        ann_dir.mkdir(parents=True)

        for i in range(2):
            image = Image.new("RGB", (640, 480), color="red")
            img_path = img_dir / f"pipeline_{i:06d}.jpg"
            image.save(img_path)

            ann_path = ann_dir / f"pipeline_{i:06d}.txt"
            ann_path.write_text("100,100,50,50,1,1,0,0\n200,200,50,50,1,4,0,0\n")

        # Create dataset
        dataset = VisDroneDataset(image_dir=str(img_dir), annotation_dir=str(ann_dir))
        dataloader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn)

        # Create model
        model = get_model("fasterrcnn_mobilenet", num_classes=12, pretrained=False)
        model.eval()

        # Process batches through model
        with torch.no_grad():
            for images, _ in dataloader:
                predictions = model(images)
                assert len(predictions) > 0

        assert True, "Full pipeline executed successfully"


class TestAugmentationIntegration:
    """Test augmentation integration with dataset."""

    def test_augmentation_preserves_box_format(self, temp_dir):
        """Test augmentation preserves bounding box format."""
        img_dir = temp_dir / "images"
        ann_dir = temp_dir / "annotations"
        img_dir.mkdir(parents=True)
        ann_dir.mkdir(parents=True)

        image = Image.new("RGB", (640, 480), color="green")
        img_path = img_dir / "aug_000000.jpg"
        image.save(img_path)

        ann_path = ann_dir / "aug_000000.txt"
        ann_path.write_text("100,100,50,50,1,1,0,0\n")

        augmentation = get_training_augmentation()
        dataset = VisDroneDataset(
            image_dir=str(img_dir),
            annotation_dir=str(ann_dir),
            transforms=augmentation,
        )

        img_tensor, target = dataset[0]

        # Check format
        assert img_tensor.dim() == 3, "Image should be 3D (C, H, W)"
        assert img_tensor.shape[0] == 3, "Image should be RGB"
        assert target["boxes"].dim() == 2, "Boxes should be 2D"
        assert target["boxes"].shape[1] == 4, "Boxes should be [x1,y1,x2,y2]"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
