"""
Tests for VisDroneDataset class.
"""

from pathlib import Path

import pytest
import torch

from visdrone_toolkit.dataset import VisDroneDataset


class TestVisDroneDataset:
    """Tests for VisDroneDataset class."""

    def test_dataset_initialization(self, mock_visdrone_dataset):
        """Test dataset can be initialized."""
        dataset = VisDroneDataset(
            image_dir=str(mock_visdrone_dataset["image_dir"]),
            annotation_dir=str(mock_visdrone_dataset["annotation_dir"]),
        )

        assert len(dataset) == mock_visdrone_dataset["num_images"]
        assert len(dataset.image_files) == mock_visdrone_dataset["num_images"]

    def test_dataset_length(self, mock_visdrone_dataset):
        """Test __len__ method."""
        dataset = VisDroneDataset(
            image_dir=str(mock_visdrone_dataset["image_dir"]),
            annotation_dir=str(mock_visdrone_dataset["annotation_dir"]),
        )

        assert len(dataset) == 3

    def test_dataset_getitem(self, mock_visdrone_dataset):
        """Test __getitem__ method."""
        dataset = VisDroneDataset(
            image_dir=str(mock_visdrone_dataset["image_dir"]),
            annotation_dir=str(mock_visdrone_dataset["annotation_dir"]),
        )

        image, target = dataset[0]

        # Check image
        assert isinstance(image, torch.Tensor)
        assert image.dim() == 3  # (C, H, W)
        assert image.shape[0] == 3  # RGB

        # Check target
        assert isinstance(target, dict)
        assert "boxes" in target
        assert "labels" in target
        assert "image_id" in target
        assert "area" in target
        assert "iscrowd" in target

    def test_filter_ignored_boxes(self, mock_visdrone_dataset):
        """Test filtering of ignored boxes (score=0)."""
        # With filtering
        dataset_filtered = VisDroneDataset(
            image_dir=str(mock_visdrone_dataset["image_dir"]),
            annotation_dir=str(mock_visdrone_dataset["annotation_dir"]),
            filter_ignored=True,
        )

        _, target_filtered = dataset_filtered[0]

        # Without filtering
        dataset_unfiltered = VisDroneDataset(
            image_dir=str(mock_visdrone_dataset["image_dir"]),
            annotation_dir=str(mock_visdrone_dataset["annotation_dir"]),
            filter_ignored=False,
        )

        _, target_unfiltered = dataset_unfiltered[0]

        # Filtered should have fewer boxes
        assert len(target_filtered["boxes"]) <= len(target_unfiltered["boxes"])

    def test_filter_crowd_regions(self, mock_visdrone_dataset):
        """Test filtering of crowd regions (category=0)."""
        dataset = VisDroneDataset(
            image_dir=str(mock_visdrone_dataset["image_dir"]),
            annotation_dir=str(mock_visdrone_dataset["annotation_dir"]),
            filter_crowd=True,
        )

        _, target = dataset[0]

        # Should not have any category 0 (ignored-regions)
        assert 0 not in target["labels"]

    def test_annotation_parsing(self, mock_visdrone_dataset):
        """Test VisDrone annotation parsing."""
        dataset = VisDroneDataset(
            image_dir=str(mock_visdrone_dataset["image_dir"]),
            annotation_dir=str(mock_visdrone_dataset["annotation_dir"]),
            filter_ignored=True,
            filter_crowd=True,
        )

        _, target = dataset[0]

        # Check boxes format [x1, y1, x2, y2]
        boxes = target["boxes"]
        assert boxes.shape[1] == 4

        # Check valid coordinates (x2 > x1, y2 > y1)
        assert torch.all(boxes[:, 2] > boxes[:, 0])
        assert torch.all(boxes[:, 3] > boxes[:, 1])

        # Check labels are integers
        assert target["labels"].dtype == torch.int64

    def test_empty_annotation(self, temp_dir, sample_image):
        """Test handling of images with no annotations."""
        # Create dataset with empty annotation
        img_dir = temp_dir / "images"
        ann_dir = temp_dir / "annotations"
        img_dir.mkdir(parents=True)
        ann_dir.mkdir(parents=True)

        # Save image
        img_path = img_dir / "test.jpg"
        sample_image.save(img_path)

        # Save empty annotation
        ann_path = ann_dir / "test.txt"
        ann_path.write_text("")

        dataset = VisDroneDataset(
            image_dir=str(img_dir),
            annotation_dir=str(ann_dir),
        )

        image, target = dataset[0]

        # Should return empty boxes and labels
        assert len(target["boxes"]) == 0
        assert len(target["labels"]) == 0

    def test_get_image_path(self, mock_visdrone_dataset):
        """Test get_image_path method."""
        dataset = VisDroneDataset(
            image_dir=str(mock_visdrone_dataset["image_dir"]),
            annotation_dir=str(mock_visdrone_dataset["annotation_dir"]),
        )

        path = dataset.get_image_path(0)
        assert isinstance(path, Path)
        assert path.exists()
        assert path.suffix.lower() in [".jpg", ".jpeg", ".png"]

    def test_get_class_name(self, mock_visdrone_dataset):
        """Test get_class_name method."""
        dataset = VisDroneDataset(
            image_dir=str(mock_visdrone_dataset["image_dir"]),
            annotation_dir=str(mock_visdrone_dataset["annotation_dir"]),
        )

        assert dataset.get_class_name(0) == "ignored-regions"
        assert dataset.get_class_name(1) == "pedestrian"
        assert dataset.get_class_name(4) == "car"
        assert dataset.get_class_name(999) == "unknown"

    def test_get_num_classes(self, mock_visdrone_dataset):
        """Test get_num_classes static method."""
        num_classes = VisDroneDataset.get_num_classes()
        assert num_classes == 12

    def test_invalid_directory(self):
        """Test error handling for invalid directories."""
        with pytest.raises(ValueError):
            VisDroneDataset(
                image_dir="/nonexistent/path",
                annotation_dir="/nonexistent/path",
            )

    def test_target_keys(self, mock_visdrone_dataset):
        """Test that target has all required keys."""
        dataset = VisDroneDataset(
            image_dir=str(mock_visdrone_dataset["image_dir"]),
            annotation_dir=str(mock_visdrone_dataset["annotation_dir"]),
        )

        _, target = dataset[0]

        required_keys = ["boxes", "labels", "image_id", "area", "iscrowd"]
        for key in required_keys:
            assert key in target

    def test_area_computation(self, mock_visdrone_dataset):
        """Test that area is computed correctly."""
        dataset = VisDroneDataset(
            image_dir=str(mock_visdrone_dataset["image_dir"]),
            annotation_dir=str(mock_visdrone_dataset["annotation_dir"]),
            filter_ignored=True,
            filter_crowd=True,
        )

        _, target = dataset[0]

        boxes = target["boxes"]
        areas = target["area"]

        # Compute expected areas
        expected_areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

        assert torch.allclose(areas, expected_areas)

    def test_multiple_indices(self, mock_visdrone_dataset):
        """Test accessing multiple dataset indices."""
        dataset = VisDroneDataset(
            image_dir=str(mock_visdrone_dataset["image_dir"]),
            annotation_dir=str(mock_visdrone_dataset["annotation_dir"]),
        )

        # Access all images
        for i in range(len(dataset)):
            image, target = dataset[i]
            assert isinstance(image, torch.Tensor)
            assert isinstance(target, dict)

    def test_image_tensor_range(self, mock_visdrone_dataset):
        """Test that image tensor values are in [0, 1] range."""
        dataset = VisDroneDataset(
            image_dir=str(mock_visdrone_dataset["image_dir"]),
            annotation_dir=str(mock_visdrone_dataset["annotation_dir"]),
        )

        image, _ = dataset[0]

        assert torch.all(image >= 0)
        assert torch.all(image <= 1)
