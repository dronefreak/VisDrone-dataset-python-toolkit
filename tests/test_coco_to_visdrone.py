"""
Tests for COCO to VisDrone converter.
"""

import json
from pathlib import Path

import pytest

from visdrone_toolkit.converters.coco_to_visdrone import coco_to_visdrone, validate_visdrone_format


@pytest.fixture
def coco_json_file(tmp_path: Path) -> Path:
    """Create a sample COCO JSON file for testing."""
    coco_data = {
        "images": [
            {"id": 1, "file_name": "image1.jpg", "width": 640, "height": 480},
            {"id": 2, "file_name": "image2.jpg", "width": 640, "height": 480},
        ],
        "annotations": [
            {"id": 1, "image_id": 1, "bbox": [10, 20, 30, 40], "category_id": 1},
            {"id": 2, "image_id": 1, "bbox": [50, 60, 70, 80], "category_id": 3},
            {"id": 3, "image_id": 2, "bbox": [100, 110, 120, 130], "category_id": 5},
        ],
        "categories": [
            {"id": 1, "name": "pedestrian"},
            {"id": 3, "name": "car"},
            {"id": 5, "name": "truck"},
        ],
    }

    json_path = tmp_path / "coco.json"
    with open(json_path, "w") as f:
        json.dump(coco_data, f)

    return json_path


class TestCOCOToVisDrone:
    """Tests for COCO to VisDrone converter."""

    def test_creates_files(self, coco_json_file: Path, tmp_path: Path) -> None:
        """Test that converter creates VisDrone annotation files."""
        output_dir = tmp_path / "output"

        stats = coco_to_visdrone(coco_json_file, output_dir)

        assert stats["num_images"] == 2
        assert stats["num_annotations"] == 3
        assert (output_dir / "image1.txt").exists()
        assert (output_dir / "image2.txt").exists()

    def test_output_format(self, coco_json_file: Path, tmp_path: Path) -> None:
        """Test that output format is correct VisDrone format."""
        output_dir = tmp_path / "output"

        coco_to_visdrone(coco_json_file, output_dir)

        with open(output_dir / "image1.txt") as f:
            lines = f.readlines()

        # Format: x, y, w, h, confidence, class_id, truncation, occlusion
        parts = lines[0].strip().split(",")
        assert len(parts) == 8
        assert parts[5] == "0"  # pedestrian class ID

    def test_skips_unknown_classes(self, coco_json_file: Path, tmp_path: Path) -> None:
        """Test that unknown classes are skipped."""
        output_dir = tmp_path / "output"

        # Add unknown class annotation
        with open(coco_json_file) as f:
            data = json.load(f)

        data["annotations"].append(
            {"id": 4, "image_id": 1, "bbox": [1, 2, 3, 4], "category_id": 999}
        )

        with open(coco_json_file, "w") as f:
            json.dump(data, f)

        stats = coco_to_visdrone(coco_json_file, output_dir)
        assert stats["num_skipped"] == 1

    def test_validate_format(self, coco_json_file: Path, tmp_path: Path) -> None:
        """Test VisDrone format validation."""
        output_dir = tmp_path / "output"

        coco_to_visdrone(coco_json_file, output_dir)

        is_valid = validate_visdrone_format(output_dir)
        assert is_valid

    def test_empty_annotations(self, tmp_path: Path) -> None:
        """Test conversion with empty annotations."""
        coco_data = {
            "images": [{"id": 1, "file_name": "image1.jpg", "width": 640, "height": 480}],
            "annotations": [],
            "categories": [{"id": 1, "name": "pedestrian"}],
        }

        json_path = tmp_path / "coco.json"
        with open(json_path, "w") as f:
            json.dump(coco_data, f)

        output_dir = tmp_path / "output"
        stats = coco_to_visdrone(json_path, output_dir)

        assert stats["num_images"] == 1
        assert stats["num_annotations"] == 0
