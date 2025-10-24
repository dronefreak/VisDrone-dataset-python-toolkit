"""
Tests for annotation converters.
"""


import pytest

from visdrone_toolkit.converters import convert_to_coco, convert_to_yolo
from visdrone_toolkit.converters.visdrone_to_coco import validate_coco_format
from visdrone_toolkit.converters.visdrone_to_yolo import validate_yolo_format


class TestCOCOConverter:
    """Tests for VisDrone to COCO format converter."""

    def test_basic_conversion(self, mock_visdrone_dataset, temp_dir):
        """Test basic COCO format conversion."""
        output_json = temp_dir / "annotations.json"

        result = convert_to_coco(
            image_dir=str(mock_visdrone_dataset["image_dir"]),
            annotation_dir=str(mock_visdrone_dataset["annotation_dir"]),
            output_json=str(output_json),
        )

        assert output_json.exists()
        assert isinstance(result, dict)
        assert "images" in result
        assert "annotations" in result
        assert "categories" in result

    def test_coco_structure(self, mock_visdrone_dataset, temp_dir):
        """Test COCO format structure."""
        output_json = temp_dir / "annotations.json"

        result = convert_to_coco(
            image_dir=str(mock_visdrone_dataset["image_dir"]),
            annotation_dir=str(mock_visdrone_dataset["annotation_dir"]),
            output_json=str(output_json),
        )

        # Check required keys
        required_keys = ["info", "licenses", "categories", "images", "annotations"]
        for key in required_keys:
            assert key in result

        # Check categories structure
        for cat in result["categories"]:
            assert "id" in cat
            assert "name" in cat
            assert "supercategory" in cat

    def test_coco_annotations_format(self, mock_visdrone_dataset, temp_dir):
        """Test COCO annotations format."""
        output_json = temp_dir / "annotations.json"

        result = convert_to_coco(
            image_dir=str(mock_visdrone_dataset["image_dir"]),
            annotation_dir=str(mock_visdrone_dataset["annotation_dir"]),
            output_json=str(output_json),
        )

        if len(result["annotations"]) > 0:
            ann = result["annotations"][0]
            assert "id" in ann
            assert "image_id" in ann
            assert "category_id" in ann
            assert "bbox" in ann
            assert "area" in ann
            assert len(ann["bbox"]) == 4  # [x, y, width, height]

    def test_filter_ignored(self, mock_visdrone_dataset, temp_dir):
        """Test filtering ignored boxes."""
        output_filtered = temp_dir / "annotations_filtered.json"
        output_unfiltered = temp_dir / "annotations_unfiltered.json"

        # With filtering
        result_filtered = convert_to_coco(
            image_dir=str(mock_visdrone_dataset["image_dir"]),
            annotation_dir=str(mock_visdrone_dataset["annotation_dir"]),
            output_json=str(output_filtered),
            filter_ignored=True,
        )

        # Without filtering
        result_unfiltered = convert_to_coco(
            image_dir=str(mock_visdrone_dataset["image_dir"]),
            annotation_dir=str(mock_visdrone_dataset["annotation_dir"]),
            output_json=str(output_unfiltered),
            filter_ignored=False,
        )

        # Filtered should have fewer or equal annotations
        assert len(result_filtered["annotations"]) <= len(result_unfiltered["annotations"])

    def test_filter_crowd(self, mock_visdrone_dataset, temp_dir):
        """Test filtering crowd regions."""
        output_filtered = temp_dir / "annotations_crowd_filtered.json"

        result = convert_to_coco(
            image_dir=str(mock_visdrone_dataset["image_dir"]),
            annotation_dir=str(mock_visdrone_dataset["annotation_dir"]),
            output_json=str(output_filtered),
            filter_crowd=True,
        )

        # Should not have category 0 (ignored-regions)
        category_ids = [ann["category_id"] for ann in result["annotations"]]
        assert 0 not in category_ids

    def test_validate_coco_format(self, mock_visdrone_dataset, temp_dir):
        """Test COCO format validation."""
        output_json = temp_dir / "annotations_valid.json"

        convert_to_coco(
            image_dir=str(mock_visdrone_dataset["image_dir"]),
            annotation_dir=str(mock_visdrone_dataset["annotation_dir"]),
            output_json=str(output_json),
        )

        is_valid = validate_coco_format(str(output_json))
        assert is_valid


class TestYOLOConverter:
    """Tests for VisDrone to YOLO format converter."""

    def test_basic_conversion(self, mock_visdrone_dataset, temp_dir):
        """Test basic YOLO format conversion."""
        output_dir = temp_dir / "yolo_labels"

        convert_to_yolo(
            image_dir=str(mock_visdrone_dataset["image_dir"]),
            annotation_dir=str(mock_visdrone_dataset["annotation_dir"]),
            output_dir=str(output_dir),
        )

        assert output_dir.exists()

        # Check that label files are created
        label_files = list(output_dir.glob("*.txt"))
        assert len(label_files) == mock_visdrone_dataset["num_images"]

    def test_yolo_format_structure(self, mock_visdrone_dataset, temp_dir):
        """Test YOLO format structure (class x y w h)."""
        output_dir = temp_dir / "yolo_labels"

        convert_to_yolo(
            image_dir=str(mock_visdrone_dataset["image_dir"]),
            annotation_dir=str(mock_visdrone_dataset["annotation_dir"]),
            output_dir=str(output_dir),
        )

        # Check a label file
        label_files = list(output_dir.glob("*.txt"))
        if len(label_files) > 0:
            with open(label_files[0]) as f:
                for line in f:
                    if line.strip():
                        parts = line.strip().split()
                        assert len(parts) == 5  # class x y w h

                        # Check values are valid
                        class_id = int(parts[0])
                        x, y, w, h = map(float, parts[1:])

                        assert class_id >= 0
                        assert 0 <= x <= 1
                        assert 0 <= y <= 1
                        assert 0 <= w <= 1
                        assert 0 <= h <= 1

    def test_yolo_yaml_creation(self, mock_visdrone_dataset, temp_dir):
        """Test dataset.yaml creation."""
        output_dir = temp_dir / "yolo_labels"

        convert_to_yolo(
            image_dir=str(mock_visdrone_dataset["image_dir"]),
            annotation_dir=str(mock_visdrone_dataset["annotation_dir"]),
            output_dir=str(output_dir),
            create_yaml=True,
        )

        yaml_path = output_dir.parent / "dataset.yaml"
        assert yaml_path.exists()

        # Check yaml content
        content = yaml_path.read_text()
        assert "nc:" in content  # number of classes
        assert "names:" in content  # class names

    def test_filter_crowd_yolo(self, mock_visdrone_dataset, temp_dir):
        """Test filtering crowd regions in YOLO format."""
        output_dir_filtered = temp_dir / "yolo_labels_filtered"
        output_dir_unfiltered = temp_dir / "yolo_labels_unfiltered"

        # Convert with filtering
        convert_to_yolo(
            image_dir=str(mock_visdrone_dataset["image_dir"]),
            annotation_dir=str(mock_visdrone_dataset["annotation_dir"]),
            output_dir=str(output_dir_filtered),
            filter_crowd=True,
            create_yaml=False,
        )

        # Convert without filtering
        convert_to_yolo(
            image_dir=str(mock_visdrone_dataset["image_dir"]),
            annotation_dir=str(mock_visdrone_dataset["annotation_dir"]),
            output_dir=str(output_dir_unfiltered),
            filter_crowd=False,
            create_yaml=False,
        )

        # Count annotations in both versions
        label_files_filtered = list(output_dir_filtered.glob("*.txt"))
        label_files_unfiltered = list(output_dir_unfiltered.glob("*.txt"))

        total_filtered = 0
        total_unfiltered = 0

        for label_file in label_files_filtered:
            with open(label_file) as f:
                total_filtered += len([line for line in f if line.strip()])

        for label_file in label_files_unfiltered:
            with open(label_file) as f:
                total_unfiltered += len([line for line in f if line.strip()])

        # Filtered should have fewer annotations (crowd regions removed)
        assert total_filtered < total_unfiltered

    def test_validate_yolo_format(self, mock_visdrone_dataset, temp_dir):
        """Test YOLO format validation."""
        output_dir = temp_dir / "yolo_labels_valid"

        convert_to_yolo(
            image_dir=str(mock_visdrone_dataset["image_dir"]),
            annotation_dir=str(mock_visdrone_dataset["annotation_dir"]),
            output_dir=str(output_dir),
        )

        is_valid = validate_yolo_format(str(output_dir))
        assert is_valid

    def test_empty_annotation_yolo(self, temp_dir, sample_image):
        """Test YOLO conversion with empty annotation."""
        img_dir = temp_dir / "images"
        ann_dir = temp_dir / "annotations"
        output_dir = temp_dir / "yolo_labels"

        img_dir.mkdir()
        ann_dir.mkdir()

        # Create image with empty annotation
        img_path = img_dir / "test.jpg"
        sample_image.save(img_path)

        ann_path = ann_dir / "test.txt"
        ann_path.write_text("")

        convert_to_yolo(
            image_dir=str(img_dir),
            annotation_dir=str(ann_dir),
            output_dir=str(output_dir),
        )

        # Should create empty label file
        label_path = output_dir / "test.txt"
        assert label_path.exists()
        assert label_path.read_text() == ""


class TestConverterEdgeCases:
    """Tests for edge cases in converters."""

    def test_missing_annotation_file(self, temp_dir, sample_image):
        """Test conversion when annotation file is missing."""
        img_dir = temp_dir / "images"
        ann_dir = temp_dir / "annotations"
        img_dir.mkdir()
        ann_dir.mkdir()

        # Create image without annotation
        img_path = img_dir / "test.jpg"
        sample_image.save(img_path)

        output_json = temp_dir / "annotations.json"

        # Should handle gracefully
        _ = convert_to_coco(
            image_dir=str(img_dir),
            annotation_dir=str(ann_dir),
            output_json=str(output_json),
        )

        assert output_json.exists()

    def test_invalid_image_directory(self, temp_dir):
        """Test error handling for invalid image directory."""
        with pytest.raises(ValueError):
            convert_to_coco(
                image_dir="/nonexistent/path",
                annotation_dir=str(temp_dir),
                output_json=str(temp_dir / "output.json"),
            )

    def test_malformed_annotation(self, temp_dir, sample_image):
        """Test handling of malformed annotation lines."""
        img_dir = temp_dir / "images"
        ann_dir = temp_dir / "annotations"
        img_dir.mkdir()
        ann_dir.mkdir()

        # Create image
        img_path = img_dir / "test.jpg"
        sample_image.save(img_path)

        # Create annotation with malformed lines
        ann_path = ann_dir / "test.txt"
        ann_path.write_text("100,100,100,100,1,1,0,0\ninvalid line\n200,200,50,50,1,4,0,0\n")

        output_json = temp_dir / "annotations.json"

        # Should handle gracefully and skip invalid lines
        result = convert_to_coco(
            image_dir=str(img_dir),
            annotation_dir=str(ann_dir),
            output_json=str(output_json),
        )

        # Should have 2 valid annotations (skipping the invalid line)
        assert len(result["annotations"]) == 2
