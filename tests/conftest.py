"""
Pytest fixtures for VisDrone toolkit tests.

Fixtures provide reusable test data and mock objects.
"""

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    tmp_dir = tempfile.mkdtemp()
    yield Path(tmp_dir)
    shutil.rmtree(tmp_dir)


@pytest.fixture
def sample_image():
    """Create a sample RGB image."""
    # Create 640x480 RGB image
    image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    return Image.fromarray(image)


@pytest.fixture
def sample_image_array():
    """Create a sample image as numpy array."""
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)


@pytest.fixture
def sample_boxes():
    """Create sample bounding boxes in [x1, y1, x2, y2] format."""
    boxes = np.array(
        [
            [100, 100, 200, 200],
            [250, 150, 350, 300],
            [400, 200, 500, 400],
        ],
        dtype=np.float32,
    )
    return boxes


@pytest.fixture
def sample_labels():
    """Create sample class labels."""
    return np.array([1, 4, 9], dtype=np.int64)


@pytest.fixture
def sample_scores():
    """Create sample confidence scores."""
    return np.array([0.95, 0.87, 0.76], dtype=np.float32)


@pytest.fixture
def visdrone_annotation_content():
    """Sample VisDrone annotation content."""
    # Format: x,y,w,h,score,category,truncation,occlusion
    return """100,100,100,100,1,1,0,0
250,150,100,150,1,4,0,1
400,200,100,200,1,9,1,0
50,50,30,30,0,0,0,0
300,300,50,50,1,0,0,0
"""


@pytest.fixture
def mock_visdrone_dataset(temp_dir, sample_image, visdrone_annotation_content):
    """Create a mock VisDrone dataset structure."""
    # Create directories
    img_dir = temp_dir / "images"
    ann_dir = temp_dir / "annotations"
    img_dir.mkdir(parents=True)
    ann_dir.mkdir(parents=True)

    # Create sample images and annotations
    for i in range(3):
        # Save image
        img_path = img_dir / f"image_{i:06d}.jpg"
        sample_image.save(img_path)

        # Save annotation
        ann_path = ann_dir / f"image_{i:06d}.txt"
        ann_path.write_text(visdrone_annotation_content)

    return {
        "image_dir": img_dir,
        "annotation_dir": ann_dir,
        "num_images": 3,
    }


@pytest.fixture
def sample_target():
    """Create a sample target dictionary for object detection."""
    return {
        "boxes": torch.tensor([[100, 100, 200, 200], [250, 150, 350, 300]], dtype=torch.float32),
        "labels": torch.tensor([1, 4], dtype=torch.int64),
        "image_id": torch.tensor([0]),
        "area": torch.tensor([10000.0, 15000.0], dtype=torch.float32),
        "iscrowd": torch.tensor([0, 0], dtype=torch.int64),
    }


@pytest.fixture
def sample_prediction():
    """Create a sample prediction dictionary."""
    return {
        "boxes": torch.tensor([[95, 95, 205, 205], [240, 145, 360, 310]], dtype=torch.float32),
        "labels": torch.tensor([1, 4], dtype=torch.int64),
        "scores": torch.tensor([0.95, 0.87], dtype=torch.float32),
    }


@pytest.fixture
def device():
    """Get available device (CPU for testing)."""
    return torch.device("cpu")


@pytest.fixture
def sample_coco_json():
    """Sample COCO format JSON structure."""
    return {
        "info": {
            "description": "Test dataset",
            "version": "1.0",
            "year": 2025,
        },
        "licenses": [],
        "categories": [
            {"id": 1, "name": "pedestrian", "supercategory": "person"},
            {"id": 4, "name": "car", "supercategory": "vehicle"},
        ],
        "images": [
            {
                "id": 1,
                "file_name": "image_000000.jpg",
                "width": 640,
                "height": 480,
            }
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [100, 100, 100, 100],
                "area": 10000,
                "iscrowd": 0,
            }
        ],
    }


@pytest.fixture
def num_classes():
    """Number of classes in VisDrone dataset."""
    return 12


@pytest.fixture
def class_names():
    """VisDrone class names."""
    return [
        "ignored-regions",
        "pedestrian",
        "people",
        "bicycle",
        "car",
        "van",
        "truck",
        "tricycle",
        "awning-tricycle",
        "bus",
        "motor",
        "others",
    ]
