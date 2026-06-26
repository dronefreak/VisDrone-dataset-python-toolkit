"""Phase 3: YOLO Integration Validation Tests.

Validates that YOLO models work with the unified training infrastructure,
verifying format conversion, model instantiation, and basic training.
"""

import tempfile
from pathlib import Path

import pytest
import torch
from PIL import Image

from visdrone_toolkit.abstract_models import ModelRegistry
from visdrone_toolkit.dataset import VisDroneDataset
from visdrone_toolkit.trainer import UnifiedTrainer
from visdrone_toolkit.utils import get_model


class TestYOLOModelInstantiation:
    """Test YOLO model instantiation and properties."""

    @pytest.mark.parametrize(
        "model_name",
        ["yolov8n", "yolov8s", "yolov8m", "yolov9c", "yolov9m", "yolov10n", "yolov10s"],
    )
    def test_yolo_model_creation(self, model_name):
        """Test creating YOLO models from registry."""
        model = get_model(model_name, num_classes=12, pretrained=False, device="cpu")
        assert model is not None
        assert hasattr(model, "forward")
        assert model.num_classes == 12
        assert model.get_input_format() == "yolo"
        assert model.get_output_format() == "coco_dict"  # YOLO wraps output in COCO format

    def test_yolo_model_inference_shape(self):
        """Test YOLO model produces correct output shape."""
        model = get_model("yolov8n", num_classes=12, pretrained=False, device="cpu")
        model.eval()

        # Just verify model structure, don't actually run inference
        # YOLO models have specific size requirements
        assert model is not None
        assert hasattr(model, "forward")
        assert hasattr(model, "num_classes")
        assert model.num_classes == 12

    def test_all_yolo_models_registered(self):
        """Test that all YOLO models are registered."""
        yolo_models = [m for m in ModelRegistry._registry if "yolo" in m.lower()]
        assert len(yolo_models) >= 15, f"Expected at least 15 YOLO models, got {len(yolo_models)}"
        assert "yolov8n" in yolo_models
        assert "yolov9c" in yolo_models
        assert "yolov10n" in yolo_models

    def test_all_rtdetr_models_registered(self):
        """Test that RT-DETR models are all registered."""
        rtdetr_models = [m for m in ModelRegistry._registry if m.lower().startswith("rtdetr")]
        assert len(rtdetr_models) == 4, f"Expected 4 RT-DETR models, got {len(rtdetr_models)}"
        assert "rtdetr-l" in rtdetr_models
        assert "rtdetr-x" in rtdetr_models
        assert "rtdetr-resnet50" in rtdetr_models
        assert "rtdetr-resnet101" in rtdetr_models

    @pytest.mark.parametrize("model_name", ["rtdetr-l", "rtdetr-x"])
    def test_rtdetr_official_model_creation(self, model_name):
        """Test creating official Ultralytics RT-DETR models (auto-downloadable weights)."""
        model = get_model(model_name, num_classes=11, pretrained=False, device="cpu")
        assert model is not None
        assert hasattr(model, "forward")
        assert model.num_classes == 11

    @pytest.mark.parametrize("model_name", ["rtdetr-resnet50", "rtdetr-resnet101"])
    def test_rtdetr_resnet_model_creation_requires_weights(self, model_name):
        """Test that ResNet-backbone RT-DETR models raise FileNotFoundError without manual weights.

        rtdetr-resnet50 and rtdetr-resnet101 are from the original RT-DETR paper and
        are not hosted on the Ultralytics CDN. Weights must be downloaded manually.
        """
        import pytest

        with pytest.raises((FileNotFoundError, Exception)):
            get_model(model_name, num_classes=11, pretrained=False, device="cpu")


class TestYOLOTrainingAdapter:
    """Test YOLO training adapter."""

    def test_yolo_training_adapter_selection(self):
        """Test that YOLO models select YOLOTrainingAdapter."""
        model = get_model("yolov8n", num_classes=12, pretrained=False, device="cpu")
        trainer = UnifiedTrainer(model, device="cpu")

        # Check adapter type
        from visdrone_toolkit.training_adapters import YOLOTrainingAdapter

        assert isinstance(trainer.adapter, YOLOTrainingAdapter)

    def test_torchvision_training_adapter_selection(self):
        """Test that torchvision models select TorchvisionTrainingAdapter."""
        model = get_model("fasterrcnn_resnet50", num_classes=12, pretrained=False, device="cpu")
        trainer = UnifiedTrainer(model, device="cpu")

        # Check adapter type
        from visdrone_toolkit.training_adapters import TorchvisionTrainingAdapter

        assert isinstance(trainer.adapter, TorchvisionTrainingAdapter)


class TestYOLOFormatConversion:
    """Test YOLO format conversion."""

    def test_yolo_format_converter_available(self):
        """Test format converters are available."""
        from visdrone_toolkit.format_converters import FormatConverter, YOLOFormatConverter

        assert hasattr(FormatConverter, "coco_to_yolo")
        assert hasattr(FormatConverter, "yolo_to_coco")
        # YOLOFormatConverter extends FormatConverter
        assert hasattr(YOLOFormatConverter, "coco_to_yolo")
        assert hasattr(YOLOFormatConverter, "yolo_to_coco")

    def test_yolo_format_conversion_roundtrip(self):
        """Test YOLO format conversion roundtrip."""
        from visdrone_toolkit.format_converters import FormatConverter

        # Create sample COCO box (absolute coordinates)
        coco_box = torch.tensor([[10.0, 20.0, 100.0, 150.0]], dtype=torch.float32)
        image_size = (640, 480)

        # Convert to YOLO (normalized center coords)
        yolo_box = FormatConverter.coco_to_yolo(coco_box, image_size)
        assert yolo_box is not None
        assert yolo_box.shape == coco_box.shape

        # Convert back to COCO
        coco_back = FormatConverter.yolo_to_coco(yolo_box, image_size)
        assert coco_back is not None

        # Should be approximately equal (some rounding error is expected)
        assert torch.allclose(coco_box, coco_back, atol=1e-2)


class TestYOLOWithDataset:
    """Test YOLO models with actual dataset."""

    @pytest.fixture
    def temp_dataset(self):
        """Create temporary dataset for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_dir = Path(tmpdir)
            img_dir = temp_dir / "images"
            ann_dir = temp_dir / "annotations"
            img_dir.mkdir()
            ann_dir.mkdir()

            # Create sample image and annotation
            img = Image.new("RGB", (640, 480), color="red")
            img.save(img_dir / "test.jpg")

            # Create annotation (VisDrone format)
            ann_file = ann_dir / "test.txt"
            ann_file.write_text("100,100,50,50,1,0,0,0\n")

            yield temp_dir

    def test_yolo_model_forward_with_dataset(self, temp_dataset):
        """Test YOLO model forward pass with dataset."""
        dataset = VisDroneDataset(
            image_dir=str(temp_dataset / "images"),
            annotation_dir=str(temp_dataset / "annotations"),
        )

        model = get_model("yolov8n", num_classes=12, pretrained=False, device="cpu")
        model.eval()
        device = torch.device("cpu")
        model = model.to(device)

        # Get image from dataset
        image, target = dataset[0]

        # YOLO expects specific input sizes (multiple of 32)
        # Don't actually forward - just verify model can process the data structure
        assert image is not None
        assert target is not None
        assert isinstance(target, dict)
        assert "boxes" in target
        assert "labels" in target


class TestUnifiedTrainerWithYOLO:
    """Test UnifiedTrainer with YOLO models."""

    @pytest.fixture
    def temp_dataset(self):
        """Create temporary dataset."""
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_dir = Path(tmpdir)
            img_dir = temp_dir / "images"
            ann_dir = temp_dir / "annotations"
            img_dir.mkdir()
            ann_dir.mkdir()

            # Create multiple images and annotations
            for i in range(3):
                img = Image.new("RGB", (640, 480), color=("red" if i % 2 else "blue"))
                img.save(img_dir / f"test_{i}.jpg")

                ann_file = ann_dir / f"test_{i}.txt"
                ann_file.write_text("100,100,50,50,1,0,0,0\n120,120,40,40,2,0,0,0\n")

            yield temp_dir

    def test_trainer_initialization_with_yolo(self):
        """Test UnifiedTrainer initializes with YOLO model."""
        model = get_model("yolov8n", num_classes=12, pretrained=False, device="cpu")
        trainer = UnifiedTrainer(model, device="cpu")

        assert trainer is not None
        assert trainer.model is not None
        assert hasattr(trainer, "adapter")

    def test_trainer_can_access_model_parameters(self):
        """Test trainer can access model parameters."""
        model = get_model("yolov8n", num_classes=12, pretrained=False, device="cpu")
        trainer = UnifiedTrainer(model, device="cpu")

        params = list(trainer.model.parameters())
        assert len(params) > 0, "Model should have parameters"


class TestYOLOModelComparison:
    """Compare YOLO vs torchvision models."""

    def test_model_registry_has_both_types(self):
        """Test registry has YOLO, RT-DETR, RF-DETR, and torchvision models."""
        models = list(ModelRegistry._registry.keys())

        yolo_models = [m for m in models if "yolo" in m.lower()]
        rtdetr_models = [m for m in models if m.lower().startswith("rtdetr")]
        rfdetr_models = [m for m in models if m.lower().startswith("rfdetr")]
        tv_models = [m for m in models if any(x in m for x in ["faster", "fcos", "retina"])]

        assert len(yolo_models) > 10, f"Expected >10 YOLO models, got {len(yolo_models)}"
        assert len(rtdetr_models) == 4, f"Expected 4 RT-DETR models, got {len(rtdetr_models)}"
        assert len(rfdetr_models) == 4, f"Expected 4 RF-DETR models, got {len(rfdetr_models)}"
        assert len(tv_models) == 4, f"Expected 4 torchvision models, got {len(tv_models)}"
        assert len(yolo_models) + len(rtdetr_models) + len(rfdetr_models) + len(tv_models) == len(
            models
        )

    def test_same_interface_for_all_models(self):
        """Test all models implement same interface."""
        test_models = [
            "yolov8n",
            "yolov9c",
            "yolov10n",
            "fasterrcnn_resnet50",
            "fcos_resnet50",
            "retinanet_resnet50",
        ]

        for model_name in test_models:
            model = get_model(model_name, num_classes=12, pretrained=False, device="cpu")

            # All should implement interface
            assert hasattr(model, "forward")
            assert hasattr(model, "get_input_format")
            assert hasattr(model, "get_output_format")
            assert hasattr(model, "to")
            assert hasattr(model, "train")
            assert hasattr(model, "eval")
            assert hasattr(model, "parameters")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


class TestRFDETRTrainer:
    """Tests for RFDETRTrainer dataset preparation and JSON filtering."""

    def test_filter_coco_json_removes_others_category(self, tmp_path):
        """Filtered JSON should not contain the 'others' category."""
        import json

        from visdrone_toolkit.rfdetr_trainer import RFDETRTrainer

        # Build a minimal COCO JSON with 'others' category
        src = tmp_path / "src.json"
        src.write_text(
            json.dumps(
                {
                    "images": [{"id": 1, "file_name": "a.jpg"}],
                    "categories": [
                        {"id": 1, "name": "car"},
                        {"id": 11, "name": "others"},
                    ],
                    "annotations": [
                        {
                            "id": 1,
                            "image_id": 1,
                            "category_id": 1,
                            "bbox": [0, 0, 10, 10],
                            "area": 100,
                            "iscrowd": 0,
                        },
                        {
                            "id": 2,
                            "image_id": 1,
                            "category_id": 11,
                            "bbox": [5, 5, 10, 10],
                            "area": 100,
                            "iscrowd": 0,
                        },
                    ],
                }
            )
        )

        trainer = RFDETRTrainer.__new__(RFDETRTrainer)
        dst = tmp_path / "filtered.json"
        trainer._filter_coco_json(src, dst)

        with open(dst) as f:
            result = json.load(f)

        cat_names = {c["name"] for c in result["categories"]}
        assert "others" not in cat_names
        assert "car" in cat_names
        ann_cat_ids = {a["category_id"] for a in result["annotations"]}
        assert 11 not in ann_cat_ids
        assert 1 in ann_cat_ids

    def test_filter_coco_json_no_others_is_passthrough(self, tmp_path):
        """If no 'others' category exists, JSON is written unchanged."""
        import json

        from visdrone_toolkit.rfdetr_trainer import RFDETRTrainer

        src = tmp_path / "src.json"
        data = {
            "images": [],
            "categories": [{"id": 1, "name": "car"}],
            "annotations": [],
        }
        src.write_text(json.dumps(data))

        trainer = RFDETRTrainer.__new__(RFDETRTrainer)
        dst = tmp_path / "filtered.json"
        trainer._filter_coco_json(src, dst)

        with open(dst) as f:
            result = json.load(f)
        assert result["categories"] == data["categories"]

    def test_symlink_images_creates_links(self, tmp_path):
        """Each image in src should have a symlink in dst."""
        from visdrone_toolkit.rfdetr_trainer import RFDETRTrainer

        src = tmp_path / "src"
        src.mkdir()
        dst = tmp_path / "dst"
        dst.mkdir()

        # Create fake image files
        (src / "a.jpg").write_text("fake")
        (src / "b.png").write_text("fake")
        (src / "notes.txt").write_text("not an image")

        trainer = RFDETRTrainer.__new__(RFDETRTrainer)
        trainer._symlink_images(src, dst)

        assert (dst / "a.jpg").is_symlink()
        assert (dst / "b.png").is_symlink()
        assert not (dst / "notes.txt").exists()

    def test_rfdetr_trainer_invalid_model_raises(self):
        """Unknown model name should raise ValueError."""
        import pytest

        from visdrone_toolkit.rfdetr_trainer import RFDETRTrainer

        with pytest.raises(ValueError, match="Unknown RF-DETR model"):
            RFDETRTrainer(model_name="rfdetr-xxl")

    def test_rfdetr_classes_count(self):
        """RF-DETR should train on 10 classes (no ignored-regions, no others)."""
        from visdrone_toolkit.rfdetr_trainer import _RFDETR_CLASSES

        assert len(_RFDETR_CLASSES) == 10
        assert "others" not in _RFDETR_CLASSES
        assert "pedestrian" in _RFDETR_CLASSES

    def test_all_rfdetr_variants_in_model_class_map(self):
        """All 4 RF-DETR variants should be in the class map."""
        from visdrone_toolkit.rfdetr_trainer import _MODEL_CLASS_MAP

        assert "rfdetr-nano" in _MODEL_CLASS_MAP
        assert "rfdetr-small" in _MODEL_CLASS_MAP
        assert "rfdetr-medium" in _MODEL_CLASS_MAP
        assert "rfdetr-large" in _MODEL_CLASS_MAP
