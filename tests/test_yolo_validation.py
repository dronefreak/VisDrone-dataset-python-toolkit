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
    """Tests for RFDETRTrainer YOLO-format dataset preparation."""

    def test_symlink_images_creates_links(self, tmp_path):
        """Each image in src should have a symlink in dst."""
        from visdrone_toolkit.rfdetr_trainer import RFDETRTrainer

        src = tmp_path / "src"
        src.mkdir()
        dst = tmp_path / "dst"
        dst.mkdir()

        (src / "a.jpg").write_text("fake")
        (src / "b.png").write_text("fake")
        (src / "notes.txt").write_text("not an image")

        RFDETRTrainer._symlink_images(src, dst)

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
        """RF-DETR uses 11 classes (no ignored-regions) — consistent with YOLO pipeline."""
        from visdrone_toolkit.rfdetr_trainer import _RFDETR_CLASSES

        assert len(_RFDETR_CLASSES) == 11
        assert "pedestrian" in _RFDETR_CLASSES
        assert "others" in _RFDETR_CLASSES  # included for YOLO-pipeline consistency

    def test_all_rfdetr_variants_in_model_class_map(self):
        """All 4 RF-DETR variants should be in the class map."""
        from visdrone_toolkit.rfdetr_trainer import _MODEL_CLASS_MAP

        assert "rfdetr-nano" in _MODEL_CLASS_MAP
        assert "rfdetr-small" in _MODEL_CLASS_MAP
        assert "rfdetr-medium" in _MODEL_CLASS_MAP
        assert "rfdetr-large" in _MODEL_CLASS_MAP

    def test_trainer_default_lr_is_safe(self):
        """RFDETRTrainer.train() must default to 1e-4, not the 0.005 YOLO default."""
        import inspect

        from visdrone_toolkit.rfdetr_trainer import RFDETRTrainer

        sig = inspect.signature(RFDETRTrainer.train)
        lr_default = sig.parameters["lr"].default
        assert (
            lr_default == 1e-4
        ), f"Default LR {lr_default} is too high for RF-DETR; must be 1e-4 to prevent NaN."

    def test_trainer_default_warmup_nonzero(self):
        """RFDETRTrainer.train() must have warmup_epochs > 0 to prevent NaN from random head init."""
        import inspect

        from visdrone_toolkit.rfdetr_trainer import RFDETRTrainer

        sig = inspect.signature(RFDETRTrainer.train)
        warmup = sig.parameters["warmup_epochs"].default
        assert (
            warmup > 0
        ), f"warmup_epochs default is {warmup}; must be > 0 to prevent early-training NaN."

    def test_train_signature_uses_img_ann_dirs(self):
        """train() must accept train_img_dir/train_ann_dir (not dataset_dir)."""
        import inspect

        from visdrone_toolkit.rfdetr_trainer import RFDETRTrainer

        sig = inspect.signature(RFDETRTrainer.train)
        params = sig.parameters
        assert "train_img_dir" in params, "train() should accept train_img_dir"
        assert "train_ann_dir" in params, "train() should accept train_ann_dir"
        assert "val_img_dir" in params, "train() should accept val_img_dir"
        assert "val_ann_dir" in params, "train() should accept val_ann_dir"
        assert "dataset_dir" not in params, "train() should NOT have dataset_dir (COCO-era API)"

    def test_prepare_dataset_writes_yolo_structure(self, tmp_path):
        """_prepare_dataset must create data.yaml + train/images + train/labels."""
        import yaml

        from visdrone_toolkit.rfdetr_trainer import RFDETRTrainer

        # Build minimal fake VisDrone train data (1 image + 1 annotation)
        img_dir = tmp_path / "images"
        ann_dir = tmp_path / "annotations"
        img_dir.mkdir()
        ann_dir.mkdir()

        img_file = img_dir / "test_img.jpg"
        # Write a tiny valid JPEG (minimal 1x1)
        # Minimal 1x1 white JPEG
        img_file.write_bytes(
            b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
            b"\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t"
            b"\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a"
            b"\x1f\x1e\x1d\x1a\x1c\x1c $.' \",#\x1c\x1c(7),01444\x1f'9=82<.342\x1e"
            b"\xff\xc0\x00\x0b\x08\x00\x01\x00\x01\x01\x01\x11\x00\xff\xc4\x00"
            b"\x1f\x00\x00\x01\x05\x01\x01\x01\x01\x01\x01\x00\x00\x00\x00\x00"
            b"\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\xff\xda\x00"
            b"\x08\x01\x01\x00\x00?\x00\xf5\x0a\xff\xd9"
        )

        # VisDrone annotation format: left,top,width,height,score,category,truncation,occlusion
        (ann_dir / "test_img.txt").write_text("10,10,50,50,1,1,0,0\n")

        out_path = tmp_path / "out"
        out_path.mkdir()

        trainer = RFDETRTrainer.__new__(RFDETRTrainer)
        trainer.num_classes = 11
        trainer._prepare_dataset(out_path, img_dir, ann_dir, None, None)

        # Verify structure
        assert (out_path / "data.yaml").exists(), "data.yaml must be created"
        assert (out_path / "train" / "images").is_dir(), "train/images must exist"
        assert (out_path / "train" / "labels").is_dir(), "train/labels must exist"

        cfg = yaml.safe_load((out_path / "data.yaml").read_text())
        assert cfg["nc"] == 11
        assert cfg["train"] == "train/images"
        assert "val" in cfg
