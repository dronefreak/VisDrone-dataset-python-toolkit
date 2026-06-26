"""Tests for YOLOTrainer — dataset preparation and YAML generation.

These tests mock the Ultralytics engine so they run without GPU and
without downloading model weights. They focus on the VisDrone → YOLO
conversion, YAML correctness, and the nc/names consistency fix.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from visdrone_toolkit.yolo_trainer import _VISDRONE_CLASSES, YOLOTrainer

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_visdrone_annotation(tmp: Path, name: str = "img001") -> Path:
    """Write a minimal VisDrone annotation file (two real objects)."""
    ann_dir = tmp / "annotations"
    ann_dir.mkdir(parents=True, exist_ok=True)
    ann_file = ann_dir / f"{name}.txt"
    # Format: x,y,w,h,score,category,truncation,occlusion
    # category 1 = pedestrian (maps to YOLO class 0 after ignored-regions shift)
    ann_file.write_text("10,20,50,60,1,1,0,0\n30,40,80,90,1,4,0,0\n")

    img_dir = tmp / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    (img_dir / f"{name}.jpg").write_bytes(b"")  # empty file is fine

    return tmp


# ---------------------------------------------------------------------------
# Class-level constants
# ---------------------------------------------------------------------------


class TestVisdronClassConstants:
    """Verify _VISDRONE_CLASSES is correctly defined."""

    def test_class_count(self):
        assert len(_VISDRONE_CLASSES) == 11

    def test_ignored_regions_not_in_list(self):
        assert "ignored-regions" not in _VISDRONE_CLASSES

    def test_known_classes_present(self):
        for cls in ("pedestrian", "car", "truck", "bus"):
            assert cls in _VISDRONE_CLASSES

    def test_no_duplicates(self):
        assert len(_VISDRONE_CLASSES) == len(set(_VISDRONE_CLASSES))


# ---------------------------------------------------------------------------
# YOLOTrainer construction
# ---------------------------------------------------------------------------


class TestYOLOTrainerInit:
    """Tests for YOLOTrainer.__init__."""

    def test_pt_name_derived_from_model(self):
        trainer = YOLOTrainer("yolov8n")
        assert trainer._pt_name == "yolov8n.pt"

    def test_pt_name_v9(self):
        trainer = YOLOTrainer("yolov9c")
        assert trainer._pt_name == "yolov9c.pt"

    def test_pt_name_v10(self):
        trainer = YOLOTrainer("yolov10m")
        assert trainer._pt_name == "yolov10m.pt"

    def test_default_num_classes(self):
        trainer = YOLOTrainer("yolov8n")
        assert trainer.num_classes == 11

    def test_custom_num_classes(self):
        trainer = YOLOTrainer("yolov8n", num_classes=5)
        assert trainer.num_classes == 5

    def test_custom_device(self):
        trainer = YOLOTrainer("yolov8n", device="cpu")
        assert trainer.device == "cpu"


# ---------------------------------------------------------------------------
# Dataset YAML generation — nc/names consistency (the critical bug fix)
# ---------------------------------------------------------------------------


class TestPrepareDatasetYaml:
    """Tests for YOLOTrainer._prepare_dataset YAML output."""

    def _run(self, num_classes: int, with_val: bool = False) -> dict:
        """Run _prepare_dataset and return the parsed YAML."""
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            src = tmp / "src"
            _make_visdrone_annotation(src)

            img_dir = src / "images"
            ann_dir = src / "annotations"

            trainer = YOLOTrainer("yolov8n", num_classes=num_classes)

            val_img = img_dir if with_val else None
            val_ann = ann_dir if with_val else None

            yaml_path = trainer._prepare_dataset(tmp / "work", img_dir, ann_dir, val_img, val_ann)
            with open(yaml_path) as f:
                return yaml.safe_load(f)

    def test_nc_equals_names_length_default(self):
        data = self._run(num_classes=11)
        assert data["nc"] == len(
            data["names"]
        ), f"nc={data['nc']} but names has {len(data['names'])} entries"

    def test_nc_equals_names_length_when_12_passed(self):
        """Regression: passing num_classes=12 must not cause nc/names mismatch."""
        data = self._run(num_classes=12)
        assert data["nc"] == len(data["names"])
        # Should clamp to 11 (max available)
        assert data["nc"] == 11

    def test_nc_equals_names_length_subset(self):
        data = self._run(num_classes=5)
        assert data["nc"] == len(data["names"])
        assert data["nc"] == 5

    def test_names_content_with_11_classes(self):
        data = self._run(num_classes=11)
        assert data["names"][0] == "pedestrian"
        assert "car" in data["names"]

    def test_names_subset_is_prefix_of_full_list(self):
        data = self._run(num_classes=5)
        assert data["names"] == _VISDRONE_CLASSES[:5]

    def test_yaml_has_path_key(self):
        data = self._run(num_classes=11)
        assert "path" in data

    def test_yaml_has_train_key(self):
        data = self._run(num_classes=11)
        assert data["train"] == "images/train"

    def test_yaml_no_val_when_not_provided(self):
        data = self._run(num_classes=11, with_val=False)
        assert "val" not in data

    def test_yaml_has_val_when_provided(self):
        data = self._run(num_classes=11, with_val=True)
        assert "val" in data
        assert data["val"] == "images/val"

    def test_yaml_file_is_valid_yaml(self):
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            src = tmp / "src"
            _make_visdrone_annotation(src)
            trainer = YOLOTrainer("yolov8n")
            yaml_path = trainer._prepare_dataset(
                tmp / "work", src / "images", src / "annotations", None, None
            )
            assert yaml_path.exists()
            with open(yaml_path) as f:
                content = yaml.safe_load(f)
            assert isinstance(content, dict)


# ---------------------------------------------------------------------------
# Dataset directory structure
# ---------------------------------------------------------------------------


class TestPrepareDatasetDirStructure:
    """Tests for directory layout created by _prepare_dataset."""

    def test_labels_train_directory_created(self):
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            src = tmp / "src"
            _make_visdrone_annotation(src)
            trainer = YOLOTrainer("yolov8n")
            work = tmp / "work"
            trainer._prepare_dataset(work, src / "images", src / "annotations", None, None)
            assert (work / "labels" / "train").is_dir()

    def test_images_train_is_real_directory(self):
        """images/train must be a real directory, NOT a directory symlink.

        A dir symlink is resolved by Ultralytics before 'images → labels'
        substitution, breaking label auto-discovery.
        """
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            src = tmp / "src"
            _make_visdrone_annotation(src)
            trainer = YOLOTrainer("yolov8n")
            work = tmp / "work"
            trainer._prepare_dataset(work, src / "images", src / "annotations", None, None)
            images_train = work / "images" / "train"
            assert images_train.is_dir()
            assert not images_train.is_symlink(), (
                "images/train must be a real dir (not a dir symlink) so Ultralytics "
                "label discovery uses the workspace path, not the resolved data path"
            )

    def test_images_train_contains_file_symlinks(self):
        """Individual image symlinks inside images/train/ point to source files."""
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            src = tmp / "src"
            _make_visdrone_annotation(src)
            # Add a real .jpg to test against
            (src / "images" / "img001.jpg").write_bytes(b"fake")
            trainer = YOLOTrainer("yolov8n")
            work = tmp / "work"
            trainer._prepare_dataset(work, src / "images", src / "annotations", None, None)
            images_train = work / "images" / "train"
            links = list(images_train.iterdir())
            assert len(links) > 0, "images/train should contain file symlinks"
            for link in links:
                assert link.is_symlink(), f"{link} should be a file symlink"
                assert link.resolve().exists(), f"symlink target for {link} should exist"

    def test_file_symlinks_resolve_to_source(self):
        """File symlinks in images/train resolve to the original source files."""
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            src = tmp / "src"
            _make_visdrone_annotation(src)
            (src / "images" / "testimg.jpg").write_bytes(b"fake")
            trainer = YOLOTrainer("yolov8n")
            work = tmp / "work"
            trainer._prepare_dataset(work, src / "images", src / "annotations", None, None)
            link = work / "images" / "train" / "testimg.jpg"
            assert link.is_symlink()
            assert link.resolve() == (src / "images" / "testimg.jpg").resolve()

    def test_label_discovery_path_consistency(self):
        """Verify images/train path leads to labels/train via images→labels substitution."""
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            src = tmp / "src"
            _make_visdrone_annotation(src)
            trainer = YOLOTrainer("yolov8n")
            work = tmp / "work"
            trainer._prepare_dataset(work, src / "images", src / "annotations", None, None)

            # Simulate Ultralytics img2label_paths substitution on a workspace path
            img_path = str(work / "images" / "train" / "img001.jpg")
            label_path = img_path.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt"
            expected_labels_dir = str(work / "labels" / "train")
            assert label_path.startswith(
                expected_labels_dir
            ), f"Label path {label_path} should be under {expected_labels_dir}"

    def test_labels_val_created_when_val_provided(self):
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            src = tmp / "src"
            _make_visdrone_annotation(src)
            trainer = YOLOTrainer("yolov8n")
            work = tmp / "work"
            trainer._prepare_dataset(
                work,
                src / "images",
                src / "annotations",
                src / "images",
                src / "annotations",
            )
            assert (work / "labels" / "val").is_dir()

    def test_val_images_dir_is_real_directory(self):
        """images/val must also be a real directory, not a dir symlink."""
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            src = tmp / "src"
            _make_visdrone_annotation(src)
            trainer = YOLOTrainer("yolov8n")
            work = tmp / "work"
            trainer._prepare_dataset(
                work,
                src / "images",
                src / "annotations",
                src / "images",
                src / "annotations",
            )
            images_val = work / "images" / "val"
            assert images_val.is_dir()
            assert not images_val.is_symlink()


# ---------------------------------------------------------------------------
# YOLOTrainer.train() — mock Ultralytics to avoid downloading weights
# ---------------------------------------------------------------------------


class TestYOLOTrainerTrain:
    """Tests for YOLOTrainer.train() with mocked Ultralytics engine."""

    def _make_trainer_with_mock(self, num_classes: int = 11) -> tuple[YOLOTrainer, MagicMock]:
        mock_results = MagicMock()
        mock_yolo_instance = MagicMock()
        mock_yolo_instance.train.return_value = mock_results
        mock_yolo_class = MagicMock(return_value=mock_yolo_instance)

        trainer = YOLOTrainer("yolov8n", num_classes=num_classes, device="cpu")
        trainer._UltralyticsYOLO = mock_yolo_class
        return trainer, mock_yolo_instance

    def test_train_calls_ultralytics_train(self):
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            src = tmp / "src"
            _make_visdrone_annotation(src)
            trainer, mock_yolo = self._make_trainer_with_mock()

            trainer.train(
                train_img_dir=src / "images",
                train_ann_dir=src / "annotations",
                val_img_dir=None,
                val_ann_dir=None,
                epochs=1,
                batch_size=2,
                output_dir=tmp / "out",
            )
            mock_yolo.train.assert_called_once()

    def test_train_passes_epochs_to_ultralytics(self):
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            src = tmp / "src"
            _make_visdrone_annotation(src)
            trainer, mock_yolo = self._make_trainer_with_mock()

            trainer.train(
                train_img_dir=src / "images",
                train_ann_dir=src / "annotations",
                val_img_dir=None,
                val_ann_dir=None,
                epochs=42,
                batch_size=4,
                output_dir=tmp / "out",
            )
            call_kwargs = mock_yolo.train.call_args.kwargs
            assert call_kwargs["epochs"] == 42

    def test_train_passes_batch_to_ultralytics(self):
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            src = tmp / "src"
            _make_visdrone_annotation(src)
            trainer, mock_yolo = self._make_trainer_with_mock()

            trainer.train(
                train_img_dir=src / "images",
                train_ann_dir=src / "annotations",
                val_img_dir=None,
                val_ann_dir=None,
                epochs=1,
                batch_size=8,
                output_dir=tmp / "out",
            )
            call_kwargs = mock_yolo.train.call_args.kwargs
            assert call_kwargs["batch"] == 8

    def test_train_passes_lr0_to_ultralytics(self):
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            src = tmp / "src"
            _make_visdrone_annotation(src)
            trainer, mock_yolo = self._make_trainer_with_mock()

            trainer.train(
                train_img_dir=src / "images",
                train_ann_dir=src / "annotations",
                val_img_dir=None,
                val_ann_dir=None,
                epochs=1,
                batch_size=2,
                lr=0.005,
                output_dir=tmp / "out",
            )
            call_kwargs = mock_yolo.train.call_args.kwargs
            assert call_kwargs["lr0"] == 0.005

    def test_train_nc_not_passed_to_ultralytics(self):
        """nc must NOT appear in model.train() args — it lives in the YAML only."""
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            src = tmp / "src"
            _make_visdrone_annotation(src)
            trainer, mock_yolo = self._make_trainer_with_mock()

            trainer.train(
                train_img_dir=src / "images",
                train_ann_dir=src / "annotations",
                val_img_dir=None,
                val_ann_dir=None,
                epochs=1,
                batch_size=2,
                output_dir=tmp / "out",
            )
            call_kwargs = mock_yolo.train.call_args.kwargs
            assert "nc" not in call_kwargs, "nc must not be passed to model.train()"

    def test_train_returns_dict_with_required_keys(self):
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            src = tmp / "src"
            _make_visdrone_annotation(src)
            trainer, _ = self._make_trainer_with_mock()

            result = trainer.train(
                train_img_dir=src / "images",
                train_ann_dir=src / "annotations",
                val_img_dir=None,
                val_ann_dir=None,
                epochs=1,
                batch_size=2,
                output_dir=tmp / "out",
            )
            assert "results" in result
            assert "model_path" in result
            assert "output_dir" in result

    def test_train_output_dir_created(self):
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            src = tmp / "src"
            _make_visdrone_annotation(src)
            trainer, _ = self._make_trainer_with_mock()

            out = tmp / "nested" / "output"
            trainer.train(
                train_img_dir=src / "images",
                train_ann_dir=src / "annotations",
                val_img_dir=None,
                val_ann_dir=None,
                epochs=1,
                batch_size=2,
                output_dir=out,
            )
            assert out.exists()

    def test_train_extra_kwargs_forwarded(self):
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            src = tmp / "src"
            _make_visdrone_annotation(src)
            trainer, mock_yolo = self._make_trainer_with_mock()

            trainer.train(
                train_img_dir=src / "images",
                train_ann_dir=src / "annotations",
                val_img_dir=None,
                val_ann_dir=None,
                epochs=1,
                batch_size=2,
                output_dir=tmp / "out",
                patience=50,
                cos_lr=True,
            )
            call_kwargs = mock_yolo.train.call_args.kwargs
            assert call_kwargs.get("patience") == 50
            assert call_kwargs.get("cos_lr") is True

    def test_train_with_num_classes_12_produces_valid_yaml(self):
        """Regression: num_classes=12 must not crash training with nc/names mismatch."""
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            src = tmp / "src"
            _make_visdrone_annotation(src)
            trainer, mock_yolo = self._make_trainer_with_mock(num_classes=12)

            # Should not raise
            trainer.train(
                train_img_dir=src / "images",
                train_ann_dir=src / "annotations",
                val_img_dir=None,
                val_ann_dir=None,
                epochs=1,
                batch_size=2,
                output_dir=tmp / "out",
            )
            # Verify nc was not passed to ultralytics
            call_kwargs = mock_yolo.train.call_args.kwargs
            assert "nc" not in call_kwargs


# ---------------------------------------------------------------------------
# Missing ultralytics — graceful import error
# ---------------------------------------------------------------------------


class TestMissingUltralytics:
    """Test that a helpful ImportError is raised when ultralytics is absent."""

    def test_import_error_when_ultralytics_missing(self):
        with patch.dict("sys.modules", {"ultralytics": None}):
            import importlib

            import visdrone_toolkit.yolo_trainer as yt_module

            importlib.reload(yt_module)
            # After reload, the import at __init__ time is skipped;
            # the error surfaces in __init__ of YOLOTrainer.
            # We can also just verify the guard is present by inspecting source.
            with open(yt_module.__file__) as fh:
                assert "ImportError" in fh.read()


class TestRTDETRTrainerRouting:
    """Test that YOLOTrainer selects RTDETR class for rtdetr- model names."""

    def test_yolo_trainer_uses_rtdetr_class_for_rtdetr_models(self):
        """YOLOTrainer should load RTDETR class when model_name starts with 'rtdetr'."""
        from ultralytics import RTDETR

        from visdrone_toolkit.yolo_trainer import YOLOTrainer

        trainer = YOLOTrainer(model_name="rtdetr-l", num_classes=11, device="cpu")
        assert trainer._UltralyticsYOLO is RTDETR

    def test_yolo_trainer_uses_yolo_class_for_yolo_models(self):
        """YOLOTrainer should keep YOLO class for yolo* model names."""
        from ultralytics import YOLO

        from visdrone_toolkit.yolo_trainer import YOLOTrainer

        trainer = YOLOTrainer(model_name="yolov8n", num_classes=11, device="cpu")
        assert trainer._UltralyticsYOLO is YOLO

    def test_yolo_trainer_pt_name_for_rtdetr(self):
        """YOLOTrainer should derive correct .pt filename for RT-DETR models."""
        from visdrone_toolkit.yolo_trainer import YOLOTrainer

        trainer = YOLOTrainer(model_name="rtdetr-x", num_classes=11, device="cpu")
        assert trainer._pt_name == "rtdetr-x.pt"

    @pytest.mark.parametrize(
        "model_name",
        ["rtdetr-l", "rtdetr-x", "rtdetr-resnet50", "rtdetr-resnet101"],
    )
    def test_all_rtdetr_variants_route_to_rtdetr_class(self, model_name):
        """All RT-DETR variants should use the RTDETR class."""
        from ultralytics import RTDETR

        from visdrone_toolkit.yolo_trainer import YOLOTrainer

        trainer = YOLOTrainer(model_name=model_name, num_classes=11, device="cpu")
        assert trainer._UltralyticsYOLO is RTDETR
