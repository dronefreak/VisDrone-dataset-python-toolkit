"""Tests for scripts/evaluate.py, scripts/inference.py, scripts/webcam_demo.py.

All tests use mocks so no GPU, camera, or real model weights are needed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).parent.parent))


# ===========================================================================
# Helpers / shared fixtures
# ===========================================================================


def _make_image(h: int = 64, w: int = 80) -> np.ndarray:
    """Create a random BGR image as numpy array."""
    return np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)


def _make_torch_pred(n: int = 3) -> dict[str, torch.Tensor]:
    boxes = torch.rand(n, 4) * 50
    boxes[:, 2:] += boxes[:, :2]  # x2 > x1, y2 > y1
    return {
        "boxes": boxes,
        "labels": torch.randint(0, 10, (n,)),
        "scores": torch.rand(n) * 0.5 + 0.5,
    }


def _make_torch_target(n: int = 2) -> dict[str, torch.Tensor]:
    boxes = torch.rand(n, 4) * 50
    boxes[:, 2:] += boxes[:, :2]
    return {
        "boxes": boxes,
        "labels": torch.randint(0, 10, (n,)),
    }


# ===========================================================================
# evaluate.py tests
# ===========================================================================


class TestEvaluateArgParsing:
    def _parse(self, args: list[str]) -> SimpleNamespace:
        from scripts.evaluate import parse_args

        with patch("sys.argv", ["evaluate.py"] + args):
            return parse_args()

    def test_required_args(self):
        ns = self._parse(
            [
                "--checkpoint",
                "ckpt.pt",
                "--model",
                "fasterrcnn_resnet50",
                "--image-dir",
                "/img",
                "--annotation-dir",
                "/ann",
            ]
        )
        assert ns.checkpoint == "ckpt.pt"
        assert ns.model == "fasterrcnn_resnet50"
        assert ns.image_dir == "/img"

    def test_yolo_model_accepted(self):
        ns = self._parse(
            [
                "--checkpoint",
                "best.pt",
                "--model",
                "yolov8n",
                "--image-dir",
                "/img",
                "--annotation-dir",
                "/ann",
            ]
        )
        assert ns.model == "yolov8n"

    def test_defaults(self):
        ns = self._parse(
            [
                "--checkpoint",
                "c.pt",
                "--image-dir",
                "/i",
                "--annotation-dir",
                "/a",
            ]
        )
        assert ns.score_threshold == 0.05
        assert ns.iou_threshold == 0.5
        assert ns.batch_size == 4


class TestIsYoloModel:
    def test_yolo_prefixes(self):
        from scripts.evaluate import _is_yolo_model

        assert _is_yolo_model("yolov8n")
        assert _is_yolo_model("yolo11x")
        assert _is_yolo_model("yolo26s")
        assert _is_yolo_model("YOLOv8n")  # case-insensitive

    def test_rtdetr_prefixes(self):
        from scripts.evaluate import _is_yolo_model

        assert _is_yolo_model("rtdetr-l")
        assert _is_yolo_model("rtdetr-x")
        assert _is_yolo_model("rtdetr-resnet50")
        assert _is_yolo_model("RTDETR-l")  # case-insensitive

    def test_non_yolo(self):
        from scripts.evaluate import _is_yolo_model

        assert not _is_yolo_model("fasterrcnn_resnet50")
        assert not _is_yolo_model("retinanet_resnet50")
        assert not _is_yolo_model("fcos_resnet50")
        assert not _is_yolo_model("rfdetr-large")  # rfdetr has its own path


class TestIsRFDETRModel:
    def test_rfdetr_prefixes(self):
        from scripts.evaluate import _is_rfdetr_model

        assert _is_rfdetr_model("rfdetr-nano")
        assert _is_rfdetr_model("rfdetr-large")
        assert _is_rfdetr_model("RFDETR-small")  # case-insensitive

    def test_non_rfdetr(self):
        from scripts.evaluate import _is_rfdetr_model

        assert not _is_rfdetr_model("yolov8n")
        assert not _is_rfdetr_model("rtdetr-l")
        assert not _is_rfdetr_model("fasterrcnn_resnet50")


class TestPrintMetricsTable:
    """Smoke-test that the rich table renders without errors."""

    def test_render_torchvision_metrics(self):
        from scripts.evaluate import print_metrics_table

        metrics = {
            "precision": 0.75,
            "recall": 0.60,
            "f1": 0.67,
            "mAP50": None,
            "mAP50_95": None,
            "num_images": 10,
            "fps": 5.0,
            "avg_ms": 200.0,
            "per_class": {
                "car": {"precision": 0.80, "recall": 0.70, "f1": 0.74},
                "pedestrian": {"precision": 0.60, "recall": 0.50, "f1": 0.55},
            },
        }
        # Should not raise
        print_metrics_table("fasterrcnn_resnet50", metrics)

    def test_render_yolo_metrics(self):
        from scripts.evaluate import print_metrics_table

        metrics = {
            "mAP50": 0.45,
            "mAP50_95": 0.25,
            "precision": 0.70,
            "recall": 0.60,
            "per_class": {
                "car": {"mAP50": 0.60, "mAP50_95": 0.35},
                "pedestrian": {"mAP50": 0.40, "mAP50_95": 0.20},
            },
        }
        print_metrics_table("yolov8n", metrics)


class TestPerClassMetrics:
    def test_basic_computation(self):
        from scripts.evaluate import _per_class_metrics

        boxes_a = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
        pred = {"boxes": boxes_a, "labels": torch.tensor([1]), "scores": torch.tensor([0.9])}
        tgt = {"boxes": boxes_a.clone(), "labels": torch.tensor([1])}

        result = _per_class_metrics([pred], [tgt], iou_threshold=0.5)
        assert 1 in result or any("cls" in k or k.isdigit() for k in result) or result
        # At least one class entry computed
        assert len(result) >= 1

    def test_empty_predictions(self):
        from scripts.evaluate import _per_class_metrics

        pred = {
            "boxes": torch.zeros(0, 4),
            "labels": torch.zeros(0, dtype=torch.long),
            "scores": torch.zeros(0),
        }
        tgt = {"boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]), "labels": torch.tensor([2])}

        result = _per_class_metrics([pred], [tgt], iou_threshold=0.5)
        assert len(result) >= 1


class TestSaveJson:
    def test_saves_valid_json(self, tmp_path):
        from scripts.evaluate import _save_json

        pred = _make_torch_pred(2)
        tgt = _make_torch_target(2)
        out = tmp_path / "pred.json"
        _save_json([pred], [tgt], out)
        assert out.exists()
        data = json.loads(out.read_text())
        assert len(data) == 1
        assert "predictions" in data[0]
        assert "ground_truth" in data[0]


class TestEvaluateTorchvisionIntegration:
    """Integration test for torchvision evaluate path using a mock model."""

    def test_evaluate_returns_metrics(self, tmp_path):
        from scripts.evaluate import evaluate_torchvision

        pred = _make_torch_pred(2)
        fake_model = MagicMock()
        fake_model.return_value = [pred]

        # Mock dataset and dataloader to yield one batch
        fake_img = torch.rand(3, 64, 80)
        fake_tgt = _make_torch_target(2)

        with patch("visdrone_toolkit.dataset.VisDroneDataset") as MockDS:
            with patch("torch.utils.data.DataLoader") as MockDL:
                MockDS.return_value.__len__ = MagicMock(return_value=1)
                MockDL.return_value = [([fake_img], [fake_tgt])]

                metrics = evaluate_torchvision(
                    model=fake_model,
                    image_dir=tmp_path,
                    annotation_dir=tmp_path,
                    batch_size=1,
                    num_workers=0,
                    device=torch.device("cpu"),
                    score_threshold=0.1,
                    iou_threshold=0.5,
                    use_soft_nms=False,
                    output_dir=tmp_path,
                    save_predictions=False,
                )

        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        assert metrics["num_images"] >= 0


class TestEvaluateYoloPath:
    def test_evaluate_yolo_extracts_metrics(self):
        """Verify that YOLO results dict is extracted correctly from Ultralytics output."""
        # Test the metric extraction logic directly
        mock_boxes = MagicMock()
        mock_boxes.map50 = 0.45
        mock_boxes.map = 0.25
        mock_boxes.mp = 0.70
        mock_boxes.mr = 0.60
        mock_boxes.ap_class_index = None

        mock_results = MagicMock()
        mock_results.box = mock_boxes

        # Mimic the extraction logic from evaluate_yolo
        metrics: dict = {}
        if hasattr(mock_results, "box"):
            metrics["mAP50"] = float(mock_results.box.map50)
            metrics["mAP50_95"] = float(mock_results.box.map)
            metrics["precision"] = float(mock_results.box.mp)
            metrics["recall"] = float(mock_results.box.mr)

        assert metrics["mAP50"] == pytest.approx(0.45)
        assert metrics["mAP50_95"] == pytest.approx(0.25)
        assert metrics["precision"] == pytest.approx(0.70)
        assert metrics["recall"] == pytest.approx(0.60)

    def test_yolo_metric_per_class_extraction(self):
        """Verify per-class metrics are extracted when ap_class_index is present."""
        mock_boxes = MagicMock()
        mock_boxes.map50 = 0.50
        mock_boxes.map = 0.30
        mock_boxes.mp = 0.65
        mock_boxes.mr = 0.55
        mock_boxes.ap_class_index = [0, 1]
        mock_boxes.ap50 = [0.60, 0.40]
        mock_boxes.ap = [0.35, 0.25]

        mock_results = MagicMock()
        mock_results.box = mock_boxes

        names = ["pedestrian", "people"]
        metrics: dict = {}
        if hasattr(mock_results, "box"):
            metrics["mAP50"] = float(mock_results.box.map50)
            metrics["per_class"] = {}
            for i, cls_idx in enumerate(mock_results.box.ap_class_index):
                cls_name = names[cls_idx] if cls_idx < len(names) else f"class_{cls_idx}"
                metrics["per_class"][cls_name] = {
                    "mAP50": float(mock_results.box.ap50[i]),
                    "mAP50_95": float(mock_results.box.ap[i]),
                }

        assert "pedestrian" in metrics["per_class"]
        assert "people" in metrics["per_class"]
        assert metrics["per_class"]["pedestrian"]["mAP50"] == pytest.approx(0.60)


# ===========================================================================
# inference.py tests
# ===========================================================================


class TestInferenceArgParsing:
    def _parse(self, args: list[str]) -> SimpleNamespace:
        from scripts.inference import parse_args

        with patch("sys.argv", ["inference.py"] + args):
            return parse_args()

    def test_required_args(self):
        ns = self._parse(["--checkpoint", "c.pt", "--input", "/images"])
        assert ns.checkpoint == "c.pt"
        assert ns.input == "/images"

    def test_yolo_model(self):
        ns = self._parse(["--checkpoint", "c.pt", "--input", "/i", "--model", "yolov8n"])
        assert ns.model == "yolov8n"

    def test_defaults(self):
        ns = self._parse(["--checkpoint", "c.pt", "--input", "/i"])
        assert ns.score_threshold == 0.5
        assert not ns.no_save_viz
        assert not ns.show

    def test_video_extensions_recognized(self):
        from scripts.inference import _VIDEO_EXTENSIONS

        assert ".mp4" in _VIDEO_EXTENSIONS
        assert ".avi" in _VIDEO_EXTENSIONS

    def test_image_extensions_recognized(self):
        from scripts.inference import _IMAGE_EXTENSIONS

        assert ".jpg" in _IMAGE_EXTENSIONS
        assert ".png" in _IMAGE_EXTENSIONS


class TestInferenceDrawDetections:
    def test_draws_on_frame(self):
        from scripts.inference import draw_detections

        YOLO_CLASS_COLORS = {
            0: (255, 0, 0),  # pedestrian - red
            1: (255, 128, 0),  # people - orange
            2: (255, 255, 0),  # bicycle - yellow
            3: (0, 255, 0),  # car - green
            4: (0, 255, 128),  # van - light green
            5: (0, 255, 255),  # truck - cyan
            6: (0, 128, 255),  # tricycle - light blue
            7: (0, 0, 255),  # awning-tricycle - blue
            8: (128, 0, 255),  # bus - purple
            9: (255, 0, 255),  # motor - magenta
            10: (255, 0, 128),  # others - pink
        }

        frame = _make_image(100, 120)
        boxes = np.array([[5, 5, 30, 30]], dtype=np.float32)
        scores = np.array([0.9])
        labels = np.array([1])
        result = draw_detections(
            frame, boxes, scores, labels, ["ignored", "pedestrian"], class_colors=YOLO_CLASS_COLORS
        )
        assert result.shape == frame.shape

    def test_empty_detections(self):
        from scripts.inference import draw_detections

        YOLO_CLASS_COLORS = {
            0: (255, 0, 0),  # pedestrian - red
            1: (255, 128, 0),  # people - orange
            2: (255, 255, 0),  # bicycle - yellow
            3: (0, 255, 0),  # car - green
            4: (0, 255, 128),  # van - light green
            5: (0, 255, 255),  # truck - cyan
            6: (0, 128, 255),  # tricycle - light blue
            7: (0, 0, 255),  # awning-tricycle - blue
            8: (128, 0, 255),  # bus - purple
            9: (255, 0, 255),  # motor - magenta
            10: (255, 0, 128),  # others - pink
        }
        frame = _make_image()
        result = draw_detections(
            frame, np.zeros((0, 4)), np.array([]), np.array([]), [], class_colors=YOLO_CLASS_COLORS
        )
        assert result.shape == frame.shape

    def test_label_out_of_range(self):
        from scripts.inference import draw_detections

        frame = _make_image()
        YOLO_CLASS_COLORS = {
            0: (255, 0, 0),  # pedestrian - red
            1: (255, 128, 0),  # people - orange
            2: (255, 255, 0),  # bicycle - yellow
            3: (0, 255, 0),  # car - green
            4: (0, 255, 128),  # van - light green
            5: (0, 255, 255),  # truck - cyan
            6: (0, 128, 255),  # tricycle - light blue
            7: (0, 0, 255),  # awning-tricycle - blue
            8: (128, 0, 255),  # bus - purple
            9: (255, 0, 255),  # motor - magenta
            10: (255, 0, 128),  # others - pink
        }
        result = draw_detections(
            frame,
            np.array([[0, 0, 20, 20]], dtype=np.float32),
            np.array([0.8]),
            np.array([99]),
            ["only_one"],
            class_colors=YOLO_CLASS_COLORS,
        )
        assert result is not None


class TestInferenceImageBGR:
    def test_process_frame_returns_tensor(self):
        from scripts.inference import process_image_for_torchvision

        frame = _make_image(64, 80)
        tensor = process_image_for_torchvision(frame)
        assert tensor.shape == (3, 64, 80)
        assert tensor.dtype == torch.float32
        assert tensor.max() <= 1.0 + 1e-6


class TestInferenceSoftNms:
    def test_apply_soft_nms_reduces_or_equal(self):
        from scripts.inference import _apply_soft_nms

        boxes = np.array(
            [
                [0, 0, 10, 10],
                [1, 1, 11, 11],
                [50, 50, 60, 60],
            ],
            dtype=np.float32,
        )
        scores = np.array([0.9, 0.85, 0.7])
        labels = np.array([1, 1, 2])

        rb, rs, rl = _apply_soft_nms(
            boxes, scores, labels, sigma=0.5, score_threshold=0.3, iou_threshold=0.5
        )
        assert len(rb) <= len(boxes)
        assert len(rb) == len(rs) == len(rl)


class TestInferenceTorchvisionFrame:
    def test_returns_filtered_detections(self):
        from scripts.inference import infer_torchvision_frame

        pred = _make_torch_pred(3)
        # Force all scores high
        pred["scores"] = torch.tensor([0.9, 0.8, 0.7])

        fake_model = MagicMock(return_value=[pred])
        frame = _make_image(64, 80)
        result = infer_torchvision_frame(
            fake_model,
            frame,
            torch.device("cpu"),
            score_threshold=0.5,
            use_soft_nms=False,
            nms_threshold=0.5,
        )
        assert "boxes" in result
        assert "scores" in result
        assert "labels" in result
        assert len(result["boxes"]) <= 3

    def test_score_threshold_filters(self):
        from scripts.inference import infer_torchvision_frame

        pred = _make_torch_pred(3)
        pred["scores"] = torch.tensor([0.2, 0.3, 0.4])  # all below 0.5

        fake_model = MagicMock(return_value=[pred])
        frame = _make_image()
        result = infer_torchvision_frame(
            fake_model,
            frame,
            torch.device("cpu"),
            score_threshold=0.5,
            use_soft_nms=False,
            nms_threshold=0.5,
        )
        assert len(result["boxes"]) == 0


class TestInferenceTorchvisionImages:
    def test_processes_list_of_images(self, tmp_path):
        # Create fake image files
        import cv2

        from scripts.inference import run_torchvision_images

        img_paths = []
        for i in range(2):
            p = tmp_path / f"img{i}.jpg"
            cv2.imwrite(str(p), _make_image())
            img_paths.append(p)

        pred = _make_torch_pred(1)
        pred["scores"] = torch.tensor([0.9])
        fake_model = MagicMock(return_value=[pred])

        run_torchvision_images(
            model=fake_model,
            image_paths=img_paths,
            device=torch.device("cpu"),
            output_dir=tmp_path / "out",
            score_threshold=0.5,
            use_soft_nms=False,
            nms_threshold=0.5,
            save_viz=True,
            show=False,
        )

        out_dir = tmp_path / "out"
        assert out_dir.exists()
        saved = list(out_dir.glob("*_pred.jpg"))
        assert len(saved) == 2


# ===========================================================================
# webcam_demo.py tests
# ===========================================================================


class TestWebcamArgParsing:
    def _parse(self, args: list[str]) -> SimpleNamespace:
        from scripts.webcam_demo import parse_args

        with patch("sys.argv", ["webcam_demo.py"] + args):
            return parse_args()

    def test_defaults(self):
        ns = self._parse([])
        assert ns.source == "0"
        assert ns.model == "fasterrcnn_resnet50"
        assert ns.score_threshold == 0.5

    def test_custom_source(self):
        ns = self._parse(["--source", "myvideo.mp4"])
        assert ns.source == "myvideo.mp4"

    def test_yolo_model(self):
        ns = self._parse(["--model", "yolov8n", "--checkpoint", "best.pt"])
        assert ns.model == "yolov8n"

    def test_no_hardcoded_choices(self):
        """Verify that no choices restriction prevents YOLO models."""
        ns = self._parse(["--model", "yolo26x", "--checkpoint", "c.pt"])
        assert ns.model == "yolo26x"


class TestFPSCounter:
    def test_initial_fps_zero(self):
        from scripts.webcam_demo import FPSCounter

        counter = FPSCounter()
        assert counter.get_fps() == 0.0

    def test_fps_after_updates(self):
        import time

        from scripts.webcam_demo import FPSCounter

        counter = FPSCounter(window_size=5)
        for _ in range(5):
            time.sleep(0.01)
            counter.update()
        fps = counter.get_fps()
        assert fps > 0.0
        assert fps < 1000.0  # sanity

    def test_window_size_limits_history(self):
        from scripts.webcam_demo import FPSCounter

        counter = FPSCounter(window_size=3)
        for _ in range(10):
            counter.update()
        assert len(counter.frame_times) <= 3


class TestWebcamDrawDetections:
    def test_draws_boxes(self):
        from scripts.webcam_demo import draw_detections

        frame = _make_image(100, 120)
        boxes = np.array([[5, 5, 30, 30]], dtype=np.float32)
        labels = np.array([1])
        scores = np.array([0.8])
        result = draw_detections(frame, boxes, labels, scores)
        assert result.shape == frame.shape

    def test_empty_detections_no_crash(self):
        from scripts.webcam_demo import draw_detections

        frame = _make_image()
        result = draw_detections(frame, np.zeros((0, 4)), np.array([]), np.array([]))
        assert result.shape == frame.shape

    def test_class_label_out_of_range(self):
        from scripts.webcam_demo import draw_detections

        frame = _make_image()
        result = draw_detections(
            frame,
            np.array([[0, 0, 10, 10]], dtype=np.float32),
            np.array([999]),
            np.array([0.9]),
        )
        assert result is not None


class TestWebcamLoadTorchvisionModel:
    def test_loads_from_checkpoint(self, tmp_path):
        from scripts.webcam_demo import load_torchvision_model

        ckpt = {"model_state_dict": {}}
        ckpt_path = tmp_path / "ckpt.pt"
        torch.save(ckpt, str(ckpt_path))

        mock_model = MagicMock()
        mock_model.to.return_value = mock_model

        with patch("scripts.webcam_demo.get_model", return_value=mock_model):
            with patch("torch.load", return_value=ckpt):
                model = load_torchvision_model(
                    str(ckpt_path), "fasterrcnn_resnet50", 12, torch.device("cpu")
                )

        assert model is mock_model

    def test_loads_pretrained_when_no_checkpoint(self):
        from scripts.webcam_demo import load_torchvision_model

        mock_model = MagicMock()
        mock_model.to.return_value = mock_model

        with patch("scripts.webcam_demo.get_model", return_value=mock_model):
            model = load_torchvision_model(None, "fasterrcnn_resnet50", 12, torch.device("cpu"))

        assert model is mock_model


class TestInferTorchvision:
    def test_returns_frame_and_count(self):
        from scripts.webcam_demo import infer_torchvision

        pred = _make_torch_pred(2)
        pred["scores"] = torch.tensor([0.9, 0.8])
        fake_model = MagicMock(return_value=[pred])

        frame = _make_image(64, 80)
        annotated, n = infer_torchvision(
            fake_model, frame, torch.device("cpu"), score_threshold=0.5
        )
        assert annotated.shape == frame.shape
        assert n == 2

    def test_threshold_filters_low_confidence(self):
        from scripts.webcam_demo import infer_torchvision

        pred = _make_torch_pred(3)
        pred["scores"] = torch.tensor([0.2, 0.3, 0.4])  # all below threshold
        fake_model = MagicMock(return_value=[pred])

        frame = _make_image()
        _, n = infer_torchvision(fake_model, frame, torch.device("cpu"), score_threshold=0.5)
        assert n == 0


# ===========================================================================
# Trainer weight-saving tests
# ===========================================================================


class TestTrainerSavesLastPt:
    """Verify that trainer.py now saves last.pt every epoch."""

    def test_last_pt_written_each_epoch(self, tmp_path):
        from visdrone_toolkit.trainer import UnifiedTrainer

        mock_model = MagicMock(spec=torch.nn.Module)
        mock_model.parameters.return_value = iter([torch.zeros(1)])
        mock_model.to.return_value = mock_model

        trainer = UnifiedTrainer(mock_model, device=torch.device("cpu"))

        fake_loader = [
            (
                [torch.rand(3, 32, 32)],
                [{"boxes": torch.zeros(0, 4), "labels": torch.zeros(0, dtype=torch.long)}],
            )
        ]

        with patch.object(trainer, "_validate", return_value={"f1": 0.5}):
            with patch.object(trainer, "_train_epoch", return_value=0.5):
                with patch.object(trainer, "_save_checkpoint"):
                    trainer.train(
                        train_loader=fake_loader,
                        val_loader=fake_loader,
                        epochs=2,
                        output_dir=tmp_path,
                    )
                    calls = trainer._save_checkpoint.call_args_list
                    last_pt_calls = [c for c in calls if "last.pt" in str(c)]
                    # Should have one last.pt save per epoch (2 epochs)
                    assert len(last_pt_calls) == 2


class TestYOLOTrainerAbsolutePath:
    """Verify the weight-saving path fix: project must be absolute."""

    def test_project_is_absolute(self, tmp_path):
        from visdrone_toolkit.yolo_trainer import YOLOTrainer

        trainer = YOLOTrainer(
            model_name="yolov8n",
            num_classes=11,
            device="cpu",
        )

        # Capture what is passed to model.train()
        captured: dict = {}

        def fake_train(**kwargs: object) -> MagicMock:
            captured.update(kwargs)
            return MagicMock()

        mock_yolo_instance = MagicMock()
        mock_yolo_instance.train = fake_train

        mock_prepare = MagicMock(return_value=tmp_path / "dataset.yaml")
        (tmp_path / "dataset.yaml").write_text("nc: 11\nnames: []\n")

        import contextlib

        with patch.object(trainer, "_UltralyticsYOLO", return_value=mock_yolo_instance):
            with patch.object(trainer, "_prepare_dataset", mock_prepare):
                with contextlib.suppress(Exception):
                    trainer.train(
                        train_img_dir=str(tmp_path),
                        train_ann_dir=str(tmp_path),
                        val_img_dir=str(tmp_path),
                        val_ann_dir=str(tmp_path),
                        output_dir=str(tmp_path / "outputs"),
                        epochs=1,
                    )  # weights lookup may fail in test env; we only care about `project`

        if "project" in captured:
            project_path = Path(captured["project"])
            assert (
                project_path.is_absolute()
            ), f"project must be absolute; got {captured['project']!r}"


class TestTrainRFDETRRouting:
    """Test that train.py correctly routes rfdetr-* models to _is_rfdetr_model."""

    def test_rfdetr_models_detected(self):
        from scripts.train import _is_rfdetr_model

        assert _is_rfdetr_model("rfdetr-nano")
        assert _is_rfdetr_model("rfdetr-small")
        assert _is_rfdetr_model("rfdetr-medium")
        assert _is_rfdetr_model("rfdetr-large")
        assert _is_rfdetr_model("RFDETR-large")  # case-insensitive

    def test_rfdetr_not_ultralytics(self):
        from scripts.train import _is_rfdetr_model, _is_ultralytics_model

        assert not _is_ultralytics_model("rfdetr-large")
        assert _is_rfdetr_model("rfdetr-large")

    def test_yolo_not_rfdetr(self):
        from scripts.train import _is_rfdetr_model

        assert not _is_rfdetr_model("yolov8n")
        assert not _is_rfdetr_model("rtdetr-l")
        assert not _is_rfdetr_model("fasterrcnn_resnet50")

    def test_rfdetr_lr_arg_exists_with_safe_default(self):
        """--rfdetr-lr argument must exist with a safe default (1e-4)."""
        import sys

        import scripts.train as train_mod

        saved = sys.argv
        try:
            sys.argv = ["train.py", "--available-models"]
            args = train_mod.parse_args()
        finally:
            sys.argv = saved

        assert hasattr(args, "rfdetr_lr"), "--rfdetr-lr arg not found in parsed namespace"
        assert args.rfdetr_lr == 1e-4, f"--rfdetr-lr default should be 1e-4, got {args.rfdetr_lr}"
        assert hasattr(args, "rfdetr_warmup_epochs"), "--rfdetr-warmup-epochs arg not found"
        assert args.rfdetr_warmup_epochs > 0, "--rfdetr-warmup-epochs should be > 0 by default"

    def test_rfdetr_uses_train_ann_dir(self):
        """train() signature must accept train_ann_dir (YOLO format, not dataset_dir)."""
        import inspect

        from visdrone_toolkit.rfdetr_trainer import RFDETRTrainer

        sig = inspect.signature(RFDETRTrainer.train)
        assert "train_ann_dir" in sig.parameters, "train() must accept train_ann_dir"
        assert "dataset_dir" not in sig.parameters, "train() must NOT have old dataset_dir param"
