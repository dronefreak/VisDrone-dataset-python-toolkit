"""YOLO training via Ultralytics engine.

Delegates training to Ultralytics' native trainer, which implements the full
YOLO training pipeline (TaskAlignedAssigner, DFL loss, box/cls/dfl losses, etc.).

This avoids "abstraction optimism" — YOLO training is fundamentally different
from torchvision and cannot be unified at the backward pass level.

What IS unified across all models (handled by train.py orchestration):
- CLI interface
- Dataset loading and filtering
- Checkpoint directory management
- Logging format
- Evaluation metrics

What is NOT unified (each framework uses its own engine):
- Loss computation
- Gradient flow
- Augmentation pipeline (Ultralytics uses Mosaic/MixUp internally)
- Label assignment strategy
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import yaml

from visdrone_toolkit.converters.visdrone_to_yolo import convert_to_yolo

_VISDRONE_CLASSES = [
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
]  # 11 classes after filtering ignored-regions (class 0)


class YOLOTrainer:
    """Trains YOLO models using the Ultralytics training engine.

    Handles:
    - Converting VisDrone annotations to YOLO format (on the fly, in a temp dir)
    - Generating the dataset YAML required by Ultralytics
    - Delegating training to ultralytics.YOLO.train()
    - Saving the final model to the requested output directory

    Does NOT attempt to re-implement YOLO's internal loss or assignment logic.
    """

    def __init__(
        self,
        model_name: str,
        num_classes: int = 11,
        device: str = "cuda",
    ) -> None:
        """Initialize YOLOTrainer.

        Args:
            model_name: Registered model name, e.g. 'yolov8n', 'yolov9c', 'yolov10m'
            num_classes: Number of detection classes (default 11 for VisDrone w/o ignored)
            device: Device string passed to Ultralytics ('cuda', 'cpu', '0', '0,1', ...)
        """
        try:
            from ultralytics import YOLO as UltralyticsYOLO
        except ImportError as err:
            raise ImportError(
                "Ultralytics is required for YOLO training. "
                "Install with: pip install ultralytics>=8.0.0"
            ) from err

        # Derive the .pt filename from the registered model name
        # e.g. 'yolov8n' -> 'yolov8n.pt', 'yolov10m' -> 'yolov10m.pt'
        self._pt_name = f"{model_name}.pt"
        self._model_name = model_name
        self.num_classes = num_classes
        self.device = device
        self._UltralyticsYOLO = UltralyticsYOLO

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(
        self,
        train_img_dir: str | Path,
        train_ann_dir: str | Path,
        val_img_dir: str | Path | None,
        val_ann_dir: str | Path | None,
        epochs: int = 100,
        batch_size: int = 16,
        lr: float = 0.001,
        imgsz: int = 640,
        use_amp: bool = True,
        output_dir: str | Path = "outputs",
        workers: int = 4,
        **extra_kwargs: Any,
    ) -> dict[str, Any]:
        """Train a YOLO model on VisDrone data.

        Converts VisDrone annotations to YOLO format in a temporary directory,
        writes a dataset YAML, then calls ultralytics.YOLO.train().

        Args:
            train_img_dir: Path to training images
            train_ann_dir: Path to VisDrone training annotations
            val_img_dir: Path to validation images (optional)
            val_ann_dir: Path to VisDrone validation annotations (optional)
            epochs: Number of training epochs
            batch_size: Batch size
            lr: Initial learning rate (lr0 in Ultralytics terminology)
            imgsz: Input image size
            use_amp: Use automatic mixed precision
            output_dir: Where to save the final model and logs
            workers: Number of DataLoader workers
            **extra_kwargs: Passed directly to ultralytics.YOLO.train()

        Returns:
            dict with keys: 'results', 'model_path', 'output_dir'
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory(prefix="visdrone_yolo_") as tmp:
            tmp_path = Path(tmp)
            dataset_yaml = self._prepare_dataset(
                tmp_path, train_img_dir, train_ann_dir, val_img_dir, val_ann_dir
            )

            model = self._UltralyticsYOLO(self._pt_name)

            results = model.train(
                data=str(dataset_yaml),
                epochs=epochs,
                batch=batch_size,
                imgsz=imgsz,
                lr0=lr,
                amp=use_amp,
                device=self.device,
                workers=workers,
                project=str(output_dir),
                name=self._model_name,
                exist_ok=True,
                nc=self.num_classes,
                **extra_kwargs,
            )

        # Ultralytics saves best/last weights under project/name/weights/
        weights_dir = output_dir / self._model_name / "weights"
        best_model = weights_dir / "best.pt"
        last_model = weights_dir / "last.pt"
        final_path = best_model if best_model.exists() else last_model

        return {
            "results": results,
            "model_path": str(final_path) if final_path.exists() else None,
            "output_dir": str(output_dir / self._model_name),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _prepare_dataset(
        self,
        tmp_path: Path,
        train_img_dir: str | Path,
        train_ann_dir: str | Path,
        val_img_dir: str | Path | None,
        val_ann_dir: str | Path | None,
    ) -> Path:
        """Convert VisDrone data to YOLO format and write a dataset YAML.

        Args:
            tmp_path: Temp directory to write converted labels into
            train_img_dir: VisDrone training images
            train_ann_dir: VisDrone training annotations
            val_img_dir: VisDrone validation images (optional)
            val_ann_dir: VisDrone validation annotations (optional)

        Returns:
            Path to the generated dataset.yaml file
        """
        train_labels = tmp_path / "labels" / "train"
        val_labels = tmp_path / "labels" / "val"

        # Convert training annotations
        convert_to_yolo(
            image_dir=train_img_dir,
            annotation_dir=train_ann_dir,
            output_dir=train_labels,
            filter_ignored=True,
            filter_crowd=True,
            create_yaml=False,  # We write our own YAML below
        )

        # Convert validation annotations (if provided)
        if val_img_dir and val_ann_dir:
            convert_to_yolo(
                image_dir=val_img_dir,
                annotation_dir=val_ann_dir,
                output_dir=val_labels,
                filter_ignored=True,
                filter_crowd=True,
                create_yaml=False,
            )

        # Write dataset YAML — Ultralytics requires absolute image paths
        dataset: dict[str, Any] = {
            "path": str(tmp_path),
            "train": {"images": str(Path(train_img_dir).resolve()), "labels": str(train_labels)},
            "nc": self.num_classes,
            "names": _VISDRONE_CLASSES[: self.num_classes],
        }
        if val_img_dir and val_ann_dir:
            dataset["val"] = {
                "images": str(Path(val_img_dir).resolve()),
                "labels": str(val_labels),
            }

        yaml_path = tmp_path / "dataset.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(dataset, f, default_flow_style=False)

        return yaml_path
