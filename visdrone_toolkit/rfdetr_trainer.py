"""RF-DETR training via the rfdetr package (YOLO format).

RF-DETR (Roboflow DETR) is a transformer-based detector built on top of DINOv2.
Training is delegated to the rfdetr package's native engine.

This trainer uses the **YOLO format** (same raw data as the YOLO/RT-DETR pipeline):
- Raw VisDrone annotations are converted on-the-fly with ``convert_to_yolo``
- No separate COCO JSON dataset is needed
- ``data/VisDrone2019-DET-train/`` and ``data/VisDrone2019-DET-val/`` work directly

Dataset structure written to a temp directory for rfdetr's YOLO loader:
    <tmp>/
        data.yaml
        train/
            images/   (symlinked from source)
            labels/   (YOLO .txt from convert_to_yolo)
        valid/
            images/
            labels/

Requires: pip install rfdetr
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import yaml

from visdrone_toolkit.converters.visdrone_to_yolo import convert_to_yolo

# VisDrone classes after filtering ``ignored-regions`` (category 0).
# Consistent with the YOLO pipeline (_VISDRONE_CLASSES in yolo_trainer.py).
# Class IDs here are 0-based: pedestrian=0, ..., others=10.
_RFDETR_CLASSES = [
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
]  # 11 classes (ignored-regions filtered by convert_to_yolo)

# Map from registered model name → rfdetr class name
_MODEL_CLASS_MAP = {
    "rfdetr-nano": "RFDETRNano",
    "rfdetr-small": "RFDETRSmall",
    "rfdetr-medium": "RFDETRMedium",
    "rfdetr-large": "RFDETRLarge",
}

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


class RFDETRTrainer:
    """Trains RF-DETR models using the rfdetr native engine (YOLO format).

    Handles:
    - Converting VisDrone native annotations to YOLO format (on the fly, in a temp dir)
    - Writing the data.yaml required by rfdetr's YOLO dataset loader
    - Delegating training to the rfdetr package's native engine
    - Returning a standardised {model_path, output_dir} result

    Does NOT re-implement RF-DETR's internal loss or training loop.
    Uses the same raw VisDrone data as the YOLO pipeline — no COCO JSON needed.
    """

    # RF-DETR-safe default LR. The global train.py default (0.005) is calibrated
    # for YOLO/torchvision SGD and is ~50× too large for RF-DETR's AdamW encoder.
    _DEFAULT_LR: float = 1e-4
    _DEFAULT_LR_ENCODER_RATIO: float = 1.5  # lr_encoder = lr * ratio

    def __init__(
        self,
        model_name: str,
        num_classes: int = 11,
        device: str = "cuda",
    ) -> None:
        """Initialize RFDETRTrainer.

        Args:
            model_name: Registered model name, e.g. 'rfdetr-large'
            num_classes: Number of detection classes (default 11 for VisDrone w/o ignored-regions)
            device: Device string ('cuda', 'cpu', 'cuda:0')
        """
        if model_name not in _MODEL_CLASS_MAP:
            raise ValueError(
                f"Unknown RF-DETR model '{model_name}'. "
                f"Available: {list(_MODEL_CLASS_MAP.keys())}"
            )

        try:
            import rfdetr as _rfdetr_pkg  # noqa: F401
        except ImportError as err:
            raise ImportError(
                "rfdetr is required for RF-DETR training. Install with: pip install rfdetr"
            ) from err

        self._model_name = model_name
        self._rfdetr_class_name = _MODEL_CLASS_MAP[model_name]
        self.num_classes = num_classes
        self.device = device

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(
        self,
        train_img_dir: str | Path,
        train_ann_dir: str | Path,
        val_img_dir: str | Path | None = None,
        val_ann_dir: str | Path | None = None,
        epochs: int = 100,
        batch_size: int = 4,
        lr: float = _DEFAULT_LR,
        output_dir: str | Path = "outputs",
        workers: int = 2,
        grad_accum_steps: int = 4,
        use_ema: bool = True,
        warmup_epochs: float = 5.0,
        amp_dtype: str = "bf16",
        resume: str | Path | None = None,
        **extra_kwargs: Any,
    ) -> dict[str, Any]:
        """Train an RF-DETR model on VisDrone data (YOLO format).

        Converts VisDrone annotations to YOLO format in a temporary directory,
        writes data.yaml, then calls rfdetr's native training engine.

        Args:
            train_img_dir: Path to training images.
            train_ann_dir: Path to VisDrone training annotations.
            val_img_dir: Path to validation images (optional).
            val_ann_dir: Path to VisDrone validation annotations (optional).
            epochs: Number of training epochs.
            batch_size: Per-GPU batch size. Use ``"auto"`` to let rfdetr choose.
            lr: Initial learning rate. Defaults to 1e-4 (RF-DETR's safe value).
                Do NOT use the global train.py default (0.005) — causes NaN losses.
            output_dir: Where to save checkpoints and logs.
            workers: Number of DataLoader workers.
            grad_accum_steps: Gradient accumulation steps.
            use_ema: Whether to use EMA (recommended for RF-DETR).
            warmup_epochs: LR warmup epochs (default 5).
            amp_dtype: Mixed-precision dtype. ``"bf16"`` is more stable on Ampere+ GPUs.
            resume: Path to a checkpoint (``.pth``) to resume from. Passed as
                ``ckpt_path`` to PyTorch Lightning's ``trainer.fit()``, so epoch
                count, optimizer state, and LR scheduler are all restored.
            **extra_kwargs: Forwarded to rfdetr's TrainConfig.

        Returns:
            dict with keys ``results``, ``model_path``, ``output_dir``.
        """
        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory(prefix="visdrone_rfdetr_") as tmp:
            tmp_path = Path(tmp)
            self._prepare_dataset(tmp_path, train_img_dir, train_ann_dir, val_img_dir, val_ann_dir)

            model = self._build_model()

            train_kwargs: dict[str, Any] = dict(
                dataset_dir=str(tmp_path),
                dataset_file="yolo",
                epochs=epochs,
                batch_size=batch_size,
                lr=lr,
                lr_encoder=lr * self._DEFAULT_LR_ENCODER_RATIO,
                output_dir=str(output_dir),
                num_workers=workers,
                grad_accum_steps=grad_accum_steps,
                use_ema=use_ema,
                warmup_epochs=warmup_epochs,
                amp_dtype=amp_dtype,
                device=self.device,
                **extra_kwargs,
            )
            if resume is not None:
                train_kwargs["resume"] = str(resume)
            results = model.train(**train_kwargs)

        # RF-DETR saves checkpoints to output_dir/
        best_ckpt = output_dir / "best_checkpoint.pth"
        last_ckpt = output_dir / "checkpoint.pth"
        final_path = (
            best_ckpt if best_ckpt.exists() else (last_ckpt if last_ckpt.exists() else None)
        )

        return {
            "results": results,
            "model_path": str(final_path) if final_path else None,
            "output_dir": str(output_dir),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_model(self) -> Any:
        """Instantiate the rfdetr model class with num_classes."""
        import rfdetr

        cls = getattr(rfdetr, self._rfdetr_class_name)
        return cls(num_classes=self.num_classes)

    def _prepare_dataset(
        self,
        tmp_path: Path,
        train_img_dir: str | Path,
        train_ann_dir: str | Path,
        val_img_dir: str | Path | None,
        val_ann_dir: str | Path | None,
    ) -> None:
        """Convert VisDrone data to YOLO format and write data.yaml.

        Creates the directory structure expected by rfdetr's YOLO loader:
            tmp_path/
                data.yaml
                train/images/   (symlinks to source images)
                train/labels/   (YOLO .txt annotations)
                valid/images/   (optional)
                valid/labels/   (optional)

        Args:
            tmp_path: Temporary directory root.
            train_img_dir: VisDrone training images directory.
            train_ann_dir: VisDrone training annotations directory.
            val_img_dir: VisDrone validation images directory (optional).
            val_ann_dir: VisDrone validation annotations directory (optional).
        """
        names = _RFDETR_CLASSES[: self.num_classes]

        # Training split
        train_images = tmp_path / "train" / "images"
        train_labels = tmp_path / "train" / "labels"
        train_images.mkdir(parents=True)
        train_labels.mkdir(parents=True)
        self._symlink_images(Path(train_img_dir), train_images)
        convert_to_yolo(
            image_dir=train_img_dir,
            annotation_dir=train_ann_dir,
            output_dir=train_labels,
            filter_ignored=True,
            filter_crowd=True,
            create_yaml=False,
        )

        data: dict[str, Any] = {
            "names": names,
            "nc": len(names),
            "train": "train/images",
            "val": "valid/images",  # rfdetr YOLO loader requires this key
        }

        # Validation split (optional)
        if val_img_dir and val_ann_dir:
            val_images = tmp_path / "valid" / "images"
            val_labels = tmp_path / "valid" / "labels"
            val_images.mkdir(parents=True)
            val_labels.mkdir(parents=True)
            self._symlink_images(Path(val_img_dir), val_images)
            convert_to_yolo(
                image_dir=val_img_dir,
                annotation_dir=val_ann_dir,
                output_dir=val_labels,
                filter_ignored=True,
                filter_crowd=True,
                create_yaml=False,
            )

        yaml_path = tmp_path / "data.yaml"
        with open(yaml_path, "w") as fh:
            yaml.dump(data, fh, default_flow_style=False)

    @staticmethod
    def _symlink_images(src_dir: Path, dst_dir: Path) -> None:
        """Create per-file image symlinks from src_dir into dst_dir."""
        for img_path in src_dir.iterdir():
            if img_path.suffix.lower() in _IMAGE_SUFFIXES:
                link = dst_dir / img_path.name
                if not link.exists():
                    link.symlink_to(img_path.resolve())
