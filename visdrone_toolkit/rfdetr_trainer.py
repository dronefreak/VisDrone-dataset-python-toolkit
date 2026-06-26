"""RF-DETR training via the rfdetr package.

RF-DETR (Roboflow DETR) is a transformer-based detector built on top of DINOv2.
Training is delegated to the rfdetr package's PyTorch Lightning stack.

What this trainer handles:
- Filtering the COCO JSON to remove the ``others`` category (consistency with YOLO pipeline)
- Creating a temp dataset directory with symlinked images + filtered annotations
- Calling rfdetr's native training engine
- Returning a standardised result dict with model_path and output_dir

Dataset format required by rfdetr (Roboflow COCO):
    <dataset_dir>/
        train/
            _annotations.coco.json
            image1.jpg
            ...
        valid/
            _annotations.coco.json
            ...
        test/          (optional)
            _annotations.coco.json
            ...

The VisDrone data is already pre-prepared at ``data/VisDrone2019-DET-RF-DETR/``.

Requires: pip install "rfdetr[train]" pytorch_lightning
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

# VisDrone classes after filtering ``ignored-regions`` (class 0) AND ``others``
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
]  # 10 classes (no ignored-regions, no others)

# Category name to exclude from COCO JSON (id=11 in the pre-prepared dataset)
_EXCLUDED_CATEGORY = "others"

# Map from registered model name → rfdetr class name
_MODEL_CLASS_MAP = {
    "rfdetr-nano": "RFDETRNano",
    "rfdetr-small": "RFDETRSmall",
    "rfdetr-medium": "RFDETRMedium",
    "rfdetr-large": "RFDETRLarge",
}


class RFDETRTrainer:
    """Trains RF-DETR models using the rfdetr PyTorch Lightning engine.

    Handles:
    - Filtering ``others`` from the Roboflow COCO JSON annotations
    - Symlinking images into a temp directory for filtered training
    - Delegating training to the rfdetr package's native engine
    - Returning standardised {model_path, output_dir} result

    Does NOT re-implement RF-DETR's internal loss or training loop.
    """

    def __init__(
        self,
        model_name: str,
        num_classes: int = 10,
        device: str = "cuda",
    ) -> None:
        """Initialize RFDETRTrainer.

        Args:
            model_name: Registered model name, e.g. 'rfdetr-large'
            num_classes: Number of detection classes (default 10 for VisDrone w/o ignored + others)
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
                "rfdetr is required for RF-DETR training. " "Install with: pip install rfdetr"
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
        dataset_dir: str | Path,
        epochs: int = 100,
        batch_size: int = 4,
        lr: float = 1e-4,
        output_dir: str | Path = "outputs",
        workers: int = 2,
        grad_accum_steps: int = 4,
        use_ema: bool = True,
        **extra_kwargs: Any,
    ) -> dict[str, Any]:
        """Train an RF-DETR model on VisDrone (Roboflow COCO) data.

        Filters ``others`` annotations from the COCO JSON, writes a temp
        dataset directory with symlinked images, then calls rfdetr's training
        engine.

        Args:
            dataset_dir: Path to Roboflow COCO format dataset
                         (e.g. ``data/VisDrone2019-DET-RF-DETR/``).
            epochs: Number of training epochs.
            batch_size: Per-GPU batch size. Use ``"auto"`` to let rfdetr choose.
            lr: Initial learning rate.
            output_dir: Where to save checkpoints and logs.
            workers: Number of DataLoader workers.
            grad_accum_steps: Gradient accumulation steps.
            use_ema: Whether to use EMA (recommended for RF-DETR).
            **extra_kwargs: Forwarded to rfdetr's TrainConfig.

        Returns:
            dict with keys ``results``, ``model_path``, ``output_dir``.
        """
        dataset_dir = Path(dataset_dir).resolve()
        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory(prefix="visdrone_rfdetr_") as tmp:
            tmp_path = Path(tmp)
            filtered_dir = self._prepare_dataset(tmp_path, dataset_dir)

            model = self._build_model()

            results = model.train(
                dataset_dir=str(filtered_dir),
                dataset_file="roboflow",
                epochs=epochs,
                batch_size=batch_size,
                lr=lr,
                lr_encoder=lr * 1.5,
                output_dir=str(output_dir),
                num_workers=workers,
                grad_accum_steps=grad_accum_steps,
                use_ema=use_ema,
                device=self.device,
                **extra_kwargs,
            )

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

    def _prepare_dataset(self, tmp_path: Path, dataset_dir: Path) -> Path:
        """Create a filtered dataset in tmp_path.

        For each split (train, valid, test):
        - Creates a real directory
        - Symlinks each image file from the source split directory
        - Writes a filtered ``_annotations.coco.json`` (``others`` removed)

        Args:
            tmp_path: Temporary directory root.
            dataset_dir: Source Roboflow COCO dataset directory.

        Returns:
            Path to the filtered dataset root (same as tmp_path).
        """
        for split in ("train", "valid", "test"):
            src_split = dataset_dir / split
            if not src_split.exists():
                continue  # test split is optional

            dst_split = tmp_path / split
            dst_split.mkdir()

            # Symlink each image file
            self._symlink_images(src_split, dst_split)

            # Write filtered COCO JSON
            src_json = src_split / "_annotations.coco.json"
            if src_json.exists():
                dst_json = dst_split / "_annotations.coco.json"
                self._filter_coco_json(src_json, dst_json)

        return tmp_path

    def _symlink_images(self, src_dir: Path, dst_dir: Path) -> None:
        """Create per-file image symlinks from src_dir into dst_dir.

        Using per-file symlinks (not a directory symlink) so downstream code
        that resolves symlinks can still find images.
        """
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
        for img_path in src_dir.iterdir():
            if img_path.suffix.lower() in image_extensions:
                link = dst_dir / img_path.name
                if not link.exists():
                    link.symlink_to(img_path.resolve())

    def _filter_coco_json(self, src_json: Path, dst_json: Path) -> None:
        """Write a filtered COCO JSON removing the ``others`` category.

        Args:
            src_json: Source ``_annotations.coco.json``.
            dst_json: Destination path for filtered JSON.
        """
        with open(src_json) as fh:
            data = json.load(fh)

        # Find excluded category IDs
        excluded_ids = {
            cat["id"] for cat in data.get("categories", []) if cat["name"] == _EXCLUDED_CATEGORY
        }

        if not excluded_ids:
            # Nothing to filter — write as-is
            shutil.copy2(src_json, dst_json)
            return

        filtered_categories = [cat for cat in data["categories"] if cat["id"] not in excluded_ids]
        filtered_annotations = [
            ann for ann in data.get("annotations", []) if ann["category_id"] not in excluded_ids
        ]

        filtered_data = {
            **data,
            "categories": filtered_categories,
            "annotations": filtered_annotations,
        }

        with open(dst_json, "w") as fh:
            json.dump(filtered_data, fh)
