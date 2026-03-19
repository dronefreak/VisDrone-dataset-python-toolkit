"""
RF-DETR training script for VisDrone dataset.

RF-DETR uses a DETR-style training paradigm with Hungarian matching losses
(set-based bipartite matching) that is fundamentally different from the
anchor-based or dense-prediction models in train.py. This script provides
the correct native training pipeline.

Workflow:
    1. Convert VisDrone annotations to COCO format in a temporary directory.
    2. Symlink images into the rfdetr-expected directory structure.
    3. Call model.train() from the rfdetr package.

Requirements:
    pip install rfdetr

Usage:
    python scripts/train_rfdetr.py \\
        --train-img-dir data/VisDrone2019-DET-train/images \\
        --train-ann-dir data/VisDrone2019-DET-train/annotations \\
        --val-img-dir   data/VisDrone2019-DET-val/images \\
        --val-ann-dir   data/VisDrone2019-DET-val/annotations \\
        --model rfdetr_base \\
        --epochs 50 \\
        --batch-size 4 \\
        --output-dir outputs/rfdetr_base

    # Resume from a checkpoint
    python scripts/train_rfdetr.py ... --resume outputs/rfdetr_base/checkpoint.pth

    # Fine-tune with a different learning rate
    python scripts/train_rfdetr.py ... --lr 1e-5 --epochs 20

RF-DETR Recommended Hyperparameters (VisDrone):
    - Model:       rfdetr_base (good balance), rfdetr_large (best accuracy)
    - Epochs:      50–100
    - Batch size:  4 (12 GB GPU), 2 (8 GB GPU)
    - Grad accum:  4 (effective batch 16 is ideal for DETR-style training)
    - LR:          1e-4 (AdamW, cosine decay)
    - LR encoder:  1.5e-4 (higher LR for the backbone encoder)
    - Image size:  560 (rfdetr_base default), 560 (rfdetr_large)
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

from rich.console import Console
from rich.table import Table

console = Console()

# VisDrone class names (12 classes, 0-indexed internally in rfdetr)
VISDRONE_CATEGORIES = [
    {"id": 1, "name": "pedestrian", "supercategory": "person"},
    {"id": 2, "name": "people", "supercategory": "person"},
    {"id": 3, "name": "bicycle", "supercategory": "vehicle"},
    {"id": 4, "name": "car", "supercategory": "vehicle"},
    {"id": 5, "name": "van", "supercategory": "vehicle"},
    {"id": 6, "name": "truck", "supercategory": "vehicle"},
    {"id": 7, "name": "tricycle", "supercategory": "vehicle"},
    {"id": 8, "name": "awning-tricycle", "supercategory": "vehicle"},
    {"id": 9, "name": "bus", "supercategory": "vehicle"},
    {"id": 10, "name": "motor", "supercategory": "vehicle"},
    {"id": 11, "name": "others", "supercategory": "none"},
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train RF-DETR on VisDrone dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Dataset paths
    parser.add_argument("--train-img-dir", required=True, help="Training images directory")
    parser.add_argument("--train-ann-dir", required=True, help="Training annotations directory")
    parser.add_argument("--val-img-dir", help="Validation images directory")
    parser.add_argument("--val-ann-dir", help="Validation annotations directory")

    # Model
    parser.add_argument(
        "--model",
        default="rfdetr_base",
        choices=["rfdetr_base", "rfdetr_large"],
        help="RF-DETR model variant",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=11,
        help="Number of VisDrone object classes (default: 11, excluding ignored-regions)",
    )
    parser.add_argument(
        "--pretrained",
        action="store_true",
        default=True,
        help="Load RF-DETR COCO pretrained weights",
    )
    parser.add_argument("--no-pretrained", dest="pretrained", action="store_false")

    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size per GPU (effective batch = batch_size × grad_accum_steps)",
    )
    parser.add_argument(
        "--grad-accum-steps",
        type=int,
        default=4,
        help="Gradient accumulation steps (DETR training benefits from large effective batch)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate for transformer decoder & classification head",
    )
    parser.add_argument(
        "--lr-encoder",
        type=float,
        default=1.5e-4,
        help="Learning rate for encoder/backbone (typically slightly higher)",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="AdamW weight decay",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=560,
        help="Input image size (square, both rfdetr_base and rfdetr_large use 560 by default)",
    )

    # Regularisation
    parser.add_argument(
        "--use-ema",
        action="store_true",
        default=True,
        help="Use Exponential Moving Average for model weights",
    )
    parser.add_argument("--no-ema", dest="use_ema", action="store_false")

    # Dataset filtering
    parser.add_argument(
        "--filter-ignored",
        action="store_true",
        default=True,
        help="Filter ignored regions (score=0 in VisDrone annotations)",
    )
    parser.add_argument("--no-filter-ignored", dest="filter_ignored", action="store_false")
    parser.add_argument(
        "--filter-crowd",
        action="store_true",
        default=True,
        help="Filter crowd/ignored regions (category=0)",
    )
    parser.add_argument("--no-filter-crowd", dest="filter_crowd", action="store_false")

    # Checkpointing
    parser.add_argument(
        "--output-dir",
        default="outputs/rfdetr",
        help="Directory for checkpoints and logs",
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="Path to checkpoint to resume training from",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=10,
        help="Save checkpoint every N epochs",
    )

    # Data preparation
    parser.add_argument(
        "--dataset-dir",
        default=None,
        help=(
            "Directory for COCO-formatted dataset used by rfdetr. "
            "If not given a temporary directory is used."
        ),
    )
    parser.add_argument(
        "--keep-dataset",
        action="store_true",
        help="Keep the converted COCO dataset directory after training",
    )

    # Device
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="Training device",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Dataset preparation helpers
# ---------------------------------------------------------------------------


def _parse_visdrone_annotation(ann_path: Path) -> list[dict]:
    """Parse a single VisDrone annotation file."""
    boxes = []
    with open(ann_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 8:
                continue
            boxes.append(
                {
                    "x": int(parts[0]),
                    "y": int(parts[1]),
                    "w": int(parts[2]),
                    "h": int(parts[3]),
                    "score": int(parts[4]),
                    "category": int(parts[5]),
                }
            )
    return boxes


def convert_split_to_coco(
    image_dir: Path,
    ann_dir: Path,
    dest_dir: Path,
    split_name: str,
    filter_ignored: bool = True,
    filter_crowd: bool = True,
) -> None:
    """
    Convert one VisDrone split (train or val) to COCO format and link images.

    Produces:
        dest_dir/<split_name>/images/    (symlinks or copies of original images)
        dest_dir/<split_name>/_annotations.coco.json
    """
    from PIL import Image as PILImage

    split_dir = dest_dir / split_name
    images_out = split_dir / "images"
    images_out.mkdir(parents=True, exist_ok=True)

    coco: dict = {
        "info": {"description": f"VisDrone {split_name} — COCO format for RF-DETR"},
        "licenses": [],
        "categories": VISDRONE_CATEGORIES,
        "images": [],
        "annotations": [],
    }

    image_files = sorted(
        f for f in image_dir.iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    ann_id = 1

    console.print(f"[yellow]Converting {len(image_files)} {split_name} images...[/yellow]")

    for img_id, img_path in enumerate(image_files, start=1):
        # Link or copy image
        dest_img = images_out / img_path.name
        if not dest_img.exists():
            try:
                os.symlink(img_path.resolve(), dest_img)
            except OSError:
                shutil.copy2(img_path, dest_img)

        # Get image dimensions
        try:
            with PILImage.open(img_path) as pil_img:
                width, height = pil_img.size
        except Exception as exc:
            console.print(f"[red]Cannot open {img_path.name}: {exc}[/red]")
            continue

        coco["images"].append(
            {
                "id": img_id,
                "file_name": img_path.name,
                "width": width,
                "height": height,
            }
        )

        ann_path = ann_dir / (img_path.stem + ".txt")
        if not ann_path.exists():
            continue

        for box in _parse_visdrone_annotation(ann_path):
            if filter_ignored and box["score"] == 0:
                continue
            if filter_crowd and box["category"] == 0:
                continue
            if box["w"] <= 0 or box["h"] <= 0:
                continue
            # Category 0 (ignored-regions) filtered above; keep 1-11
            if box["category"] < 1 or box["category"] > 11:
                continue

            coco["annotations"].append(
                {
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": box["category"],
                    "bbox": [box["x"], box["y"], box["w"], box["h"]],
                    "area": box["w"] * box["h"],
                    "iscrowd": 0,
                    "segmentation": [],
                }
            )
            ann_id += 1

    ann_json = split_dir / "_annotations.coco.json"
    with open(ann_json, "w") as f:
        json.dump(coco, f)

    console.print(
        f"[green]✓[/green] {split_name}: "
        f"{len(coco['images'])} images, {len(coco['annotations'])} annotations → {ann_json}"
    )


def prepare_dataset(
    train_img_dir: str,
    train_ann_dir: str,
    val_img_dir: str | None,
    val_ann_dir: str | None,
    dest_dir: Path,
    filter_ignored: bool,
    filter_crowd: bool,
) -> None:
    """Convert VisDrone train (and optionally val) splits to COCO format."""
    convert_split_to_coco(
        Path(train_img_dir),
        Path(train_ann_dir),
        dest_dir,
        "train",
        filter_ignored=filter_ignored,
        filter_crowd=filter_crowd,
    )

    if val_img_dir and val_ann_dir:
        convert_split_to_coco(
            Path(val_img_dir),
            Path(val_ann_dir),
            dest_dir,
            "valid",
            filter_ignored=filter_ignored,
            filter_crowd=filter_crowd,
        )
    else:
        # rfdetr requires a valid split even if empty; create a minimal one
        console.print(
            "[yellow]No validation set provided. "
            "Creating an empty 'valid' split (training-only mode).[/yellow]"
        )
        valid_dir = dest_dir / "valid"
        (valid_dir / "images").mkdir(parents=True, exist_ok=True)
        minimal_coco = {
            "info": {},
            "licenses": [],
            "categories": VISDRONE_CATEGORIES,
            "images": [],
            "annotations": [],
        }
        with open(valid_dir / "_annotations.coco.json", "w") as f:
            json.dump(minimal_coco, f)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    # Check rfdetr is installed
    try:
        from rfdetr import RFDETRBase, RFDETRLarge  # noqa: F401
    except ImportError:
        console.print(
            "[bold red]Error:[/bold red] The 'rfdetr' package is not installed.\n"
            "Install it with:\n"
            "    [cyan]pip install rfdetr[/cyan]\n"
            "or:\n"
            "    [cyan]pip install -r requirements.txt[/cyan]"
        )
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Print training summary
    # -----------------------------------------------------------------------
    console.rule("[bold blue]RF-DETR Training — VisDrone[/bold blue]")

    summary = Table(show_header=False, box=None)
    summary.add_column("Key", style="cyan")
    summary.add_column("Value", style="white")
    summary.add_row("Model", args.model)
    summary.add_row("Epochs", str(args.epochs))
    summary.add_row("Batch size", str(args.batch_size))
    summary.add_row("Grad accum steps", str(args.grad_accum_steps))
    summary.add_row(
        "Effective batch size", str(args.batch_size * args.grad_accum_steps)
    )
    summary.add_row("LR (decoder)", str(args.lr))
    summary.add_row("LR (encoder)", str(args.lr_encoder))
    summary.add_row("Weight decay", str(args.weight_decay))
    summary.add_row("Image size", str(args.image_size))
    summary.add_row("EMA", "yes" if args.use_ema else "no")
    summary.add_row("Pretrained", "yes" if args.pretrained else "no")
    summary.add_row("Output dir", str(output_dir))
    console.print(summary)
    console.print()

    # -----------------------------------------------------------------------
    # Prepare COCO-format dataset
    # -----------------------------------------------------------------------
    managed_tmpdir: tempfile.TemporaryDirectory | None = None
    if args.dataset_dir:
        dataset_dir = Path(args.dataset_dir)
        dataset_dir.mkdir(parents=True, exist_ok=True)
    else:
        managed_tmpdir = tempfile.TemporaryDirectory(prefix="visdrone_rfdetr_")
        dataset_dir = Path(managed_tmpdir.name)

    console.print("[yellow]Preparing COCO-format dataset...[/yellow]")
    try:
        prepare_dataset(
            train_img_dir=args.train_img_dir,
            train_ann_dir=args.train_ann_dir,
            val_img_dir=args.val_img_dir,
            val_ann_dir=args.val_ann_dir,
            dest_dir=dataset_dir,
            filter_ignored=args.filter_ignored,
            filter_crowd=args.filter_crowd,
        )
    except Exception as exc:
        console.print(f"[red]Dataset preparation failed: {exc}[/red]")
        if managed_tmpdir and not args.keep_dataset:
            managed_tmpdir.cleanup()
        sys.exit(1)

    console.print(f"[green]✓[/green] Dataset ready at: {dataset_dir}\n")

    # -----------------------------------------------------------------------
    # Build RF-DETR model
    # -----------------------------------------------------------------------
    from rfdetr import RFDETRBase, RFDETRLarge

    console.print(f"[yellow]Creating RF-DETR model: {args.model}[/yellow]")

    pretrain_weights: str | None
    if args.pretrained:
        pretrain_weights = (
            "rf-detr-base.pth" if args.model == "rfdetr_base" else "rf-detr-large.pth"
        )
    else:
        pretrain_weights = None

    rfdetr_cls = RFDETRBase if args.model == "rfdetr_base" else RFDETRLarge

    model = rfdetr_cls(
        num_classes=args.num_classes,
        pretrain_weights=pretrain_weights,
        device=args.device,
    )
    console.print(f"[green]✓[/green] Model created (num_classes={args.num_classes})\n")

    # -----------------------------------------------------------------------
    # Build train kwargs
    # -----------------------------------------------------------------------
    train_kwargs: dict = {
        "dataset_dir": str(dataset_dir),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_accum_steps": args.grad_accum_steps,
        "lr": args.lr,
        "lr_encoder": args.lr_encoder,
        "weight_decay": args.weight_decay,
        "checkpoint_dir": str(output_dir),
        "use_ema": args.use_ema,
    }

    # Some rfdetr versions support image_size and checkpoint_interval
    try:
        import inspect
        sig = inspect.signature(model.train)
        if "image_size" in sig.parameters:
            train_kwargs["image_size"] = args.image_size
        if "checkpoint_interval" in sig.parameters:
            train_kwargs["checkpoint_interval"] = args.checkpoint_interval
        if "resume_checkpoint" in sig.parameters and args.resume:
            train_kwargs["resume_checkpoint"] = args.resume
    except Exception:
        pass  # Silently skip unsupported kwargs

    console.rule("[bold green]Starting RF-DETR Training[/bold green]")
    console.print(
        "[dim]Training uses DETR-style set-based loss (Hungarian matching).\n"
        "Checkpoints will be saved to:[/dim] "
        f"[cyan]{output_dir}[/cyan]\n"
    )

    # -----------------------------------------------------------------------
    # Train
    # -----------------------------------------------------------------------
    try:
        model.train(**train_kwargs)
    except KeyboardInterrupt:
        console.print("\n[yellow]Training interrupted by user (Ctrl+C).[/yellow]")
        console.print(
            f"[cyan]Partial checkpoints may be available in {output_dir}[/cyan]"
        )
    except Exception as exc:
        console.print(f"[bold red]Training error:[/bold red] {exc}")
        raise
    finally:
        # Clean up temporary dataset dir unless user asked to keep it
        if managed_tmpdir is not None and not args.keep_dataset:
            managed_tmpdir.cleanup()
            console.print("[dim]Temporary dataset directory cleaned up.[/dim]")

    console.rule("[bold blue]Training Complete[/bold blue]")
    console.print(f"[green]✓[/green] Checkpoints saved to: [cyan]{output_dir}[/cyan]")
    console.print(
        "\n[dim]To run inference with a trained checkpoint:[/dim]\n"
        f"  python scripts/inference.py \\\n"
        f"      --model {args.model} \\\n"
        f"      --checkpoint {output_dir}/checkpoint.pth \\\n"
        f"      --input <image_or_dir>"
    )


if __name__ == "__main__":
    main()
