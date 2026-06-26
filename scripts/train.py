"""Training script for VisDrone object detection models.

Supports all models registered in ModelRegistry including:
- Torchvision: FasterRCNN, FCOS, RetinaNet
- YOLO: v8, v9, v10
- Future: DETR and other transformers

Uses UnifiedTrainer for framework-agnostic training with automatic format conversion.
Includes automatic mixed precision, learning rate scheduling, and checkpointing.
"""

import argparse
from pathlib import Path

import torch
from rich.console import Console

from visdrone_toolkit.augmentations import get_training_augmentation
from visdrone_toolkit.dataset import VisDroneDataset
from visdrone_toolkit.trainer import UnifiedTrainer
from visdrone_toolkit.utils import collate_fn, get_model

console = Console()


def parse_args():
    parser = argparse.ArgumentParser(description="Train object detection models on VisDrone")

    parser.add_argument("--available-models", action="store_true", help="Show available models")

    # Dataset paths
    parser.add_argument(
        "--train-img-dir",
        help="Training images directory (for YOLO/torchvision). For RF-DETR models, pass the Roboflow COCO root dir here (e.g. data/VisDrone2019-DET-RF-DETR/).",
    )
    parser.add_argument(
        "--train-ann-dir", help="Training annotations directory (YOLO/torchvision only)"
    )
    parser.add_argument("--val-img-dir", help="Validation images directory (YOLO/torchvision only)")
    parser.add_argument(
        "--val-ann-dir", help="Validation annotations directory (YOLO/torchvision only)"
    )

    # Model configuration
    parser.add_argument(
        "--model",
        default="fasterrcnn_resnet50",
        help="Model name (see available_models for options)",
    )
    parser.add_argument("--num-classes", type=int, default=12, help="Number of classes")
    parser.add_argument(
        "--pretrained", action="store_true", default=True, help="Use pretrained weights"
    )
    parser.add_argument("--no-pretrained", dest="pretrained", action="store_false")

    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.005, help="Learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="SGD momentum")
    parser.add_argument("--weight-decay", type=float, default=0.0005, help="Weight decay")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")

    # Training options
    parser.add_argument("--amp", action="store_true", help="Use automatic mixed precision")
    parser.add_argument(
        "--accumulation-steps",
        type=int,
        default=1,
        help="Gradient accumulation steps",
    )

    # Data augmentation
    parser.add_argument("--augmentation", action="store_true", help="Use data augmentation")
    parser.add_argument(
        "--multiscale", action="store_true", help="Multi-scale training (600-800px)"
    )

    # Learning rate schedule
    parser.add_argument(
        "--lr-schedule",
        default="step",
        choices=["step", "multistep", "cosine"],
        help="LR schedule type",
    )
    parser.add_argument(
        "--lr-milestones",
        nargs="+",
        type=int,
        default=[30, 40],
        help="LR decay milestones for multistep",
    )

    # Checkpointing
    parser.add_argument("--output-dir", default="outputs", help="Output directory")
    parser.add_argument("--resume", help="Resume from checkpoint")
    parser.add_argument("--save-every", type=int, default=10, help="Save checkpoint every N epochs")

    # Device
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device"
    )

    args = parser.parse_args()

    # Check for available-models before requiring dataset paths
    if args.available_models:
        return args

    # For RF-DETR, only --train-img-dir (as dataset_dir) is required
    if _is_rfdetr_model(args.model):
        if not args.train_img_dir:
            parser.error(
                "For RF-DETR models, pass the Roboflow COCO root directory as --train-img-dir "
                "(e.g. --train-img-dir data/VisDrone2019-DET-RF-DETR/)"
            )
    elif not args.train_img_dir or not args.train_ann_dir:
        parser.error("--train-img-dir and --train-ann-dir are required for training")

    return args


def show_available_models():
    """Display all available models from registry and torchvision."""
    from visdrone_toolkit.abstract_models import ModelRegistry

    console.print("\n[bold cyan]Available Models:[/bold cyan]")
    console.print("\n[yellow]Torchvision (default backend):[/yellow]")
    tv_models = [
        "fasterrcnn_resnet50",
        "fasterrcnn_mobilenet",
        "fcos_resnet50",
        "retinanet_resnet50",
    ]
    for model in tv_models:
        console.print(f"  • {model}")

    console.print("\n[yellow]YOLO Models (ultralytics):[/yellow]")
    yolo_models = [m for m in ModelRegistry._registry if "yolo" in m.lower()]
    for model in sorted(yolo_models):
        console.print(f"  • {model}")

    console.print("\n[yellow]RT-DETR Models (ultralytics):[/yellow]")
    rtdetr_models = [m for m in ModelRegistry._registry if m.lower().startswith("rtdetr")]
    for model in sorted(rtdetr_models):
        console.print(f"  • {model}")

    console.print("\n[yellow]RF-DETR Models (rfdetr / PyTorch Lightning):[/yellow]")
    rfdetr_models = [m for m in ModelRegistry._registry if m.lower().startswith("rfdetr")]
    for model in sorted(rfdetr_models):
        console.print(f"  • {model}")

    console.print("\n[dim]Use --model <name> to select a model[/dim]\n")


def _is_ultralytics_model(model_name: str) -> bool:
    """Return True if the model is handled by the Ultralytics engine (YOLO or RT-DETR)."""
    name = model_name.lower()
    return name.startswith("yolo") or name.startswith("rtdetr")


def _is_rfdetr_model(model_name: str) -> bool:
    """Return True if the model is an RF-DETR model (rfdetr package)."""
    return model_name.lower().startswith("rfdetr")


def _train_ultralytics(args) -> None:
    """Route YOLO/RT-DETR training to the Ultralytics engine via YOLOTrainer."""
    from visdrone_toolkit.yolo_trainer import _VISDRONE_CLASSES, YOLOTrainer

    is_rtdetr = args.model.lower().startswith("rtdetr")
    model_family = "RT-DETR" if is_rtdetr else "YOLO"
    console.print(
        f"\n[bold yellow]{model_family} model detected — using Ultralytics training engine[/bold yellow]"
    )
    console.print(
        "[dim]Note: --multiscale, --small-anchors, --lr-schedule, --accumulation-steps "
        "are handled internally by Ultralytics for YOLO/RT-DETR models.[/dim]\n"
    )

    # YOLO always trains with 11 classes: VisDrone's ignored-regions (class 0) is
    # removed by the converter. If the user passed --num-classes 12 (the raw count),
    # clamp to the actual filtered count so nc matches len(names) in the YAML.
    num_classes = min(args.num_classes, len(_VISDRONE_CLASSES))

    device_str = args.device  # e.g. 'cuda', 'cpu', '0'

    trainer = YOLOTrainer(
        model_name=args.model,
        num_classes=num_classes,
        device=device_str,
    )

    result = trainer.train(
        train_img_dir=args.train_img_dir,
        train_ann_dir=args.train_ann_dir,
        val_img_dir=args.val_img_dir,
        val_ann_dir=args.val_ann_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        use_amp=args.amp,
        output_dir=args.output_dir,
        workers=args.num_workers,
    )

    console.print("\n[bold green]Training complete![/bold green]")
    if result["model_path"]:
        console.print(f"  Best model saved to: {result['model_path']}")
    console.print(f"  All artifacts saved to: {result['output_dir']}")


def _train_rfdetr(args) -> None:
    """Route RF-DETR model training to RFDETRTrainer (rfdetr / PyTorch Lightning)."""
    from visdrone_toolkit.rfdetr_trainer import _RFDETR_CLASSES, RFDETRTrainer

    console.print(
        "\n[bold yellow]RF-DETR model detected — using rfdetr PyTorch Lightning engine[/bold yellow]"
    )
    console.print(
        "[dim]Note: --train-img-dir should be the Roboflow COCO root dir "
        "(e.g. data/VisDrone2019-DET-RF-DETR/).[/dim]"
    )
    console.print(
        "[dim]Note: --multiscale, --lr-schedule, --small-anchors are ignored for RF-DETR.[/dim]\n"
    )

    num_classes = len(_RFDETR_CLASSES)  # always 10 (filtered)
    dataset_dir = args.train_img_dir  # reuse --train-img-dir as Roboflow COCO root

    trainer = RFDETRTrainer(
        model_name=args.model,
        num_classes=num_classes,
        device=args.device,
    )

    result = trainer.train(
        dataset_dir=dataset_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        output_dir=args.output_dir,
        workers=args.num_workers,
        grad_accum_steps=args.accumulation_steps,
    )

    console.print("\n[bold green]Training complete![/bold green]")
    if result["model_path"]:
        console.print(f"  Best model saved to: {result['model_path']}")
    console.print(f"  All artifacts saved to: {result['output_dir']}")


def _train_torchvision(args) -> None:
    """Route torchvision model training to UnifiedTrainer."""
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)

    # Create datasets
    console.print("\n[yellow]Loading datasets...[/yellow]")
    train_transforms = get_training_augmentation() if args.augmentation else None
    train_dataset = VisDroneDataset(
        image_dir=args.train_img_dir,
        annotation_dir=args.train_ann_dir,
        transforms=train_transforms,
        filter_ignored=True,
        filter_crowd=True,
        multiscale_training=args.multiscale,
    )
    console.print(f"[green]✓[/green] Loaded {len(train_dataset)} training images")

    val_dataset = None
    if args.val_img_dir and args.val_ann_dir:
        val_dataset = VisDroneDataset(
            image_dir=args.val_img_dir,
            annotation_dir=args.val_ann_dir,
            transforms=None,
            filter_ignored=True,
            filter_crowd=True,
            multiscale_training=False,
        )
        console.print(f"[green]✓[/green] Loaded {len(val_dataset)} validation images")

    from torch.utils.data import DataLoader

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=device.type == "cuda",
    )
    val_loader = None
    if val_dataset:
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
            pin_memory=device.type == "cuda",
        )

    # Create model
    console.print(f"\n[yellow]Creating model: {args.model}[/yellow]")
    model = get_model(
        model_name=args.model,
        num_classes=args.num_classes,
        pretrained=args.pretrained,
    )
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    console.print(f"[cyan]Total parameters: {total_params:,}[/cyan]")
    console.print(f"[cyan]Trainable parameters: {trainable_params:,}[/cyan]")

    trainer = UnifiedTrainer(model, device=device)

    optimizer = None
    if args.resume:
        console.print(f"\n[yellow]Resuming from checkpoint: {args.resume}[/yellow]")
        optimizer = torch.optim.SGD(
            [p for p in model.parameters() if p.requires_grad],
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
        trainer.load_checkpoint(args.resume, optimizer)
        console.print("[green]✓[/green] Checkpoint loaded")

    # Build LR scheduler
    lr_scheduler = None
    base_opt = optimizer or torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    if args.lr_schedule == "multistep":
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            base_opt, milestones=args.lr_milestones, gamma=0.1
        )
    elif args.lr_schedule == "cosine":
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(base_opt, T_max=args.epochs)

    console.print("\n[bold green]Starting training...[/bold green]\n")
    result = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        use_amp=args.amp,
        accumulation_steps=args.accumulation_steps,
        output_dir=output_dir,
        save_every=args.save_every,
        val_every=1,
    )

    console.print("\n[bold green]Training complete![/bold green]")
    console.print("[cyan]Final metrics:[/cyan]")
    console.print(f"  Best F1: {result['best_metric']:.4f}")
    console.print(f"  Checkpoints saved to: {output_dir}")


def main():
    args = parse_args()

    if args.available_models:
        show_available_models()
        return

    console.print("\n[bold cyan]Training Configuration[/bold cyan]")
    console.print(f"Model: {args.model}")
    console.print(f"Device: {args.device}")
    console.print(f"Epochs: {args.epochs}, Batch size: {args.batch_size}")
    console.print(f"Learning rate: {args.lr}")
    if args.amp:
        console.print("[green]✓[/green] Using automatic mixed precision")

    if _is_ultralytics_model(args.model):
        _train_ultralytics(args)
    elif _is_rfdetr_model(args.model):
        _train_rfdetr(args)
    else:
        _train_torchvision(args)


if __name__ == "__main__":
    main()
