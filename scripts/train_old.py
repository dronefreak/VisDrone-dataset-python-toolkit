"""
Training script for VisDrone object detection models.

Supports Faster R-CNN, FCOS, and RetinaNet with various backbones.
Includes automatic mixed precision, learning rate scheduling, and checkpointing.
"""

import argparse
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torchvision.models.detection.anchor_utils import AnchorGenerator

from visdrone_toolkit.augmentations import get_training_augmentation
from visdrone_toolkit.dataset import VisDroneDataset
from visdrone_toolkit.utils import collate_fn, get_model, load_checkpoint, save_checkpoint
from visdrone_toolkit.visualization import plot_training_curves

console = Console()


def parse_args():
    parser = argparse.ArgumentParser(description="Train object detection models on VisDrone")

    # Dataset paths
    parser.add_argument("--train-img-dir", required=True, help="Training images directory")
    parser.add_argument("--train-ann-dir", required=True, help="Training annotations directory")
    parser.add_argument("--val-img-dir", help="Validation images directory")
    parser.add_argument("--val-ann-dir", help="Validation annotations directory")

    # Model configuration
    parser.add_argument(
        "--model",
        default="fasterrcnn_resnet50",
        choices=[
            "fasterrcnn_resnet50",
            "fasterrcnn_mobilenet",
            "fcos_resnet50",
            "retinanet_resnet50",
        ],
        help="Model architecture",
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
        help="Gradient accumulation steps (simulate larger batch)",
    )
    parser.add_argument(
        "--reduce-anchors", action="store_true", help="Reduce anchor sizes to avoid OOM issues"
    )
    parser.add_argument(
        "--filter-ignored", action="store_true", default=True, help="Filter ignored boxes"
    )
    parser.add_argument(
        "--filter-crowd", action="store_true", default=True, help="Filter crowd regions"
    )

    # Data augmentation
    parser.add_argument("--augmentation", action="store_true", help="Use data augmentation")
    parser.add_argument(
        "--multiscale", action="store_true", help="Multi-scale training (600-800px)"
    )

    # Advanced training options
    parser.add_argument(
        "--small-anchors", action="store_true", help="Use smaller anchors for small objects"
    )
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
        default=[60, 80],
        help="LR decay milestones for multistep",
    )

    # Checkpointing
    parser.add_argument("--output-dir", default="outputs", help="Output directory")
    parser.add_argument("--resume", help="Resume from checkpoint")
    parser.add_argument(
        "--save-every", type=int, default=100, help="Save checkpoint every N epochs"
    )

    # Device
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda/cpu)"
    )

    return parser.parse_args()


@torch.no_grad()
def compute_metrics(predictions, targets, iou_threshold=0.5):
    """
    Compute precision, recall, and mAP for object detection.

    Args:
        predictions: List of dicts with 'boxes', 'labels', 'scores'
        targets: List of dicts with 'boxes', 'labels'
        iou_threshold: IoU threshold for matching predictions to targets

    Returns:
        dict with precision, recall, and mAP
    """
    total_tp = 0
    total_fp = 0
    total_gt = 0

    for pred, target in zip(predictions, targets):
        pred_boxes = pred["boxes"]
        pred_labels = pred["labels"]

        gt_boxes = target["boxes"]
        gt_labels = target["labels"]

        total_gt += len(gt_boxes)

        if len(pred_boxes) == 0:
            continue

        if len(gt_boxes) == 0:
            total_fp += len(pred_boxes)
            continue

        # Compute IoU matrix
        ious = box_iou(pred_boxes, gt_boxes)

        # Match predictions to ground truth
        matched_gt = set()
        for i in range(len(pred_boxes)):
            best_iou = 0
            best_gt_idx = -1

            for j in range(len(gt_boxes)):
                if j in matched_gt:
                    continue
                if pred_labels[i] != gt_labels[j]:
                    continue
                if ious[i, j] > best_iou:
                    best_iou = ious[i, j]
                    best_gt_idx = j

            if best_iou >= iou_threshold and best_gt_idx != -1:
                total_tp += 1
                matched_gt.add(best_gt_idx)
            else:
                total_fp += 1

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / total_gt if total_gt > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def box_iou(boxes1, boxes2):
    """Compute IoU between two sets of boxes."""
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]

    union = area1[:, None] + area2 - inter
    iou = inter / union

    return iou


def train_one_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    data_loader: DataLoader,
    device: torch.device,
    epoch: int,
    scaler: Optional[GradScaler] = None,
    use_amp: bool = False,
    accumulation_steps: int = 1,
) -> tuple[float, dict]:
    """Train for one epoch with rich progress tracking and gradient accumulation."""
    model.train()

    total_loss = 0
    num_batches = len(data_loader)

    console.print(f"\n[bold cyan]Epoch {epoch} - Training[/bold cyan]")
    if accumulation_steps > 1:
        console.print(
            f"[yellow]Using gradient accumulation: {accumulation_steps} steps (effective batch: {data_loader.batch_size * accumulation_steps})[/yellow]"
        )

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("[cyan]Training...", total=num_batches)

        start_time = time.time()

        for batch_idx, (images, targets) in enumerate(data_loader):
            # Move to device
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Forward pass with optional AMP
            if use_amp and scaler is not None:
                with autocast(device_type=device.type):
                    loss_dict = model(images, targets)
                    losses = sum(loss for loss in loss_dict.values()) / accumulation_steps

                # Backward pass
                scaler.scale(losses).backward()

                # Only step optimizer every accumulation_steps
                if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == num_batches:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values()) / accumulation_steps

                # Backward pass
                losses.backward()

                # Only step optimizer every accumulation_steps
                if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == num_batches:
                    optimizer.step()
                    optimizer.zero_grad()

            total_loss += losses.item() * accumulation_steps

            # Update progress
            progress.update(
                task,
                advance=1,
                description=f"[cyan]Training (Loss: {losses.item() * accumulation_steps:.4f})",
            )

    epoch_time = time.time() - start_time
    avg_loss = total_loss / num_batches

    console.print(
        f"[green]✓[/green] Epoch {epoch} completed in {epoch_time:.2f}s - Avg Loss: {avg_loss:.4f}"
    )

    return avg_loss, {"epoch_time": epoch_time}


@torch.no_grad()
def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    epoch: int,
    score_threshold: float = 0.5,
) -> tuple[float, dict]:
    """Evaluate model on validation set with metrics."""
    model.eval()  # Set to eval mode for inference

    total_loss = 0
    all_predictions = []
    all_targets = []
    num_batches = len(data_loader)

    console.print(f"\n[bold magenta]Epoch {epoch} - Validation[/bold magenta]")

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("[magenta]Validating...", total=num_batches)

        for _, (images, targets) in enumerate(data_loader):
            # Move to device
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Get predictions
            predictions = model(images)

            # Filter by score threshold
            filtered_preds = []
            for pred in predictions:
                keep = pred["scores"] > score_threshold
                filtered_preds.append(
                    {
                        "boxes": pred["boxes"][keep],
                        "labels": pred["labels"][keep],
                        "scores": pred["scores"][keep],
                    }
                )

            all_predictions.extend(filtered_preds)
            all_targets.extend(targets)

            # Compute loss (switch to train mode temporarily)
            model.train()
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            model.eval()

            total_loss += losses.item()

            progress.update(task, advance=1)

    avg_loss = total_loss / num_batches

    # Compute metrics
    metrics = compute_metrics(all_predictions, all_targets, iou_threshold=0.5)

    # Create metrics table
    table = Table(title=f"Validation Metrics (Epoch {epoch})", show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")

    table.add_row("Loss", f"{avg_loss:.4f}")
    table.add_row("Precision", f"{metrics['precision']:.4f}")
    table.add_row("Recall", f"{metrics['recall']:.4f}")
    table.add_row("F1 Score", f"{metrics['f1']:.4f}")

    console.print(table)

    return avg_loss, metrics


def main():
    args = parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set device
    device = torch.device(args.device)

    # Print header
    console.rule("[bold blue]VisDrone Training[/bold blue]")
    console.print(f"[cyan]Device:[/cyan] {device}")

    if device.type == "cuda":
        console.print(f"[cyan]GPU:[/cyan] {torch.cuda.get_device_name(0)}")
        console.print(
            f"[cyan]Memory:[/cyan] {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
        )

    # Create datasets
    console.print("\n[yellow]Loading datasets...[/yellow]")
    train_transforms = get_training_augmentation() if args.augmentation else None
    train_dataset = VisDroneDataset(
        image_dir=args.train_img_dir,
        annotation_dir=args.train_ann_dir,
        transforms=train_transforms,
        filter_ignored=args.filter_ignored,
        filter_crowd=args.filter_crowd,
        multiscale_training=args.multiscale,
    )

    if args.augmentation:
        console.print("[green]✓[/green] Using data augmentation")
    if args.multiscale:
        console.print("[green]✓[/green] Using multi-scale training (600-800px)")

    val_dataset = None
    if args.val_img_dir and args.val_ann_dir:
        val_dataset = VisDroneDataset(
            image_dir=args.val_img_dir,
            annotation_dir=args.val_ann_dir,
            transforms=None,  # No augmentation for validation
            filter_ignored=args.filter_ignored,
            filter_crowd=args.filter_crowd,
            multiscale_training=False,  # Fixed scale for validation
        )

    # Create dataloaders
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

    # Apply small anchors for small objects
    if args.small_anchors or args.reduce_anchors:
        console.print("[green]✓[/green] Using small anchors optimized for aerial detection")
        if hasattr(model, "rpn") and hasattr(model.rpn, "anchor_generator"):
            # Smaller anchors: 16, 32, 64, 128, 256 (vs default 32, 64, 128, 256, 512)
            small_anchor_sizes = ((16,), (32,), (64,), (128,), (256,))
            aspect_ratios = ((0.5, 1.0, 2.0),) * len(small_anchor_sizes)
            model.rpn.anchor_generator = AnchorGenerator(
                sizes=small_anchor_sizes, aspect_ratios=aspect_ratios
            )

            # Also update RPN parameters for better recall
            model.rpn.pre_nms_top_n_train = 2000
            model.rpn.post_nms_top_n_train = 2000
            model.rpn.pre_nms_top_n_test = 1000
            model.rpn.post_nms_top_n_test = 1000

            # Lower NMS threshold for dense scenes
            model.roi_heads.nms_thresh = 0.3
            model.roi_heads.score_thresh = 0.05
            model.roi_heads.detections_per_img = 300
        else:
            console.print("[red]✗[/red] Model does not support anchor modification")
    model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    console.print(f"[cyan]Total parameters:[/cyan] {total_params:,}")
    console.print(f"[cyan]Trainable parameters:[/cyan] {trainable_params:,}")

    # Create optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    # Learning rate scheduler
    if args.lr_schedule == "multistep":
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=args.lr_milestones, gamma=0.1
        )
        console.print(f"[green]✓[/green] Using MultiStepLR with milestones {args.lr_milestones}")
    elif args.lr_schedule == "cosine":
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        console.print("[green]✓[/green] Using CosineAnnealingLR")
    else:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)
        console.print("[green]✓[/green] Using StepLR (step_size=15)")

    # AMP scaler
    scaler = GradScaler() if args.amp and device.type == "cuda" else None
    if args.amp:
        console.print("[green]✓[/green] Using Automatic Mixed Precision (AMP)")

    # Resume from checkpoint
    start_epoch = 1
    if args.resume:
        console.print(f"\n[yellow]Resuming from checkpoint: {args.resume}[/yellow]")
        start_epoch = (
            load_checkpoint(
                args.resume,
                model,
                optimizer,
                lr_scheduler,
                device=str(device),
            )
            + 1
        )

    # Training loop
    console.rule(f"[bold green]Starting training for {args.epochs} epochs[/bold green]")

    train_losses = []
    val_losses = []
    val_metrics_history = []
    best_val_loss = float("inf")
    best_f1 = 0.0

    try:
        for epoch in range(start_epoch, args.epochs + 1):
            # Train
            train_loss, train_info = train_one_epoch(
                model,
                optimizer,
                train_loader,
                device,
                epoch,
                scaler,
                args.amp,
                args.accumulation_steps,
            )
            train_losses.append(train_loss)

            # Validate
            if val_loader:
                val_loss, val_metrics = evaluate(model, val_loader, device, epoch)
                val_losses.append(val_loss)
                val_metrics_history.append(val_metrics)

                # Save best model based on F1 score
                if val_metrics["f1"] > best_f1:
                    best_f1 = val_metrics["f1"]
                    best_path = output_dir / "best_model.pth"
                    save_checkpoint(
                        model,
                        optimizer,
                        epoch,
                        best_path,
                        lr_scheduler,
                        train_loss=train_loss,
                        val_loss=val_loss,
                    )
                    console.print(f"[green]✓ New best model saved! F1: {best_f1:.4f}[/green]")

                # Also track best validation loss
                if val_loss < best_val_loss:
                    best_val_loss = val_loss

            # Update learning rate
            lr_scheduler.step()

            # Save checkpoint
            if epoch % args.save_every == 0:
                checkpoint_path = output_dir / f"checkpoint_epoch_{epoch}.pth"
                save_checkpoint(
                    model,
                    optimizer,
                    epoch,
                    checkpoint_path,
                    lr_scheduler,
                    train_loss=train_loss,
                    val_loss=val_losses[-1] if val_losses else None,
                )

    except KeyboardInterrupt:
        console.print("\n[yellow]Training interrupted by user (Ctrl+C)[/yellow]")

        # Save interrupt checkpoint
        interrupt_path = output_dir / "interrupt_checkpoint.pth"
        current_epoch = start_epoch + len(train_losses) - 1
        save_checkpoint(
            model,
            optimizer,
            current_epoch,
            interrupt_path,
            lr_scheduler,
            train_loss=train_losses[-1] if train_losses else None,
            val_loss=val_losses[-1] if val_losses else None,
        )
        console.print(f"[green]✓ Checkpoint saved to {interrupt_path}[/green]")
        console.print(f"[cyan]Resume training with: --resume {interrupt_path}[/cyan]")

        # Still plot what we have
        if train_losses:
            curves_path = output_dir / "training_curves_interrupted.png"
            plot_training_curves(
                train_losses, val_losses if val_losses else None, save_path=curves_path, show=False
            )
            console.print(f"[green]✓ Partial training curves saved to {curves_path}[/green]")

        return  # Exit gracefully

    # Save final model
    final_path = output_dir / "final_model.pth"
    save_checkpoint(
        model,
        optimizer,
        args.epochs,
        final_path,
        lr_scheduler,
        train_loss=train_losses[-1],
        val_loss=val_losses[-1] if val_losses else None,
    )
    console.print(f"\n[green]✓ Final model saved to {final_path}[/green]")

    # Plot training curves
    curves_path = output_dir / "training_curves.png"
    plot_training_curves(
        train_losses, val_losses if val_losses else None, save_path=curves_path, show=False
    )
    console.print(f"[green]✓ Training curves saved to {curves_path}[/green]")

    # Final summary
    console.rule("[bold blue]Training Complete[/bold blue]")

    summary_table = Table(show_header=True)
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Value", style="green")

    summary_table.add_row("Output Directory", str(output_dir))
    summary_table.add_row("Best Validation Loss", f"{best_val_loss:.4f}")
    if val_metrics_history:
        summary_table.add_row("Best F1 Score", f"{best_f1:.4f}")
        summary_table.add_row("Final Precision", f"{val_metrics_history[-1]['precision']:.4f}")
        summary_table.add_row("Final Recall", f"{val_metrics_history[-1]['recall']:.4f}")

    console.print(summary_table)


if __name__ == "__main__":
    main()
