r"""
Evaluation script for VisDrone object detection models.

Computes standard object detection metrics on validation/test sets.
Supports torchvision models (P/R/F1 + mAP via pycocotools) and
YOLO models (mAP@0.5, mAP@0.5:0.95 via Ultralytics val engine).

Usage examples:
  # Torchvision model
  python scripts/evaluate.py \\
      --checkpoint outputs/fasterrcnn/best.pt \\
      --model fasterrcnn_resnet50 \\
      --image-dir data/VisDrone2019-DET-val/images \\
      --annotation-dir data/VisDrone2019-DET-val/annotations

  # YOLO model
  python scripts/evaluate.py \\
      --checkpoint outputs/yolov8n_200ep/yolov8n/weights/best.pt \\
      --model yolov8n \\
      --image-dir data/VisDrone2019-DET-val/images \\
      --annotation-dir data/VisDrone2019-DET-val/annotations
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from rich.console import Console
from rich.table import Table

from visdrone_toolkit.utils import VISDRONE_CLASSES, collate_fn, compute_metrics, get_model

console = Console()

_ULTRALYTICS_PREFIXES = ("yolo", "rtdetr")


def _is_yolo_model(name: str) -> bool:
    return name.lower().startswith(_ULTRALYTICS_PREFIXES)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate VisDrone detection models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint / .pt file")
    parser.add_argument("--model", default="fasterrcnn_resnet50", help="Model name")
    parser.add_argument("--num-classes", type=int, default=12, help="Number of classes")

    # Dataset
    parser.add_argument("--image-dir", required=True, help="Images directory")
    parser.add_argument("--annotation-dir", required=True, help="Annotations directory")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")

    # Evaluation options
    parser.add_argument("--score-threshold", type=float, default=0.05, help="Score threshold")
    parser.add_argument("--iou-threshold", type=float, default=0.5, help="IoU threshold")
    parser.add_argument("--soft-nms", action="store_true", help="Use Soft-NMS (torchvision only)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    # Output
    parser.add_argument("--output-dir", default="eval_outputs", help="Output directory")
    parser.add_argument("--save-predictions", action="store_true", help="Save predictions JSON")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# YOLO evaluation path
# ---------------------------------------------------------------------------


def evaluate_yolo(
    checkpoint_path: str,
    image_dir: str | Path,
    annotation_dir: str | Path,
    num_classes: int,
    device: str,
    output_dir: Path,
    model_name: str = "",
) -> dict[str, Any]:
    """Evaluate a YOLO or RT-DETR model using the Ultralytics val engine.

    Converts VisDrone annotations to YOLO format on-the-fly, runs
    ``model.val()``, and returns the standard Ultralytics metrics dict.
    """
    try:
        from ultralytics import YOLO as UltralyticsYOLO
    except ImportError as err:
        raise ImportError("pip install ultralytics>=8.0.0") from err

    import tempfile

    from visdrone_toolkit.yolo_trainer import _VISDRONE_CLASSES, YOLOTrainer

    is_rtdetr = model_name.lower().startswith("rtdetr")
    if is_rtdetr:
        try:
            from ultralytics import RTDETR as _LoadClass
        except ImportError as err:
            raise ImportError("pip install -U ultralytics") from err
        family_label = "RT-DETR"
    else:
        _LoadClass = UltralyticsYOLO
        family_label = "YOLO"

    console.print(
        f"\n[bold cyan]{family_label} evaluation — using Ultralytics val engine[/bold cyan]"
    )

    names = _VISDRONE_CLASSES[: min(num_classes, len(_VISDRONE_CLASSES))]
    trainer = YOLOTrainer.__new__(YOLOTrainer)
    trainer.num_classes = len(names)
    trainer._UltralyticsYOLO = UltralyticsYOLO

    with tempfile.TemporaryDirectory(prefix="visdrone_yolo_eval_") as tmp:
        tmp_path = Path(tmp)
        dataset_yaml = trainer._prepare_dataset(
            tmp_path,
            image_dir,
            annotation_dir,
            image_dir,  # use same dir for val
            annotation_dir,
        )

        model = _LoadClass(str(checkpoint_path))
        results = model.val(
            data=str(dataset_yaml),
            device=device,
            split="val",
            save_json=False,
            project=str(output_dir.resolve()),
            name="yolo_eval",
            exist_ok=True,
        )

    # Extract metrics from Ultralytics results
    metrics: dict[str, Any] = {}
    if hasattr(results, "box"):
        metrics["mAP50"] = float(results.box.map50)
        metrics["mAP50_95"] = float(results.box.map)
        metrics["precision"] = float(results.box.mp)
        metrics["recall"] = float(results.box.mr)
        # Per-class
        if hasattr(results.box, "ap_class_index") and results.box.ap_class_index is not None:
            metrics["per_class"] = {}
            for i, cls_idx in enumerate(results.box.ap_class_index):
                cls_name = names[cls_idx] if cls_idx < len(names) else f"class_{cls_idx}"
                metrics["per_class"][cls_name] = {
                    "mAP50": float(results.box.ap50[i]) if i < len(results.box.ap50) else 0.0,
                    "mAP50_95": float(results.box.ap[i]) if i < len(results.box.ap) else 0.0,
                }

    return metrics


# ---------------------------------------------------------------------------
# Torchvision evaluation path
# ---------------------------------------------------------------------------


def load_torchvision_model(
    checkpoint_path: str,
    model_name: str,
    num_classes: int,
    device: torch.device,
) -> torch.nn.Module:
    """Load a torchvision detection model from checkpoint."""
    console.print(f"Loading [bold]{model_name}[/bold] from {checkpoint_path}...")

    model = get_model(model_name=model_name, num_classes=num_classes, pretrained=False)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        if "epoch" in checkpoint:
            console.print(f"  Loaded from epoch {checkpoint['epoch']}")
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()
    console.print("  ✓ Model loaded")
    return model


@torch.no_grad()
def evaluate_torchvision(
    model: torch.nn.Module,
    image_dir: str | Path,
    annotation_dir: str | Path,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    score_threshold: float,
    iou_threshold: float,
    use_soft_nms: bool,
    output_dir: Path,
    save_predictions: bool,
) -> dict[str, Any]:
    """Evaluate a torchvision model and return metrics."""
    from torch.utils.data import DataLoader

    from visdrone_toolkit.dataset import VisDroneDataset
    from visdrone_toolkit.soft_nms_utils import apply_soft_nms_per_class

    dataset = VisDroneDataset(
        image_dir=str(image_dir),
        annotation_dir=str(annotation_dir),
        filter_ignored=True,
        filter_crowd=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=(device.type == "cuda"),
    )

    all_preds: list[dict[str, torch.Tensor]] = []
    all_targets: list[dict[str, torch.Tensor]] = []
    t0 = time.time()

    for images, targets in loader:
        for img, tgt in zip(images, targets):
            pred = model([img.to(device)])[0]
            mask = pred["scores"] >= score_threshold
            pred = {
                k: v[mask]
                for k, v in pred.items()
                if isinstance(v, torch.Tensor) and v.shape[0] == mask.shape[0]
            }

            if use_soft_nms and len(pred.get("boxes", [])) > 0:
                b, lbl, s = apply_soft_nms_per_class(
                    pred["boxes"].cpu(),
                    pred["labels"].cpu(),
                    pred["scores"].cpu(),
                    iou_threshold=0.5,
                    sigma=0.5,
                    score_threshold=score_threshold,
                )
                pred = {"boxes": b, "labels": lbl, "scores": s}

            all_preds.append(pred)
            all_targets.append(tgt)

    elapsed = time.time() - t0
    n = len(all_preds)

    # Overall metrics
    overall = compute_metrics(all_preds, all_targets, iou_threshold)

    # Per-class metrics
    per_class = _per_class_metrics(all_preds, all_targets, iou_threshold)

    # Try mAP via pycocotools
    map50: float | None = None
    map50_95: float | None = None
    import contextlib

    with contextlib.suppress(Exception):
        map50, map50_95 = _coco_map(all_preds, all_targets)

    metrics: dict[str, Any] = {
        "precision": overall["precision"],
        "recall": overall["recall"],
        "f1": overall["f1"],
        "mAP50": map50,
        "mAP50_95": map50_95,
        "per_class": per_class,
        "num_images": n,
        "fps": n / elapsed if elapsed > 0 else 0,
        "avg_ms": elapsed / n * 1000 if n > 0 else 0,
    }

    if save_predictions:
        _save_json(all_preds, all_targets, output_dir / "predictions.json")

    return metrics


def _per_class_metrics(
    predictions: list[dict], targets: list[dict], iou_threshold: float
) -> dict[str, dict[str, float]]:
    """Per-class P/R/F1."""
    from visdrone_toolkit.utils import box_iou

    all_classes: set[int] = set()
    for t in targets:
        all_classes.update(t["labels"].cpu().tolist())

    result: dict[str, dict[str, float]] = {}
    for cls in sorted(all_classes):
        tp = fp = fn = 0
        for pred, tgt in zip(predictions, targets):
            pm = pred.get("labels", torch.tensor([])).cpu() == cls
            tm = tgt["labels"].cpu() == cls
            pb = pred.get("boxes", torch.zeros(0, 4)).cpu()[pm]
            tb = tgt["boxes"].cpu()[tm]

            if len(pb) == 0 and len(tb) == 0:
                continue
            if len(pb) == 0:
                fn += len(tb)
                continue
            if len(tb) == 0:
                fp += len(pb)
                continue

            ious = box_iou(pb, tb)
            matched: set[int] = set()
            for i in range(len(pb)):
                best_iou, best_idx = ious[i].max(dim=0)
                if best_iou >= iou_threshold and best_idx.item() not in matched:
                    tp += 1
                    matched.add(best_idx.item())
                else:
                    fp += 1
            fn += len(tb) - len(matched)

        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        name = VISDRONE_CLASSES[cls] if cls < len(VISDRONE_CLASSES) else f"class_{cls}"
        result[name] = {"precision": prec, "recall": rec, "f1": f1}

    return result


def _coco_map(predictions: list[dict], targets: list[dict]) -> tuple[float, float]:
    """Compute mAP@0.5 and mAP@0.5:0.95 via pycocotools."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    gt_anns: list[dict] = []
    dt_anns: list[dict] = []
    images: list[dict] = []
    ann_id = 1

    for img_id, (pred, tgt) in enumerate(zip(predictions, targets)):
        images.append({"id": img_id})
        for box, label in zip(tgt["boxes"].cpu().numpy(), tgt["labels"].cpu().numpy()):
            x1, y1, x2, y2 = box
            gt_anns.append(
                {
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": int(label),
                    "iscrowd": 0,
                    "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                    "area": float((x2 - x1) * (y2 - y1)),
                }
            )
            ann_id += 1

        boxes = pred.get("boxes", torch.zeros(0, 4)).cpu().numpy()
        scores = pred.get("scores", torch.zeros(0)).cpu().numpy()
        labels = pred.get("labels", torch.zeros(0, dtype=torch.long)).cpu().numpy()
        for box, score, label in zip(boxes, scores, labels):
            x1, y1, x2, y2 = box
            dt_anns.append(
                {
                    "image_id": img_id,
                    "category_id": int(label),
                    "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                    "score": float(score),
                }
            )

    cats = [{"id": i, "name": n} for i, n in enumerate(VISDRONE_CLASSES)]
    coco_gt = COCO()
    coco_gt.dataset = {"images": images, "annotations": gt_anns, "categories": cats}
    coco_gt.createIndex()

    if not dt_anns:
        return 0.0, 0.0

    coco_dt = coco_gt.loadRes(dt_anns)
    ev = COCOeval(coco_gt, coco_dt, "bbox")
    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    return float(ev.stats[1]), float(ev.stats[0])  # AP@0.5, AP@0.5:0.95


def _save_json(predictions: list[dict], targets: list[dict], path: Path) -> None:
    """Save predictions to JSON."""
    data = []
    for i, (p, t) in enumerate(zip(predictions, targets)):
        data.append(
            {
                "image_id": i,
                "predictions": {
                    "boxes": p.get("boxes", torch.zeros(0, 4)).cpu().numpy().tolist(),
                    "labels": p.get("labels", torch.zeros(0)).cpu().numpy().tolist(),
                    "scores": p.get("scores", torch.zeros(0)).cpu().numpy().tolist(),
                },
                "ground_truth": {
                    "boxes": t["boxes"].cpu().numpy().tolist(),
                    "labels": t["labels"].cpu().numpy().tolist(),
                },
            }
        )
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    console.print(f"  ✓ Predictions saved to {path}")


# ---------------------------------------------------------------------------
# Table printing
# ---------------------------------------------------------------------------


def print_metrics_table(model_name: str, metrics: dict[str, Any]) -> None:
    """Print a rich table of evaluation results."""
    console.rule(f"[bold]Evaluation Results — {model_name}[/bold]")

    # Summary table
    summary = Table(title="Summary", show_header=True, header_style="bold magenta")
    summary.add_column("Metric", style="cyan")
    summary.add_column("Value", justify="right")

    def fmt(v: Any) -> str:
        if v is None:
            return "[dim]N/A[/dim]"
        if isinstance(v, float):
            return f"{v:.4f}"
        return str(v)

    for key in ("mAP50", "mAP50_95", "precision", "recall", "f1"):
        if key in metrics:
            label = {"mAP50_95": "mAP@0.5:0.95", "mAP50": "mAP@0.5"}.get(key, key.title())
            summary.add_row(label, fmt(metrics[key]))
    for key in ("fps", "avg_ms", "num_images"):
        if key in metrics:
            label = {"fps": "FPS", "avg_ms": "ms/image", "num_images": "Images"}.get(key, key)
            summary.add_row(label, fmt(metrics[key]))

    console.print(summary)

    # Per-class table
    per_class = metrics.get("per_class", {})
    if per_class:
        cls_table = Table(title="Per-Class Metrics", show_header=True, header_style="bold cyan")
        cls_table.add_column("Class", style="white")
        has_map = any("mAP50" in v for v in per_class.values())
        if has_map:
            cls_table.add_column("mAP@0.5", justify="right")
            cls_table.add_column("mAP@0.5:0.95", justify="right")
        else:
            cls_table.add_column("Precision", justify="right")
            cls_table.add_column("Recall", justify="right")
            cls_table.add_column("F1", justify="right")

        for cls_name, cls_m in sorted(per_class.items()):
            if has_map:
                cls_table.add_row(
                    cls_name,
                    f"{cls_m.get('mAP50', 0):.4f}",
                    f"{cls_m.get('mAP50_95', 0):.4f}",
                )
            else:
                cls_table.add_row(
                    cls_name,
                    f"{cls_m.get('precision', 0):.4f}",
                    f"{cls_m.get('recall', 0):.4f}",
                    f"{cls_m.get('f1', 0):.4f}",
                )

        console.print(cls_table)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device_str = args.device
    device = torch.device(device_str)

    console.print("\n[bold green]VisDrone Evaluation[/bold green]")
    console.print(f"  Model: [bold]{args.model}[/bold]")
    console.print(f"  Checkpoint: {args.checkpoint}")
    console.print(f"  Device: {device}\n")

    if _is_yolo_model(args.model):
        metrics = evaluate_yolo(
            checkpoint_path=args.checkpoint,
            image_dir=args.image_dir,
            annotation_dir=args.annotation_dir,
            num_classes=args.num_classes,
            device=device_str,
            output_dir=output_dir,
            model_name=args.model,
        )
    else:
        model = load_torchvision_model(
            checkpoint_path=args.checkpoint,
            model_name=args.model,
            num_classes=args.num_classes,
            device=device,
        )
        metrics = evaluate_torchvision(
            model=model,
            image_dir=args.image_dir,
            annotation_dir=args.annotation_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
            score_threshold=args.score_threshold,
            iou_threshold=args.iou_threshold,
            use_soft_nms=args.soft_nms,
            output_dir=output_dir,
            save_predictions=args.save_predictions,
        )

    print_metrics_table(args.model, metrics)

    # Save JSON summary
    metrics_path = output_dir / "metrics.json"
    serializable: dict[str, Any] = {
        k: (float(v) if isinstance(v, (float, np.floating)) else v)
        for k, v in metrics.items()
        if k != "per_class"
    }
    if "per_class" in metrics:
        serializable["per_class"] = {
            cls: {mk: float(mv) for mk, mv in mv_dict.items()}
            for cls, mv_dict in metrics["per_class"].items()
        }
    with open(metrics_path, "w") as f:
        json.dump(serializable, f, indent=2)
    console.print(f"\n✓ Metrics saved to [bold]{metrics_path}[/bold]")


if __name__ == "__main__":
    main()
