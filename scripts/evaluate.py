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

_YOLO_PREFIXES = ("yolo",)


def _is_yolo_model(name: str) -> bool:
    return name.lower().startswith(_YOLO_PREFIXES)


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
    
    # 🆕 NEW: Small object detection metrics
    parser.add_argument(
        "--small-object-threshold", 
        type=float, 
        default=32, 
        help="Max area (pixels) for small object classification"
    )
    parser.add_argument(
        "--benchmark", 
        action="store_true", 
        help="Run inference speed benchmark with percentiles"
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# 🆕 NEW: Small object detection metrics
# ---------------------------------------------------------------------------

def compute_small_object_metrics(
    predictions: list[dict], 
    targets: list[dict], 
    iou_threshold: float,
    small_area_threshold: int = 32
) -> dict[str, float]:
    """
    Compute mAP specifically for small objects (< 32x32 pixels).
    
    Args:
        predictions: List of prediction dicts with boxes, labels, scores
        targets: List of target dicts with boxes, labels
        iou_threshold: IoU threshold for matching
        small_area_threshold: Max area (pixels) for small object
    
    Returns:
        Dictionary with small object metrics
    """
    from visdrone_toolkit.utils import box_iou
    
    small_tp = 0
    small_fp = 0
    small_fn = 0
    total_small_gt = 0
    
    for pred, tgt in zip(predictions, targets):
        # Get ground truth boxes and filter for small objects
        gt_boxes = tgt["boxes"].cpu().numpy()
        gt_labels = tgt["labels"].cpu().numpy()
        
        # Calculate areas
        gt_areas = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])
        small_gt_mask = gt_areas < (small_area_threshold * small_area_threshold)
        small_gt_boxes = gt_boxes[small_gt_mask]
        small_gt_labels = gt_labels[small_gt_mask]
        total_small_gt += len(small_gt_boxes)
        
        if len(small_gt_boxes) == 0:
            continue
            
        # Get predictions
        pred_boxes = pred.get("boxes", torch.zeros(0, 4)).cpu().numpy()
        pred_scores = pred.get("scores", torch.zeros(0)).cpu().numpy()
        pred_labels = pred.get("labels", torch.zeros(0, dtype=torch.long)).cpu().numpy()
        
        # Filter predictions by matching class and confidence
        matched_gt = set()
        for i, (pb, ps, pl) in enumerate(zip(pred_boxes, pred_scores, pred_labels)):
            # Find matching ground truth with same class
            matching_gt = [j for j, (gb, gl) in enumerate(zip(small_gt_boxes, small_gt_labels)) 
                          if gl == pl and j not in matched_gt]
            
            if not matching_gt:
                small_fp += 1
                continue
                
            # Compute IoU with matching ground truths
            pb_tensor = torch.tensor(pb).unsqueeze(0)
            gt_tensor = torch.tensor(small_gt_boxes[matching_gt])
            ious = box_iou(pb_tensor, gt_tensor)
            best_iou, best_idx = ious.max(dim=1)
            
            if best_iou >= iou_threshold:
                small_tp += 1
                matched_gt.add(matching_gt[best_idx.item()])
            else:
                small_fp += 1
        
        small_fn += len(small_gt_boxes) - len(matched_gt)
    
    # Compute metrics
    small_precision = small_tp / (small_tp + small_fp) if (small_tp + small_fp) > 0 else 0.0
    small_recall = small_tp / (small_tp + small_fn) if (small_tp + small_fn) > 0 else 0.0
    small_f1 = 2 * small_precision * small_recall / (small_precision + small_recall) if (small_precision + small_recall) > 0 else 0.0
    
    return {
        "small_objects_precision": small_precision,
        "small_objects_recall": small_recall,
        "small_objects_f1": small_f1,
        "small_objects_gt_count": total_small_gt,
    }


# ---------------------------------------------------------------------------
# 🆕 NEW: Inference speed benchmark with percentiles
# ---------------------------------------------------------------------------

def benchmark_inference_speed(
    model: torch.nn.Module,
    dataset,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    num_runs: int = 100,
) -> dict[str, float]:
    """
    Benchmark inference speed with percentile statistics.
    
    Args:
        model: PyTorch model
        dataset: VisDroneDataset
        batch_size: Batch size for evaluation
        num_workers: Number of data loader workers
        device: Device to run on
        num_runs: Number of runs for benchmark
    
    Returns:
        Dictionary with speed metrics (ms/image, p50, p95, p99)
    """
    from torch.utils.data import DataLoader
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=(device.type == "cuda"),
    )
    
    latencies = []
    model.eval()
    
    with torch.no_grad():
        for i, (images, _) in enumerate(loader):
            if i >= num_runs // batch_size:
                break
                
            # Warmup
            if i == 0:
                for img in images:
                    _ = model([img.to(device)])
                continue
            
            # Measure inference time
            start = time.perf_counter()
            for img in images:
                _ = model([img.to(device)])
            end = time.perf_counter()
            
            # Per-image latency
            batch_latency = (end - start) / len(images) * 1000  # ms
            latencies.extend([batch_latency] * len(images))
    
    if not latencies:
        return {"fps": 0, "avg_ms": 0, "p50_ms": 0, "p95_ms": 0, "p99_ms": 0}
    
    latencies = np.array(latencies)
    return {
        "fps": 1000 / np.mean(latencies) if np.mean(latencies) > 0 else 0,
        "avg_ms": np.mean(latencies),
        "p50_ms": np.percentile(latencies, 50),
        "p95_ms": np.percentile(latencies, 95),
        "p99_ms": np.percentile(latencies, 99),
        "std_ms": np.std(latencies),
        "min_ms": np.min(latencies),
        "max_ms": np.max(latencies),
    }


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
    benchmark: bool = False,
) -> dict[str, Any]:
    """Evaluate a YOLO model using the Ultralytics val engine."""
    try:
        from ultralytics import YOLO as UltralyticsYOLO
    except ImportError as err:
        raise ImportError("pip install ultralytics>=8.0.0") from err

    import tempfile

    from visdrone_toolkit.yolo_trainer import _VISDRONE_CLASSES, YOLOTrainer

    console.print("\n[bold cyan]YOLO evaluation — using Ultralytics val engine[/bold cyan]")

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
            image_dir,
            annotation_dir,
        )

        model = UltralyticsYOLO(str(checkpoint_path))
        results = model.val(
            data=str(dataset_yaml),
            device=device,
            split="val",
            save_json=False,
            project=str(output_dir.resolve()),
            name="yolo_eval",
            exist_ok=True,
        )

    metrics: dict[str, Any] = {}
    if hasattr(results, "box"):
        metrics["mAP50"] = float(results.box.map50)
        metrics["mAP50_95"] = float(results.box.map)
        metrics["precision"] = float(results.box.mp)
        metrics["recall"] = float(results.box.mr)
        
        if hasattr(results.box, "ap_class_index") and results.box.ap_class_index is not None:
            metrics["per_class"] = {}
            for i, cls_idx in enumerate(results.box.ap_class_index):
                cls_name = names[cls_idx] if cls_idx < len(names) else f"class_{cls_idx}"
                metrics["per_class"][cls_name] = {
                    "mAP50": float(results.box.ap50[i]) if i < len(results.box.ap50) else 0.0,
                    "mAP50_95": float(results.box.ap[i]) if i < len(results.box.ap) else 0.0,
                }
    
    # 🆕 Speed benchmark for YOLO
    if benchmark:
        console.print("\n[bold yellow]Running inference speed benchmark...[/bold yellow]")
        # Simple benchmark for YOLO
        import time
        latencies = []
        for _ in range(50):
            start = time.perf_counter()
            model.predict(source=np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8), verbose=False)
            end = time.perf_counter()
            latencies.append((end - start) * 1000)
        
        if latencies:
            latencies = np.array(latencies)
            metrics["benchmark"] = {
                "avg_ms": np.mean(latencies),
                "p50_ms": np.percentile(latencies, 50),
                "p95_ms": np.percentile(latencies, 95),
                "p99_ms": np.percentile(latencies, 99),
                "fps": 1000 / np.mean(latencies) if np.mean(latencies) > 0 else 0,
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
    small_object_threshold: int = 32,
    benchmark: bool = False,
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
    
    # 🆕 Collect per-image latencies
    per_image_latencies = []

    for images, targets in loader:
        for img, tgt in zip(images, targets):
            # Measure per-image inference time
            start = time.perf_counter()
            pred = model([img.to(device)])[0]
            end = time.perf_counter()
            per_image_latencies.append((end - start) * 1000)  # ms
            
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

    # mAP via pycocotools
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
    
    # 🆕 Small object metrics
    console.print("\n[bold yellow]Computing small object metrics...[/bold yellow]")
    small_metrics = compute_small_object_metrics(
        all_preds, all_targets, iou_threshold, small_object_threshold
    )
    metrics.update(small_metrics)
    
    # 🆕 Inference speed benchmark
    if benchmark and per_image_latencies:
        latencies = np.array(per_image_latencies)
        metrics["benchmark"] = {
            "avg_ms": np.mean(latencies),
            "p50_ms": np.percentile(latencies, 50),
            "p95_ms": np.percentile(latencies, 95),
            "p99_ms": np.percentile(latencies, 99),
            "std_ms": np.std(latencies),
            "min_ms": np.min(latencies),
            "max_ms": np.max(latencies),
            "fps": 1000 / np.mean(latencies) if np.mean(latencies) > 0 else 0,
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
    return float(ev.stats[1]), float(ev.stats[0])


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
    
    # 🆕 Small object metrics
    for key in ("small_objects_precision", "small_objects_recall", "small_objects_f1"):
        if key in metrics:
            label = {
                "small_objects_precision": "Small Object Precision",
                "small_objects_recall": "Small Object Recall",
                "small_objects_f1": "Small Object F1",
            }.get(key, key)
            summary.add_row(label, fmt(metrics[key]))
    
    for key in ("fps", "avg_ms", "num_images"):
        if key in metrics:
            label = {"fps": "FPS", "avg_ms": "ms/image", "num_images": "Images"}.get(key, key)
            summary.add_row(label, fmt(metrics[key]))
    
    # 🆕 Benchmark percentiles
    if "benchmark" in metrics:
        bench = metrics["benchmark"]
        summary.add_section()
        summary.add_row("[bold cyan]Benchmark (ms)[/bold cyan]", "")
        summary.add_row("  p50", fmt(bench.get("p50_ms", 0)))
        summary.add_row("  p95", fmt(bench.get("p95_ms", 0)))
        summary.add_row("  p99", fmt(bench.get("p99_ms", 0)))

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
            benchmark=args.benchmark,
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
            small_object_threshold=args.small_object_threshold,
            benchmark=args.benchmark,
        )

    print_metrics_table(args.model, metrics)

    # Save JSON summary
    metrics_path = output_dir / "metrics.json"
    serializable: dict[str, Any] = {
        k: (float(v) if isinstance(v, (float, np.floating)) else v)
        for k, v in metrics.items()
        if k not in ("per_class", "benchmark")
    }
    if "per_class" in metrics:
        serializable["per_class"] = {
            cls: {mk: float(mv) for mk, mv in mv_dict.items()}
            for cls, mv_dict in metrics["per_class"].items()
        }
    if "benchmark" in metrics:
        serializable["benchmark"] = {
            k: float(v) for k, v in metrics["benchmark"].items()
        }
    with open(metrics_path, "w") as f:
        json.dump(serializable, f, indent=2)
    console.print(f"\n✓ Metrics saved to [bold]{metrics_path}[/bold]")


if __name__ == "__main__":
    main()